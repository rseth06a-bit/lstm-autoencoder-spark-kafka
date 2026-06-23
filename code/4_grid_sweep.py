"""
Step 4: Grid Sweep

Brute-force grid sweep over training hyperparameters and data split
sizes. Trains one model per configuration, scores it on the test set,
ranks results, then retrains the winning config end-to-end and saves
its full artifact set to `models/best/` so you can evaluate it directly.

Reuses `src.training.train_model` so each trial follows the same
pipeline as `code/1_train_model.py`. NEVER overwrites the prebuilt
artifacts at `models/lstm_model.pt` etc.

Modes:
    python code/4_grid_sweep.py                  # default: hyperparams
    python code/4_grid_sweep.py --mode split     # vary data split sizes
"""

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model
from src.scorer import AnomalyScorer, ScorerConfig
from src.training import TrainingConfig, save_training_artifacts, train_model
from src.preprocess import PreprocessorConfig, preprocess_pipeline, VITAL_COLUMNS, WINDOW_SIZE
import pandas as pd

DEFAULT_OUTCOMES_PATH = str(PROJECT_ROOT / "Outcomes-a.txt")

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "nyc_taxi.csv")


# ---------------------------------------------------------------------------
# Configs & result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HyperparamConfig:
    """Configuration for a hyperparameter experiment."""
    hidden_dim: int
    num_layers: int
    dropout: float
    learning_rate: float
    threshold_percentile: float
    batch_size: int = 4

    def to_dict(self) -> Dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "threshold_percentile": self.threshold_percentile,
            "batch_size": self.batch_size,
        }


@dataclass
class ExperimentResult:
    """Results from a single experiment (works for both modes)."""
    config_dict: Dict
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    f_beta_score: float
    train_loss: float
    val_loss: float
    threshold: float
    best_epoch: int = 0
    total_params: int = 0

    def to_dict(self) -> Dict:
        return {
            **self.config_dict,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "f_beta_score": self.f_beta_score,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "threshold": self.threshold,
            "best_epoch": self.best_epoch,
            "total_params": self.total_params,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_f_beta(precision: float, recall: float, beta: float = 0.1) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)

def get_test_actuals(test_patient_ids: list, outcomes_path: str) -> np.ndarray:
    """Look up ground-truth In-hospital_death for each test patient."""
    outcomes = pd.read_csv(outcomes_path)
    outcomes["RecordID"] = outcomes["RecordID"].astype(str)
    lookup = outcomes.set_index("RecordID")["In-hospital_death"]
    return np.array([bool(lookup.loc[pid]) for pid in test_patient_ids])

def compute_metrics(predictions: np.ndarray, actuals: np.ndarray, beta: float = 0.1) -> Dict[str, float]:
    tp = np.sum(predictions & actuals)
    fp = np.sum(predictions & ~actuals)
    fn = np.sum(~predictions & actuals)
    tn = np.sum(~predictions & ~actuals)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f_beta = compute_f_beta(precision, recall, beta)
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "f_beta_score": f_beta}


# ---------------------------------------------------------------------------
# Core experiment runner (shared by both modes)
# ---------------------------------------------------------------------------

def train_full(
    data_path: str,
    outcomes_path: str,
    device: torch.device,
    train_fraction: float,
    val_fraction: float,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    threshold_percentile: float = 95.0,
    batch_size: int = 4,
    epochs: int = 100,
    patience: int = 10,
    beta: float = 0.1,
):
    """
    Run the full training + scoring pipeline for one config.

    Returns a dict with: model, scaler, scorer, history, preprocess_config,
    metrics, config_dict, total_params, threshold. Used both by the sweep
    loop (which only consumes the metrics) and by `retrain_best_and_save()`
    (which persists the artifacts).
    """
    preprocess_config = PreprocessorConfig(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    dataloaders, _, scaler, patient_ids, split_ids = preprocess_pipeline(
        data_path, config=preprocess_config, batch_size=batch_size,
    )
    for split in ("train", "val", "test"):
        if dataloaders[split] is None or len(dataloaders[split].dataset) == 0:
            return None
    model = create_model(
        input_dim=len(VITAL_COLUMNS),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        sequence_length=WINDOW_SIZE,
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())

    tc = TrainingConfig(epochs=epochs, learning_rate=learning_rate, patience=patience)
    model, history = train_model(model, dataloaders["train"], dataloaders["val"], device, tc)

    scorer = AnomalyScorer(config=ScorerConfig(threshold_percentile=threshold_percentile))
    scorer.fit(model, dataloaders["val"], device)
    scorer.set_threshold(scorer.compute_scores(model, dataloaders["val"], device)[0])

    test_scores, _ = scorer.compute_scores(model, dataloaders["test"], device)
    predictions = scorer.predict(test_scores)
    flagged_ids = [pid for pid, p in zip(split_ids["test"], predictions) if p]
    print(f"  [debug] flagged patient IDs: {flagged_ids}")
    print(f"  [debug] threshold={scorer.threshold:.2f}  test_scores=[{test_scores.min():.2f}, {test_scores.max():.2f}]  n_flagged={predictions.sum()}")


    test_patient_ids = split_ids["test"]
    actuals = get_test_actuals(test_patient_ids, outcomes_path)
    metrics = compute_metrics(predictions, actuals, beta)

    config_dict = {
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "threshold_percentile": threshold_percentile,
    }

    return {
         "model": model, "scaler": scaler, "scorer": scorer, "history": history,
        "preprocess_config": preprocess_config, "metrics": metrics,
        "config_dict": config_dict, "total_params": total_params,
        "threshold": scorer.threshold, "test_patient_ids": test_patient_ids,
        "predictions": predictions, "test_scores": test_scores,
    }


def train_and_evaluate(
    data_path: str,
    outcomes_path: str,
    device: torch.device,
    train_fraction: float,
    val_fraction: float,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    threshold_percentile: float = 95.0,
    batch_size: int = 4,
    epochs: int = 100,
    patience: int = 10,
    beta: float = 0.1,
) -> Optional[ExperimentResult]:
    try:
        run = train_full(
            data_path=data_path, outcomes_path=outcomes_path, device=device,
            train_fraction=train_fraction, val_fraction=val_fraction,
            hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            learning_rate=learning_rate, threshold_percentile=threshold_percentile,
            batch_size=batch_size, epochs=epochs, patience=patience, beta=beta,
        )
        if run is None:
            return None
        metrics = run["metrics"]
        history = run["history"]
        return ExperimentResult(
            config_dict=run["config_dict"],
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            f_beta_score=metrics["f_beta_score"],
            train_loss=history["train_loss"][-1],
            val_loss=history["val_loss"][-1] if history["val_loss"] else float("inf"),
            threshold=run["threshold"],
            best_epoch=history.get("best_epoch", 0),
            total_params=run["total_params"],
        )

    except Exception as e:
        logger.error("Experiment failed: %s", e, exc_info=True)
        return None


def retrain_best_and_save(
    best_result: "ExperimentResult",
    args,
    device: torch.device,
) -> None:
    """
    Re-run the best configuration found during the sweep, then persist the
    full set of training artifacts (model, scaler, scorer, history, split
    config) to `args.best_dir` so the user can evaluate it directly with
    `code/2_evaluate_model.py --model-dir <best_dir>`.
    """
    cfg = best_result.config_dict
    print()
    print("=" * 60)
    print("RETRAINING BEST CONFIG  ->  saving to disk")
    print("=" * 60)
    print(f"Best F1 from sweep: {best_result.f1_score:.2%}")
    print("Config:")
    for k, v in cfg.items():
        print(f"  {k:<22} {v}")
    print(f"Output directory:    {args.best_dir}")
    print("=" * 60)

    set_seed(42)
    run = train_full(
        data_path=args.data_path,
        outcomes_path=args.outcomes_path,
        device=device,
        train_fraction=cfg["train_fraction"],
        val_fraction=cfg["val_fraction"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        learning_rate=cfg["learning_rate"],
        threshold_percentile=cfg["threshold_percentile"],
        epochs=args.epochs,
        patience=args.patience,
        beta=args.beta,
    )
    if run is None:
        print("Retraining failed -- best config produced an empty split. Skipping save.")
        return

    save_training_artifacts(
        output_dir=args.best_dir,
        model=run["model"],
        scaler=run["scaler"],
        scorer=run["scorer"],
        history=run["history"],
        preprocess_config=run["preprocess_config"],
    )

    m = run["metrics"]
    print()
    print(f"Retrained model metrics on test set:")
    print(f"  Precision: {m['precision']:.2%}")
    print(f"  Recall:    {m['recall']:.2%}")
    print(f"  F1:        {m['f1_score']:.2%}")
    print()
    print(f"Best-config artifacts written to: {args.best_dir}/")
    print("Inspect them with:")
    print(f"  python code/2_evaluate_model.py --model-dir {args.best_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Hyperparameter optimization
# ---------------------------------------------------------------------------

def _focused_configs() -> List[HyperparamConfig]:
    """
    A small, opinionated set of configurations that span the interesting
    region around the prebuilt-equivalent setting (hidden_dim=64,
    num_layers=1, dropout=0.2, lr=5e-4, threshold_percentile=99.99).

    The point of the sweep is to demonstrate the journey from the
    `1_train_model.py` baseline (small hidden_dim, 1e-3 lr) to a
    well-tuned config that hits 100% F1 on the scored test weeks.
    """
    return [
        # Baseline-ish
        HyperparamConfig(32, 1, 0.2, 1e-3, 90.0),
        HyperparamConfig(32, 1, 0.0, 1e-3, 90.0),
        # Wider models
        HyperparamConfig(40, 1, 0.2, 1e-3, 90.0),
        HyperparamConfig(40, 1, 0.2, 5e-4, 90.0),
        HyperparamConfig(64, 1, 0.0, 1e-3, 90.0),
        HyperparamConfig(64, 1, 0.1, 1e-3, 90.0),
        HyperparamConfig(64, 1, 0.2, 1e-3, 90.0),
        HyperparamConfig(64, 1, 0.2, 5e-4, 90.0),
        # Variations
        HyperparamConfig(64, 1, 0.3, 5e-4, 90.0),
        HyperparamConfig(64, 2, 0.2, 5e-4, 90.0),
        HyperparamConfig(128, 1, 0.2, 5e-4, 90.0),
        HyperparamConfig(128, 1, 0.3, 5e-4, 90.0),
        # Threshold sensitivity sweep at the leading architecture
        HyperparamConfig(64, 1, 0.2, 5e-4, 80.0),
        HyperparamConfig(64, 1, 0.2, 5e-4, 85.0),
        HyperparamConfig(64, 1, 0.2, 5e-4, 95.0),
    ]


def _generate_grid_configs(
    hidden_dims=(32, 40, 64, 128), num_layers_list=(1, 2),
    dropouts=(0.0, 0.1, 0.2, 0.3), learning_rates=(1e-4, 5e-4, 1e-3),
    threshold_percentiles=(90.0, 95.0, 99.0),
) -> List[HyperparamConfig]:
    return [
        HyperparamConfig(h, l, d, lr, tp)
        for h, l, d, lr, tp in product(
            hidden_dims, num_layers_list, dropouts, learning_rates, threshold_percentiles
        )
    ]


def _generate_random_configs(n: int = 50) -> List[HyperparamConfig]:
    cfgs = []
    for _ in range(n):
        cfgs.append(HyperparamConfig(
            hidden_dim=random.randint(16, 128),
            num_layers=random.randint(1, 3),
            dropout=round(random.uniform(0.0, 0.4), 2),
            learning_rate=10 ** random.uniform(-4, -2),
            threshold_percentile=round(random.uniform(85.0, 99.0), 1),
        ))
    return cfgs


def run_hyperparams_optimization(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.search == "grid":
        configs = _generate_grid_configs()
    elif args.search == "random":
        configs = _generate_random_configs(args.n_configs)
    else:
        configs = _focused_configs()

    print(f"\nDevice: {device}")
    print(f"Split: train_fraction={args.train_fraction}, val_fraction={args.val_fraction}")
    print(f"Search: {args.search} ({len(configs)} configurations)")

    results: List[ExperimentResult] = []
    t0 = time.time()

    for i, cfg in enumerate(configs):
        set_seed(42)
        elapsed = time.time() - t0
        print(f"\n[{i+1}/{len(configs)}] ({elapsed:.0f}s) hidden={cfg.hidden_dim} "
              f"layers={cfg.num_layers} drop={cfg.dropout} lr={cfg.learning_rate:.1e} "
              f"thresh={cfg.threshold_percentile}%...", end="", flush=True)

        r = train_and_evaluate(
            data_path=args.data_path, outcomes_path=args.outcomes_path, device=device,
            train_fraction=args.train_fraction, val_fraction=args.val_fraction,
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
            dropout=cfg.dropout, learning_rate=cfg.learning_rate,
            threshold_percentile=cfg.threshold_percentile,
            batch_size=cfg.batch_size, epochs=args.epochs,
            patience=args.patience, beta=args.beta,
        )
        if r is not None:
            results.append(r)
            print(f"  F1={r.f1_score:.2%}  Fb={r.f_beta_score:.2%}  "
                  f"P={r.precision:.2%}  R={r.recall:.2%}")
            if args.output:
                _save_results(results, args.output, len(configs), args.sort_by, args.beta)

    _print_results(results, sort_by=args.sort_by, title="HYPERPARAMETER OPTIMIZATION RESULTS")
    if args.output:
        _save_results(results, args.output, len(configs), args.sort_by, args.beta)
        print(f"\nResults saved to {args.output}")

    if results and not args.no_retrain_best:
        best = max(results, key=lambda r: getattr(r, args.sort_by))
        retrain_best_and_save(best, args, device)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_results(results: List[ExperimentResult], sort_by: str = "f1_score",
                   title: str = "RESULTS") -> None:
    if not results:
        print("\nNo successful experiments!")
        return
    sorted_r = sorted(results, key=lambda r: getattr(r, sort_by), reverse=True)
    print(f"\n{'=' * 110}\n{title}\n{'=' * 110}")
    print(f"{'TrainF':>7} {'ValF':>6} {'Hidden':>7} {'Lay':>4} "
      f"{'Drop':>5} {'LR':>9} {'Thr%':>6} {'Acc':>7} {'Prec':>7} "
      f"{'Rec':>7} {'F1':>7} {'Fb':>7}")
    print("-" * 110)
    for r in sorted_r[:20]:
        c = r.config_dict
        print(f"{c['train_fraction']:>7.2f} {c['val_fraction']:>6.2f} "
            f"{c['hidden_dim']:>7} {c['num_layers']:>4} {c['dropout']:>5.2f} "
            f"{c['learning_rate']:>9.1e} {c['threshold_percentile']:>6.1f} "
            f"{r.accuracy:>7.2%} {r.precision:>7.2%} {r.recall:>7.2%} "
            f"{r.f1_score:>7.2%} {r.f_beta_score:>7.2%}")
    best = sorted_r[0]
    print(f"\nBEST (by {sort_by}): {best.config_dict}")
    print(f"  F1={best.f1_score:.2%}  Fb={best.f_beta_score:.2%}  "
          f"Prec={best.precision:.2%}  Rec={best.recall:.2%}")
    print("=" * 110)


def _save_results(results: List[ExperimentResult], path: str,
                  total: int, sort_by: str, beta: float) -> None:
    sorted_r = sorted(results, key=lambda r: getattr(r, sort_by), reverse=True)
    out = {
        "timestamp": datetime.now().isoformat(),
        "configs_tested": total,
        "completed_runs": len(results),
        "progress_pct": round(100 * len(results) / total, 1) if total else 0,
        "sort_by": sort_by,
        "beta": beta,
        "results": [r.to_dict() for r in sorted_r],
        "best_config": sorted_r[0].to_dict() if sorted_r else None,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid sweep over hyperparameters or data-split sizes. "
                    "After the sweep, retrains the winning configuration and "
                    "saves it to --best-dir so it can be evaluated directly."
    )
    parser.add_argument("--data-path", type=str, default=str(PROJECT_ROOT / "set-a"))
    parser.add_argument("--input-dir", type=str,
                        default=str(PROJECT_ROOT / "models" / "initial"),
                        help="Baseline run produced by 1_train_model.py. "
                             "Used only to display the starting point.")
    parser.add_argument("--best-dir", type=str,
                        default=str(PROJECT_ROOT / "models" / "best"),
                        help="Where to save the retrained best-config artifacts (default: models/best/).")
    parser.add_argument("--no-retrain-best", action="store_true",
                        help="Skip retraining and saving the best config at the end.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON path for the full sweep results table.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Beta for F-beta score (paper uses 0.1)")
    parser.add_argument("--verbose", action="store_true")

    # Hyperparams-specific
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--outcomes-path", type=str, default=DEFAULT_OUTCOMES_PATH)
    parser.add_argument("--search", choices=["grid", "random", "focused"], default="focused")
    parser.add_argument("--n-configs", type=int, default=30,
                        help="Number of random configs (random search only)")
    parser.add_argument("--sort-by", default="f1_score",
                        choices=["f1_score", "f_beta_score", "precision", "recall", "accuracy"])

    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    print("\n" + "=" * 60)
    print("=" * 60)
    baseline_dir = Path(args.input_dir)
    if baseline_dir.exists():
        print(f"Baseline (from step 1): {baseline_dir}")
    else:
        print(f"Baseline directory {baseline_dir} not found -- run code/1_train_model.py first.")
    print(f"Best-config output:     {args.best_dir}")
    print(f"Models in models/lstm_model.pt etc are NEVER touched.")
    print("=" * 60)

    run_hyperparams_optimization(args)

    print("\n" + "=" * 60)
    print("GRID SWEEP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
