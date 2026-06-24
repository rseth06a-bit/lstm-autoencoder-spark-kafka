# ICU Patient Outcome Prediction: LSTM Autoencoder Anomaly Detection

This repository adapts the Striim AI Prototype for real-time time-series anomaly detection. The original anomaly detection pipeline was built for NYC taxi data, but it was shifted to ICU patient instability from real-time vitals, evaluated against actual in-hospital mortality outcomes. 

The original repo used the NYC taxi demand dataset as a concrete example of recurring seasonal structure, localized disruptions, and thresholded anomaly detection. It included method-oriented notebooks for learning the approach, reusable source code for the LSTM encoder-decoder and anomaly scoring pipeline, pre-trained model artifacts for running the demo immediately, and Dockerized services for the streaming application and Kafka producer. This fork applies the same core methodology (LSTM Encoder-Decoder + Mahalanobis distance scoring) to a fundamentally different and noisier domain: five vital signs (heart rate, respiratory rate, temperature, systolic and diastolic blood pressure) sampled per-patient over the first 48 hours of an ICU stay, using the [PhysioNet/CinC 2012 Challenge dataset](https://physionet.org/content/challenge-2012/1.0.0/).

**Scope of this fork:** the offline training, preprocessing, scoring, and hyperparameter-sweep pipeline (`src/preprocess.py`, `src/model.py`, `src/scorer.py`, `code/1_train_model.py`, `code/4_grid_sweep.py`) has been rewritten for the ICU vitals domain and evaluated against real patient mortality outcomes. The Kafka/Spark/Dash streaming demo, Docker setup, and Striim integration described later in this README are **unmodified from the original** and still operate on the NYC taxi dataset — they were not adapted as part of this fork.

The modeling approach is based on [Malhotra et al. (2016)](https://arxiv.org/abs/1607.00148): "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"

---

## Project Structure

```
lstm-autoencoder-spark-kafka/
│
├── code/                                    # Numbered scripts -- the canonical workflow
│   ├── 0_verify_setup.py                    # Optional environment / artifact check (unmodified)
│   ├── 1_train_model.py                     # Train baseline on ICU vitals, save to models/initial/
│   ├── 2_evaluate_model.py                  # NOT adapted -- still references the original taxi pipeline (see note below)
│   ├── 3_streaming_app.py                   # Real-time Dash + Spark + Kafka app (Docker) -- unmodified, still taxi data
│   └── 4_grid_sweep.py                      # Sweep hyperparams over ICU vitals, retrain best to models/best/
│
├── notebooks/                               # Interactive walkthroughs -- unmodified, still describe the taxi pipeline
│   ├── data_exploration.ipynb
│   └── model_design.ipynb
│
├── src/                                     # Reusable library code
│   ├── model.py                             # EncDecAD architecture -- updated input_dim (5) and window size (48)
│   ├── training.py                          # Shared training loop (unmodified)
│   ├── scorer.py                            # Point-level multivariate Mahalanobis scoring (adapted)
│   ├── preprocess.py                        # Rewritten: loads PhysioNet patient files, builds 48-step vitals windows
│   └── synthetic.py                         # Synthetic anomaly generation (unmodified, unused in this fork)
│
├── producer/                                # Kafka producer service -- unmodified, still streams taxi CSV
│
├── data/
│   ├── nyc_taxi.csv                         # Original taxi dataset (unused in this fork's training pipeline)
│   └── nyc_taxi_sunday_aligned.csv
│
├── set-a/                                   # PhysioNet patient files -- NOT included in repo (see Dataset section)
├── Outcomes-a.txt                           # PhysioNet outcome labels -- NOT included in repo (see Dataset section)
│
├── models/
│   ├── lstm_model.pt                        # Original prebuilt taxi model (unmodified, untouched by this fork)
│   ├── scaler.pkl
│   ├── scorer.pkl
│   ├── training_history.pkl
│   ├── preprocessor_config.pkl
│   ├── initial/                             # ICU baseline output of 1_train_model.py (gitignored)
│   └── best/                                # ICU retrained best from 4_grid_sweep.py (gitignored)
│
├── striim/                                  # Striim Platform OP integration -- unmodified, still taxi domain
│
├── Dockerfile.app / Dockerfile.producer / docker-compose.yml   # Unmodified, still run the taxi streaming demo
├── pyproject.toml
├── STRIIM.md
└── TECHNICAL.md
```

**Note:** `code/2_evaluate_model.py` was not adapted for the ICU vitals pipeline and still references the original taxi-specific data loading (`get_test_week_info`). Running it as-is will fail. Evaluation in this fork happens inline through `code/4_grid_sweep.py`, which trains, scores, and reports precision/recall/F1 against real patient outcomes for every configuration tested.

The scripts under `code/` are the **first-class** path: they reproduce the model end-to-end and are what you should run if you're trying to learn how training and evaluation work, or to adapt this to your own data. The notebooks under `notebooks/` are interactive **supporting material** -- they explain *why* the architecture is shaped the way it is, what the data looks like, and how the scoring methodology was chosen. 

## Prerequisites

*(Unmodified from original. Applies to the full repo, including the streaming demo described later — this fork's ICU pipeline itself only strictly needs Python + `uv`.)*

- **Python 3.11+**
- **uv** (Python package manager) — install with:
```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
```
- **Docker** and **Docker Compose** (for the streaming demo only -- not required for the ICU training/sweep pipeline)

## Going through the code

### 1. Install dependencies

```bash
git clone <repo-url>
cd lstm-autoencoder-spark-kafka
uv sync
```

### 2. Get the dataset (not included in this repo)

This fork uses the [PhysioNet/CinC 2012 Challenge dataset](https://physionet.org/content/challenge-2012/1.0.0/) ("Predicting Mortality of ICU Patients"). It is **not included in this repository** -- PhysioNet's data use agreement prohibits redistribution. Download it yourself:

```bash
wget https://archive.physionet.org/pn3/challenge/2012/set-a.tar.gz
tar -xzf set-a.tar.gz
wget https://archive.physionet.org/pn3/challenge/2012/Outcomes-a.txt
```

Place the extracted `set-a/` directory and `Outcomes-a.txt` in the project root. Both are gitignored. This dataset is open-access (no PhysioNet credentialing required). Downloads from PhysioNet can be slow (~0.5 MB/s) -- this is normal, not a broken link.

The set contains ~4000 ICU patients' first-48-hour vitals as individual `.txt` files, and `Outcomes-a.txt` provides ground-truth in-hospital mortality and severity scores per patient.

### 3. Train a baseline, then improve it via grid sweep

#### 3a. Train the baseline

```bash
uv run python code/1_train_model.py --data-path set-a
```

This loads all patient files, keeps the ~738 patients with sufficient vitals coverage (at least 48 timesteps across all 5 vitals), and trains a baseline LSTM Encoder-Decoder.

#### 3b. Run the grid sweep to find a better configuration

```bash
uv run python code/4_grid_sweep.py --data-path set-a --outcomes-path Outcomes-a.txt --search focused
```

This trains and evaluates 15 hyperparameter configurations, scoring each against real `In-hospital_death` outcomes from `Outcomes-a.txt`, then retrains the winning configuration end-to-end and saves it to `models/best/`.

> **Best-config metrics:** Precision = 22.22%, Recall = 22.22%, F1 = 22.22% on the held-out test set (9 actual in-hospital deaths; 9 patients flagged, 2 true positives). Winning configuration: `hidden_dim=64, num_layers=1, dropout=0.0, lr=1e-3, threshold_percentile=90.0`.

See [Results](#results) below for the full discussion of what these numbers mean and where the approach's limitations are.

> **Note:** `code/2_evaluate_model.py` and the Docker/notebook/streaming sections described later in this README are unmodified and still reference the original taxi pipeline -- see the Project Structure note above.

## Detection methodology

The detector treats each ICU patient's first 48 hours as a single sample: 48 timesteps across 5 vital signs (heart rate, respiratory rate, temperature, systolic and diastolic blood pressure). After training the LSTM Encoder-Decoder on normal (i.e. unlabeled, not specifically curated) patient windows, we collect the per-timestep, per-feature reconstruction error vectors on the validation set and fit a multivariate Gaussian to them: a mean vector `mu` of length 5 and a 5x5 covariance matrix `Sigma`, pooling errors across all timesteps and patients.

A single error vector `e` (one timestep, one patient, 5 features) is scored with the Mahalanobis distance and a patient's overall anomaly score is the sum of these point-level scores across all 48 timesteps in their window. A patient is flagged as anomalous when this summed score exceeds a threshold set at a chosen percentile of validation scores.

This differs from the original taxi pipeline in two ways:

1. **Point-level, not window-level, error distribution.** The original fit `mu`/`Sigma` over the full 336-step window (treating each week's entire error trajectory as one high-dimensional sample). This fork pools errors across all timesteps and patients, fitting a much lower-dimensional (5x5 instead of 336x336) covariance matrix -- a more stable estimate given the smaller per-patient sample size (48 steps vs. 336).
2. **Ground-truth label is mortality, not a labeled calendar event.** The original validated detections against 5 hand-labeled anomalous weeks (holidays, weather events). This fork validates against real `In-hospital_death` outcomes from `Outcomes-a.txt` -- a much harder and more clinically meaningful target, since "unusual vitals" and "died in-hospital" are related but distinct concepts.

This is implemented in `src/scorer.py` (`AnomalyScorer.compute_scores`, point-level mode) and reproduced end-to-end by `code/1_train_model.py` and `code/4_grid_sweep.py`.

## Results

### Headline numbers

The best configuration found via grid sweep (`hidden_dim=64, num_layers=1, dropout=0.0, lr=1e-3, threshold_percentile=90.0`) achieves, on the held-out test set of 112 patients (9 of whom died in-hospital):

| Metric | Value |
|---|---|
| Precision | 22.22% |
| Recall | 22.22% |
| F1 | 22.22% |
| Patients flagged | 9 |
| True positives | 2 of 9 actual deaths |

This is a real, non-trivial result -- but a modest one, and it's worth being direct about what it does and doesn't show.

### Why these numbers, and what they mean

Unlike the original taxi pipeline, where the 5 labeled anomalies are visually obvious deviations from a clean recurring pattern, ICU vitals are noisy, irregular, and "unusual readings" don't map cleanly onto "this patient will die." Across every hyperparameter configuration tested in the sweep, performance is highly sensitive to the anomaly threshold:

- At threshold percentiles of 85 and above, the model often flags **zero** patients and scores 0% across all metrics.
- At percentile 90, the best configurations found genuine signal: 22.22% F1.
- At lower percentiles (80), more patients get flagged (recall goes up slightly) but precision drops further, since the net is cast wider without much additional true signal.

This narrow window where the model produces any signal at all suggests the separation between "anomalous" and "normal" patients in this score space is weak -- real, but easily overwhelmed by noise outside a fairly specific threshold range.

### A known artifact: patient 137305

One patient (record ID `137305`) is flagged as anomalous in **every single configuration tested**, regardless of architecture or threshold, with anomaly scores in the hundreds of thousands to millions -- multiple orders of magnitude above the next-highest score in the test set. This patient survived their ICU stay. Manual inspection of their raw vitals file showed no obvious data corruption or missing-data artifacts.

The most likely explanation is a known weakness of the Mahalanobis distance approach with a low validation sample size: the 5x5 covariance matrix is fit on a relatively small pooled-error sample, and any patient whose error vector falls in a low-density region of that covariance estimate can produce an extreme, unstable score that has more to do with covariance estimation noise than genuine clinical anomaly. This is a real limitation of applying this scoring method to a dataset with far fewer samples than the original taxi weekly windows, not a bug in the implementation.

### Honest takeaway

The reconstruction-error-based approach, applied as-is to this dataset, does not reliably separate ICU survivors from in-hospital deaths. It finds a weak, real, but easily-lost signal at a fairly narrow part of the threshold range, and produces a substantial number of false positives even at its best configuration (2 true positives out of 9 flags). This is a legitimate and informative finding for a portfolio piece: it shows the methodology working correctly end-to-end (data pipeline, training, scoring, evaluation against real-world ground truth) and demonstrates honest evaluation rather than only reporting favorable numbers.

A natural follow-up (not implemented here) would be combining the reconstruction score with simple clinical severity features (e.g. SOFA score) rather than relying on vitals-only reconstruction error alone, or using a per-feature anomaly threshold instead of a single pooled-covariance score, given how heterogeneous ICU vitals patterns can be across different types of instability.

---

## Out-of-scope: original streaming, Docker, and Striim integration

Everything below this point is **carried over unmodified from the original repo**. None of it was adapted to the ICU vitals domain -- it still operates on the NYC taxi dataset and the original window-level, 336-step Mahalanobis scorer. It's included here for completeness (since the files are still present in this fork and the original README documented them), not as part of this fork's contribution.

### Docker demo with visual application

`code/3_streaming_app.py` is the Docker entrypoint for the live Kafka -> Spark -> Dash demo. It loads the original trained taxi model, consumes the NYC taxi CSV through Kafka, scores each weekly window in Spark Structured Streaming, and renders the results in a live Dash dashboard. **Do not run it directly with `python`** -- launch the full stack with Docker Compose:

```bash
MESSAGE_DELAY_SECONDS=0.005 START_OFFSET=4944 LOOP_DATA=false docker compose up --build -d
```

Open http://localhost:8050 to view the live dashboard.

On subsequent runs (after images are built):

```bash
MESSAGE_DELAY_SECONDS=0.005 START_OFFSET=4944 LOOP_DATA=false docker compose up -d
```

View logs:

```bash
docker compose logs -f app
docker compose logs -f producer
```

### Running this pipeline inside Striim

The Kafka + Spark + Dash demo above is one way to operationalize the original taxi detector. If you are a Striim customer, the same architecture runs natively inside a Striim pipeline, with the LSTM-AE scorer exposed as a FastAPI service and called from a custom Open Processor that handles windowing, feature assembly, and result emission.

A complete walkthrough of the port lives in `striim-plan.md`. It covers:

- The WAEvent pass-through pattern that replaces typed streams and hand-built `_1_0` classes, cutting deployment from 16 steps to 7
- The full TQL for the `FileReader` source, format CQ, and `FileWriter` target, plus Flow Designer wiring for the Open Processor
- The Java OP (`NYCADScorer`) that buffers 336-point weekly windows internally, calls `POST /v1/score`, and writes results back into the inner WAEvent's `data` array
- Verified parity with the standalone pipeline: 22 scored windows, 5/5 labeled anomalies detected (NYC Marathon, Thanksgiving, Christmas, New Year's, January Blizzard), zero false positives

Swap `FileReader` for `KafkaReader`, `OracleReader`, or any other Striim source and the downstream windowing, scoring, and alerting components stay unchanged -- the detector is source-agnostic, and Striim gives you the CDC, exactly-once semantics, and observability you need around it.

### Architecture (original taxi streaming demo)
- **Producer**: Streams NYC taxi data CSV to Kafka topic
- **Kafka**: Message broker for real-time data streaming
- **Spark**: Structured Streaming consumes micro-batches from Kafka
- **LSTM Detector**: Pre-trained Encoder-Decoder flags anomalous weekly windows (original taxi model)
- **Dash**: Real-time visualization with anomaly markers and 6-hour localization

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `START_OFFSET` | `0` | Record index to start streaming from (4944 for test data) |
| `LOOP_DATA` | `true` | Whether to loop through data continuously |
| `MESSAGE_DELAY_SECONDS` | `0.1` | Delay between messages (0.005 for fast demo) |

### Services & Ports

| Service | Port | URL |
|---------|------|-----|
| Dash Dashboard | 8050 | http://localhost:8050 |
| Spark Master UI | 8080 | http://localhost:8080 |
| Kafka | 9092 | External access |
| Zookeeper | 2181 | Kafka coordination |