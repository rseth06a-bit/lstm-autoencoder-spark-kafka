"""
Data Preprocessing for LSTM Encoder-Decoder Anomaly Detection

Handles loading, parsing, segmentation, normalization, and splitting
of NYC taxi demand data for the LSTM autoencoder model.

Based on Malhotra et al. (2016) EncDec-AD approach:
- Segments data into weekly chunks (336 records = 48/day x 7 days)
- Fits scaler on training data only
- Creates train/val/test splits with normal data for training
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__) #makes modular logger instance

# Constants for hospital data
VITAL_COLUMNS = ["HR", "RespRate", "Temp", "NISysABP", "NIDiasABP"]
WINDOW_SIZE = 48

@dataclass #holds state and data instead of complex logic
class PreprocessorConfig:
    """Configuration for data preprocessing."""
    sequence_length: int = WINDOW_SIZE  # new window size
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    min_readings: int = WINDOW_SIZE

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for weekly time series sequences."""

    def __init__(self, sequences: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


# ---------------------------------------------------------------------------
# Preprocessing functions (replacing NYCTaxiPreprocessor class)
# ---------------------------------------------------------------------------

def load_patient_file(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load one patient file and change vals to wide format

    Returns:
        DataFrame with one row per timestamp and one column per vital
        None if patient does not have enough data
    """
    
    df = pd.read_csv(filepath, header=0)
    df.columns = ["Time", "Parameter", "Value"]
    df = df[df["Parameter"].isin(VITAL_COLUMNS)].copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    wide = df.pivot_table(index="Time", columns="Parameter", values="Value", aggfunc="mean")
    wide = wide.reindex(columns=VITAL_COLUMNS)
    wide = wide.ffill().bfill()
    wide = wide.dropna()
    return wide if len(wide) >= WINDOW_SIZE else None

def load_all_patients(data_dir: str, config: Optional[PreprocessorConfig] = None,) -> Tuple[np.ndarray, List[str]]:
    """
    Load all patient files and extract fixed-length windows.

    Returns:
        Array of shape (num_patients, WINDOW_SIZE, num_vitals)
        List of patient IDs
    """
    config = config or PreprocessorConfig()
    data_dir = Path(data_dir)
    windows = []
    patient_ids = []
    skipped = 0
    for filepath in sorted(data_dir.glob("*.txt")):
        wide = load_patient_file(filepath)
        if wide is None:
            skipped += 1
            continue
        window = wide.values[:config.sequence_length].astype(np.float32)
        windows.append(window)
        patient_ids.append(filepath.stem)
    logger.info(f"Loaded {len(windows)} patients, skipped {skipped} with insufficient data")
    return np.array(windows), patient_ids

def create_splits(data: np.ndarray, patient_ids: List[str], config: Optional[PreprocessorConfig] = None,) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Split patients into train/val/test sets by fraction.
    """
    config = config or PreprocessorConfig()
    n = len(data)
    n_train = int(n * config.train_fraction)
    n_val = int(n * config.val_fraction)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    splits = {
        "train": data[train_idx],
        "val":   data[val_idx],
        "test":  data[test_idx],
    }
    split_ids = {
        "train": [patient_ids[i] for i in train_idx],
        "val":   [patient_ids[i] for i in val_idx],
        "test":  [patient_ids[i] for i in test_idx],
    }
    logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    return splits, split_ids


def normalize(
    splits: Dict[str, np.ndarray],
    fit_on: str = "train",
) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
    """
    Normalize data using StandardScaler (fit on training data only).

    Returns:
        Tuple of (normalized_splits, fitted_scaler)
    """
    if fit_on not in splits or len(splits[fit_on]) == 0:
        raise ValueError(f"Cannot fit scaler: '{fit_on}' split is empty")

    train_flat = splits[fit_on].reshape(-1, splits[fit_on].shape[-1])

    scaler = StandardScaler()
    scaler.fit(train_flat)

    logger.info(f"Fitted scaler on {fit_on} data:")
    logger.info(f"  Mean: {scaler.mean_[0]:.2f}")
    logger.info(f"  Std: {scaler.scale_[0]:.2f}")

    normalized_splits = {}
    for name, data in splits.items():
        if len(data) == 0:
            normalized_splits[name] = data
            continue
        original_shape = data.shape
        flat = data.reshape(-1, original_shape[-1])
        normalized_splits[name] = scaler.transform(flat).reshape(original_shape)

    return normalized_splits, scaler


def create_dataloaders(
    normalized_splits: Dict[str, np.ndarray],
    batch_size: int = 4,
) -> Dict[str, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for each split.

    Data is NOT shuffled to preserve temporal ordering.
    """
    dataloaders = {}

    for name, data in normalized_splits.items():
        if len(data) == 0:
            dataloaders[name] = None
            continue

        dataset = TimeSeriesDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(data)),
            shuffle=(name == "train"),
            drop_last=False,
        )
        dataloaders[name] = loader
        logger.debug(f"Created DataLoader for {name}: {len(dataset)} samples")

    return dataloaders

def preprocess_pipeline(
    data_dir: str,
    config: Optional[PreprocessorConfig] = None,
    batch_size: int = 16,
) -> Tuple[Dict, Dict, StandardScaler, List[str], Dict]:
    """
    Run the complete preprocessing pipeline.

    Returns:
        Tuple of (dataloaders, normalized_splits, scaler, patient_ids, split_ids)
    """
    config = config or PreprocessorConfig()
    logger.info("=" * 60)
    logger.info("Starting ICU vitals preprocessing pipeline")
    logger.info("=" * 60)
    data, patient_ids = load_all_patients(data_dir, config)
    splits, split_ids = create_splits(data, patient_ids, config)
    normalized_splits, scaler = normalize(splits)
    dataloaders = create_dataloaders(normalized_splits, batch_size=batch_size)
    logger.info("Preprocessing complete")
    return dataloaders, normalized_splits, scaler, patient_ids, split_ids