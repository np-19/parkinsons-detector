"""PyTorch Dataset and DataLoader for mPower speech recordings."""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset


LABEL_MAP = {
    "CONTROL": 0,
    "PD": 1,
}

INV_LABEL_MAP = {0: "CONTROL", 1: "PD"}


class MPowerAudioDataset(Dataset):
    """Dataset for loading preprocessed 16 kHz mono WAV files for Parkinson's detection."""

    def __init__(
        self,
        metadata_file: Union[str, Path],
        audio_root: Union[str, Path],
        target_duration: float = 5.0,
        sample_rate: int = 16000,
        is_training: bool = False,
        augment: bool = False,
        failures_file: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
    ):
        """Initialize the MPowerAudioDataset.

        Args:
            metadata_file: Path to train.csv, validation.csv, or test.csv.
            audio_root: Root directory containing processed audio.
            target_duration: Target duration in seconds (default: 5.0s -> 80,000 samples).
            sample_rate: Expected sample rate (default: 16000 Hz).
            is_training: Whether dataset is used for training.
            augment: Whether to apply data augmentation (noise injection, time masking).
            failures_file: Optional path to processed_audio_failures.csv to exclude failed rows.
            max_samples: Optional maximum number of samples to load (useful for time estimation/benchmarking).
        """
        self.metadata_file = Path(metadata_file)
        self.audio_root = Path(audio_root)
        self.target_duration = target_duration
        self.sample_rate = sample_rate
        self.target_samples = int(target_duration * sample_rate)
        self.is_training = is_training
        self.augment = augment and is_training
        self.max_samples = max_samples

        # Load metadata
        df = pd.read_csv(self.metadata_file)
        
        # Filter out known failures if failures file provided
        failed_ids = set()
        if failures_file and Path(failures_file).exists():
            fail_df = pd.read_csv(failures_file)
            if "ROW_ID" in fail_df.columns:
                failed_ids = set(fail_df["ROW_ID"].astype(str))

        records = []
        for _, row in df.iterrows():
            row_id = str(row["ROW_ID"])
            if row_id in failed_ids:
                continue

            label_str = str(row["label"]).strip()
            if label_str not in LABEL_MAP:
                continue

            split = str(row.get("split", "")).strip()

            # Locate audio file
            candidate_paths = []
            if "processed_path" in row and pd.notna(row["processed_path"]):
                candidate_paths.append(self.audio_root / str(row["processed_path"]))
            if split:
                candidate_paths.append(self.audio_root / split / label_str / f"{row_id}.wav")
            candidate_paths.append(self.audio_root / label_str / f"{row_id}.wav")
            candidate_paths.append(self.audio_root / f"{row_id}.wav")

            resolved_path = None
            for p in candidate_paths:
                if p.exists():
                    resolved_path = p
                    break

            if resolved_path is not None:
                records.append({
                    "row_id": row_id,
                    "health_code": str(row["healthCode"]),
                    "label": LABEL_MAP[label_str],
                    "label_str": label_str,
                    "path": resolved_path,
                })

        if self.max_samples is not None and self.max_samples > 0:
            records = records[: self.max_samples]

        self.records = records

        if not self.records:
            raise RuntimeError(
                f"No valid audio recordings found matching metadata at {metadata_file} "
                f"under audio root {audio_root}."
            )

    def __len__(self) -> int:
        return len(self.records)

    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise injection, gain scaling, speed perturbation, and time masking during training."""
        # Speed perturbation (0.9x to 1.1x) via linear interpolation
        if np.random.rand() < 0.5:
            speed_factor = np.random.uniform(0.9, 1.1)
            indices = np.arange(0, len(audio), speed_factor)
            indices = indices[indices < len(audio)].astype(int)
            audio = audio[indices]
            # Re-pad/crop to target length after speed change
            if len(audio) > self.target_samples:
                start = np.random.randint(0, len(audio) - self.target_samples + 1)
                audio = audio[start : start + self.target_samples]
            elif len(audio) < self.target_samples:
                audio = np.pad(audio, (0, self.target_samples - len(audio)), mode="constant")

        # Random gain scaling: 0.7 to 1.3
        if np.random.rand() < 0.7:
            gain = np.random.uniform(0.7, 1.3)
            audio = audio * gain

        # Additive Gaussian noise with random SNR between 12 dB and 30 dB
        if np.random.rand() < 0.7:
            signal_power = np.mean(audio ** 2) + 1e-12
            target_snr_db = np.random.uniform(12.0, 30.0)
            noise_power = signal_power / (10 ** (target_snr_db / 10.0))
            noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
            audio = audio + noise

        # Multiple time masks (2-3 masks per sample, SpecAugment-style)
        num_masks = np.random.randint(1, 4)  # 1 to 3 masks
        for _ in range(num_masks):
            if np.random.rand() < 0.5:
                mask_len = int(np.random.uniform(0.03, 0.15) * self.sample_rate)
                if mask_len < len(audio):
                    start = np.random.randint(0, len(audio) - mask_len)
                    audio[start : start + mask_len] = 0.0

        return audio

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        rec = self.records[idx]
        audio_path = rec["path"]

        try:
            data, sr = sf.read(str(audio_path), dtype="float32")
        except Exception as e:
            raise RuntimeError(f"Error reading audio file {audio_path}: {e}")

        # Ensure 1D mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        total_samples = len(data)

        # Center crop or pad to target length
        if total_samples > self.target_samples:
            if self.is_training:
                # Random window around center during training
                max_offset = total_samples - self.target_samples
                start = np.random.randint(0, max_offset + 1)
            else:
                # Deterministic center crop during validation / test
                start = (total_samples - self.target_samples) // 2
            audio_segment = data[start : start + self.target_samples]
        elif total_samples < self.target_samples:
            # Pad with zeros at the end
            pad_len = self.target_samples - total_samples
            audio_segment = np.pad(data, (0, pad_len), mode="constant")
        else:
            audio_segment = data

        if self.augment:
            audio_segment = self._apply_augmentation(audio_segment)

        # Normalize audio amplitude to [-1.0, 1.0]
        max_val = np.max(np.abs(audio_segment))
        if max_val > 1e-6:
            audio_segment = audio_segment / max_val

        return {
            "input_values": torch.from_numpy(audio_segment.astype(np.float32)),
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "row_id": rec["row_id"],
            "health_code": rec["health_code"],
            "label_str": rec["label_str"],
        }


def collate_audio_batch(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Collator function for batching audio tensors and participant metadata."""
    input_values = torch.stack([item["input_values"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    row_ids = [item["row_id"] for item in batch]
    health_codes = [item["health_code"] for item in batch]
    label_strs = [item["label_str"] for item in batch]

    return {
        "input_values": input_values,
        "labels": labels,
        "row_ids": row_ids,
        "health_codes": health_codes,
        "label_strs": label_strs,
    }


def create_dataloader(
    dataset: MPowerAudioDataset,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader with custom collator."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )

