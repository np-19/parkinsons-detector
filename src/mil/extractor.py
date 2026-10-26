"""High-Performance Batch Feature & Embedding Extractor using WavLM Backbone.

Runs batched multi-crop extraction on the local NVIDIA GeForce GTX 1650
with PyTorch AMP FP16 and num_workers=0 (to avoid IPC stalls on Linux/Python 3.14).
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deep_learning.model import build_speech_model

LABEL_MAP = {"CONTROL": 0, "PD": 1}


class MPowerRecordingDataset(Dataset):
    """Loads each recording once and extracts multi-crop slices."""

    def __init__(
        self,
        records: List[Dict],
        duration: float = 8.0,
        sample_rate: int = 16000,
        num_crops: int = 3,
    ):
        self.records = records
        self.duration = duration
        self.sample_rate = sample_rate
        self.target_samples = int(duration * sample_rate)
        self.num_crops = num_crops

    def __len__(self) -> int:
        return len(self.records)

    def _load_waveform(self, path: Path) -> np.ndarray:
        try:
            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != self.sample_rate:
                import torchaudio
                t_wave = torch.from_numpy(data).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                t_wave = resampler(t_wave).squeeze(0)
                data = t_wave.numpy()
        except Exception:
            import subprocess
            cmd = [
                "ffmpeg", "-v", "error", "-i", str(path),
                "-f", "f32le", "-ac", "1", "-ar", str(self.sample_rate), "pipe:1"
            ]
            out = subprocess.check_output(cmd)
            data = np.frombuffer(out, dtype=np.float32).copy()

        # Peak normalization
        max_val = np.max(np.abs(data))
        if max_val > 1e-6:
            data = data / max_val
        return data

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        data = self._load_waveform(rec["path"])
        n_samples = len(data)

        crops = []
        if n_samples <= self.target_samples:
            pad_len = self.target_samples - n_samples
            padded = np.pad(data, (0, pad_len), mode="constant")
            for _ in range(self.num_crops):
                crops.append(padded)
        else:
            max_start = n_samples - self.target_samples
            step = max_start / max(self.num_crops - 1, 1)
            for c_i in range(self.num_crops):
                start = int(c_i * step)
                crops.append(data[start : start + self.target_samples])

        crops_tensor = torch.from_numpy(np.stack(crops, axis=0).astype(np.float32))  # (num_crops, target_samples)

        return {
            "crops": crops_tensor,
            "health_code": rec["health_code"],
            "row_id": rec["row_id"],
            "label": rec["label"],
        }


def collate_recordings(batch: List[Dict]) -> Dict:
    # batch of B recordings, each has (num_crops, target_samples)
    crops_list = [item["crops"] for item in batch]
    stacked_crops = torch.cat(crops_list, dim=0)  # (B * num_crops, target_samples)
    return {
        "crops": stacked_crops,
        "health_codes": [item["health_code"] for item in batch],
        "row_ids": [item["row_id"] for item in batch],
        "labels": [item["label"] for item in batch],
        "crops_per_rec": batch[0]["crops"].shape[0],
    }


def extract_participant_bags(
    metadata_csv: Path,
    audio_root: Path,
    model_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int = 4,
    num_crops: int = 3,
    duration: float = 8.0,
    sample_rate: int = 16000,
) -> Dict:
    """Extract WavLM embeddings and group into participant bags."""
    print(f"\n=======================================================")
    print(f" Extracting Participant Bags from {metadata_csv.name}")
    print(f" Target Device: {device} (Batch size: {batch_size} recs = {batch_size * num_crops} crops)")
    print(f"=======================================================")

    df = pd.read_csv(metadata_csv)
    print(f"Loaded {len(df)} metadata rows across {df['healthCode'].nunique()} participants.")

    # Load WavLM backbone
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    m_name = cfg.get("model_name", "microsoft/wavlm-base-plus")
    num_frozen = cfg.get("num_frozen_layers", 2)
    dropout = cfg.get("dropout", 0.2)

    model = build_speech_model(
        model_name=m_name,
        num_frozen_layers=num_frozen,
        dropout=dropout,
        gradient_checkpointing=False,
    )
    raw_state_dict = checkpoint["model_state_dict"]
    state_dict = {k: v for k, v in raw_state_dict.items() if not k.startswith("loss_fn.")}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Collect valid audio files
    records = []
    for _, row in df.iterrows():
        row_id = str(row["ROW_ID"])
        lbl_str = str(row["label"]).strip()
        if lbl_str not in LABEL_MAP:
            continue
        split = str(row.get("split", "")).strip()

        candidates = [
            audio_root / split / lbl_str / f"{row_id}.wav",
            audio_root / lbl_str / f"{row_id}.wav",
            audio_root / f"{row_id}.wav",
        ]
        resolved = None
        for c in candidates:
            if c.exists():
                resolved = c
                break

        if resolved is not None:
            records.append({
                "row_id": row_id,
                "health_code": str(row["healthCode"]),
                "label": LABEL_MAP[lbl_str],
                "path": resolved,
            })

    print(f"Found {len(records)} audio recordings for {len(set(r['health_code'] for r in records))} participants.")

    dataset = MPowerRecordingDataset(
        records=records,
        duration=duration,
        sample_rate=sample_rate,
        num_crops=num_crops,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoids multiprocessing IPC issues
        collate_fn=collate_recordings,
    )

    participant_bags = defaultdict(lambda: {
        "health_code": "",
        "label": 0,
        "embeddings": [],
        "crop_probabilities": [],
        "row_ids": [],
    })

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing {len(records)} Recordings"):
            crops = batch["crops"].to(device, non_blocking=True)
            k = batch["crops_per_rec"]

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(crops)

            embs = out["embeddings"].cpu().float()       # (B * k, 768)
            probs = out["probabilities"][:, 1].cpu().float() # (B * k,)

            b_size = len(batch["health_codes"])
            for b_idx in range(b_size):
                hc = batch["health_codes"][b_idx]
                r_id = batch["row_ids"][b_idx]
                lbl = batch["labels"][b_idx]

                start_c = b_idx * k
                end_c = start_c + k

                bag = participant_bags[hc]
                bag["health_code"] = hc
                bag["label"] = lbl
                bag["embeddings"].append(embs[start_c:end_c])
                bag["crop_probabilities"].append(probs[start_c:end_c])
                bag["row_ids"].append(r_id)

    bags = {}
    for hc, b in participant_bags.items():
        embs_tensor = torch.cat(b["embeddings"], dim=0) # (num_crops_total, 768)
        probs_tensor = torch.cat(b["crop_probabilities"], dim=0)
        unique_row_ids = sorted(list(set(b["row_ids"])))
        bags[hc] = {
            "health_code": hc,
            "label": b["label"],
            "embeddings": embs_tensor,
            "crop_probabilities": probs_tensor,
            "row_ids": unique_row_ids,
            "num_recordings": len(unique_row_ids),
            "num_instances": embs_tensor.shape[0],
        }

    elapsed = time.time() - start_time
    print(f"\nExtraction complete in {elapsed:.1f}s ({len(records) / max(elapsed, 0.1):.1f} recs/sec).")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bags, output_path)
    print(f"Saved {len(bags)} participant bags to {output_path}")
    return bags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WavLM embeddings into participant bags")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata CSV (train.csv, validation.csv, test.csv)")
    parser.add_argument("--output-path", type=str, default=None, help="Explicit destination file for participant bags (.pt)")
    parser.add_argument("--test-metadata", type=str, default="data/metadata/test.csv")
    parser.add_argument("--val-metadata", type=str, default="data/metadata/validation.csv")
    parser.add_argument("--audio-root", type=str, default="data/processed_audio")
    parser.add_argument("--model-path", type=str, default="artifacts/models/deep_learning/wavelm/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="artifacts/mil")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-crops", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dev = torch.device(args.device)
    out_dir = Path(args.output_dir)

    target_meta = Path(args.metadata) if args.metadata else Path(args.test_metadata)
    target_out = Path(args.output_path) if args.output_path else (out_dir / "bags_test.pt")

    extract_participant_bags(
        metadata_csv=target_meta,
        audio_root=Path(args.audio_root),
        model_path=Path(args.model_path),
        output_path=target_out,
        device=dev,
        batch_size=args.batch_size,
        num_crops=args.num_crops,
    )

