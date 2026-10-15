"""Deep learning fine-tuning package for speech-based Parkinson's detection."""

from .dataset import MPowerAudioDataset, create_dataloader
from .model import Wav2Vec2ForParkinsons, TemporalAttentionPooling

__all__ = [
    "MPowerAudioDataset",
    "create_dataloader",
    "Wav2Vec2ForParkinsons",
    "TemporalAttentionPooling",
]

