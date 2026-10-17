"""Deep learning fine-tuning package for speech-based Parkinson's detection."""

from .dataset import MPowerAudioDataset, create_dataloader
from .model import (
    TemporalAttentionPooling,
    Wav2Vec2ForParkinsons,
    WaveLMForParkinsons,
    build_speech_model,
)

__all__ = [
    "MPowerAudioDataset",
    "create_dataloader",
    "Wav2Vec2ForParkinsons",
    "WaveLMForParkinsons",
    "TemporalAttentionPooling",
    "build_speech_model",
]

