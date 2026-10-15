"""PyTorch Neural Network architecture for speech-based Parkinson's detection.

Features:
- Pretrained Wav2Vec2 backbone (facebook/wav2vec2-base)
- Frozen 7-layer CNN feature encoder
- Selectively frozen lower Transformer layers (layers 1-6)
- Fine-tuned upper Transformer layers (layers 7-12)
- Learnable temporal attention pooling
- Multi-layer classification head with LayerNorm, Dropout, and GELU
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model


class TemporalAttentionPooling(nn.Module):
    """Learnable temporal attention pooling module.

    Computes a weighted sum across acoustic frame timestamps, allowing the network
    to focus attention on timestamps containing vocal dysarthria and micro-breaks.
    """

    def __init__(self, hidden_dim: int = 768, attention_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for temporal attention pooling.

        Args:
            hidden_states: Tensor of shape (batch_size, time_steps, hidden_dim).
            attention_mask: Optional tensor of shape (batch_size, time_steps).

        Returns:
            Tuple of (pooled_representation, attention_weights):
                pooled_representation: Tensor of shape (batch_size, hidden_dim)
                attention_weights: Tensor of shape (batch_size, time_steps)
        """
        # Calculate unnormalized attention scores: (batch_size, time_steps, 1)
        scores = self.projection(hidden_states)

        if attention_mask is not None:
            # Mask out padded positions
            mask = attention_mask.unsqueeze(-1).to(dtype=scores.dtype)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute softmax over time dimension: (batch_size, time_steps, 1)
        weights = F.softmax(scores, dim=1)

        # Weighted sum: (batch_size, hidden_dim)
        pooled = torch.sum(weights * hidden_states, dim=1)

        return pooled, weights.squeeze(-1)


class Wav2Vec2ForParkinsons(nn.Module):
    """Fine-tuned Wav2Vec2 model for Parkinson's disease classification."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        num_frozen_layers: int = 6,
        freeze_feature_extractor: bool = True,
        gradient_checkpointing: bool = True,
        dropout: float = 0.2,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        mask_time_prob: float = 0.05,
        mask_feature_prob: float = 0.05,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_frozen_layers = num_frozen_layers
        self.freeze_feature_extractor = freeze_feature_extractor

        # Load Wav2Vec2 backbone
        self.config = Wav2Vec2Config.from_pretrained(model_name)
        self.config.apply_spec_augment = True
        self.config.mask_time_prob = mask_time_prob
        self.config.mask_time_length = 10
        self.config.mask_feature_prob = mask_feature_prob
        self.config.mask_feature_length = 10

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, config=self.config)

        # Enable gradient checkpointing to save VRAM on GTX 1650
        if gradient_checkpointing:
            self.wav2vec2.gradient_checkpointing_enable()

        # Freeze feature extractor (CNN layers)
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor.requires_grad_(False)

        # Freeze first N Transformer encoder layers
        if hasattr(self.wav2vec2.encoder, "layers"):
            for idx, layer in enumerate(self.wav2vec2.encoder.layers):
                if idx < num_frozen_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True

        hidden_dim = self.config.hidden_size  # 768 for wav2vec2-base

        # Temporal attention pooling
        self.pool = TemporalAttentionPooling(hidden_dim=hidden_dim, attention_dim=128)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Loss function with optional class weighting and label smoothing
        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_values: Audio tensor of shape (batch_size, samples).
            labels: Optional class labels (batch_size,).
            attention_mask: Optional attention mask (batch_size, samples).

        Returns:
            Dictionary containing:
                - logits: (batch_size, num_classes)
                - probabilities: (batch_size, num_classes)
                - attention_weights: (batch_size, time_frames)
                - loss: scalar loss tensor if labels provided
        """
        # Wav2Vec2 forward pass
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state  # (batch_size, time_frames, hidden_dim)

        # Temporal attention pooling
        pooled, attn_weights = self.pool(hidden_states)

        # Classification logits
        logits = self.classifier(pooled)
        probabilities = F.softmax(logits, dim=-1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
            "attention_weights": attn_weights,
            "embeddings": pooled,
        }

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            result["loss"] = loss

        return result

    def get_parameter_groups(
        self,
        backbone_lr: float = 2e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 0.01,
    ) -> list:
        """Return parameter groups with differential learning rates."""
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name or "pool" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]

