"""Gated Attention Multiple Instance Learning (Attention-MIL) Architecture.

Implements the attention-based deep Multiple Instance Learning formulation
(Ilse, Tomczak, & Welling, ICML 2018) tailored for variable-length participant
bags containing multi-crop sustained vowel audio representations.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionLayer(nn.Module):
    """Gated Attention mechanism (Ilse et al., 2018).

    Computes an adaptive importance weight for each audio crop in a participant bag:
        a_i = w^T (tanh(V * h_i) * sigmoid(U * h_i)) / sqrt(L)
    Followed by softmax over instances so sum(alpha_i) == 1.
    """

    def __init__(self, in_dim: int = 768, attention_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.attention_dim = attention_dim
        self.scale = 1.0 / (attention_dim ** 0.5)

        self.linear_v = nn.Linear(in_dim, attention_dim, bias=True)
        self.linear_u = nn.Linear(in_dim, attention_dim, bias=True)
        self.linear_w = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and aggregated bag embedding.

        Args:
            h: Tensor of shape (N, in_dim), where N is the number of crops in the bag.

        Returns:
            Tuple of:
                bag_embedding: (in_dim,) weighted representation
                weights: (N,) attention probabilities summing to 1.0
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)

        # (N, attention_dim)
        v = torch.tanh(self.linear_v(h))
        u = torch.sigmoid(self.linear_u(h))

        # Element-wise gating: (N, attention_dim)
        gated = v * u

        # Unnormalized attention logits: (N, 1)
        scores = self.linear_w(gated) * self.scale

        # Softmax over instances
        weights = F.softmax(scores.squeeze(-1), dim=0)  # (N,)

        # Weighted aggregate: (in_dim,)
        bag_embedding = torch.matmul(weights, h)

        return bag_embedding, weights


class GatedAttentionMIL(nn.Module):
    """Participant-Level Gated Attention-MIL Classifier.

    Aggregates variable numbers of audio crops (from 1 or more recordings)
    into an interpretable diagnostic prediction for a participant.
    """

    def __init__(
        self,
        in_dim: int = 768,
        attention_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        use_residual_mean: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.attention = GatedAttentionLayer(in_dim=in_dim, attention_dim=attention_dim)
        self.use_residual_mean = use_residual_mean

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        if use_residual_mean:
            self.blend_param = nn.Parameter(torch.tensor(0.0))  # sigmoid -> 0.5 initial blend

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for a single participant bag.

        Args:
            h: Tensor of shape (N, in_dim) representing all crops for a participant.

        Returns:
            Dict containing:
                logits: scalar logit tensor (shape (1,))
                probability: scalar probability in [0, 1]
                attention_weights: (N,) attention scores per instance
                bag_embedding: (in_dim,) pooled representation
        """
        bag_emb, attn_weights = self.attention(h)

        if self.use_residual_mean:
            mean_emb = torch.mean(h, dim=0)
            alpha = torch.sigmoid(self.blend_param)
            pooled_emb = (1.0 - alpha) * bag_emb + alpha * mean_emb
        else:
            pooled_emb = bag_emb

        logit = self.classifier(pooled_emb)  # shape (1,)
        prob = torch.sigmoid(logit)

        return {
            "logit": logit,
            "probability": prob,
            "attention_weights": attn_weights,
            "bag_embedding": pooled_emb,
        }

