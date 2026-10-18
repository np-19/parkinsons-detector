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


def score_production_bag(
    mil_model: nn.Module,
    bag_embeddings: torch.Tensor,
    crop_probabilities: torch.Tensor,
    num_recordings: int,
    recording_threshold: float = 0.310,
    crops_per_rec: int = 3,
) -> Dict:
    """Score a participant bag using the complete deployed production scoring pipeline.

    Unified for all bag sizes (single recording n=1 or multiple recordings n>1):
    1. MIL forward pass on all crop embeddings -> mil_prob and crop attention weights.
    2. Sum crop attention weights per recording and normalize across recordings.
    3. Compute per-recording probability as the mean of its crop probabilities.
    4. Compute attention-weighted recording probability: sum(w_i * p_i).
    5. 50:50 blend: participant_score = 0.5 * weighted_p + 0.5 * mil_prob.
    6. Clinical Consensus Guardrail: If all recordings are below recording_threshold,
       cap participant_score at max(p_i).
    """
    import numpy as np

    with torch.no_grad():
        mil_out = mil_model(bag_embeddings)
        mil_prob = float(mil_out["probability"].item())
        crop_weights = mil_out["attention_weights"].detach().cpu().numpy()

        if isinstance(crop_probabilities, torch.Tensor):
            crop_p = crop_probabilities.detach().cpu().numpy()
        else:
            crop_p = np.array(crop_probabilities, dtype=float)

        n_rec = max(num_recordings, 1)
        actual_crops_per_rec = len(crop_p) // n_rec if n_rec > 0 else crops_per_rec

        rec_probs = []
        rec_weights = []
        for r in range(n_rec):
            start = r * actual_crops_per_rec
            end = start + actual_crops_per_rec
            rec_probs.append(float(np.mean(crop_p[start:end])))
            rec_weights.append(float(np.sum(crop_weights[start:end])))

        total_w = sum(rec_weights)
        if total_w > 1e-6:
            rec_weights = [w / total_w for w in rec_weights]
        else:
            rec_weights = [1.0 / n_rec] * n_rec

        weighted_p = sum(w * p for w, p in zip(rec_weights, rec_probs))
        blend_p = 0.5 * weighted_p + 0.5 * mil_prob

        # Clinical consensus guardrail
        max_rec_p = max(rec_probs) if rec_probs else blend_p
        if all(p < recording_threshold for p in rec_probs):
            final_p = min(blend_p, max_rec_p)
        else:
            final_p = blend_p

        return {
            "participant_score": float(final_p),
            "mil_prob": float(mil_prob),
            "weighted_p": float(weighted_p),
            "blend_p": float(blend_p),
            "rec_probs": rec_probs,
            "rec_weights": rec_weights,
            "crop_weights": [round(float(w), 4) for w in crop_weights],
        }

