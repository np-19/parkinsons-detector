"""Unit tests for deep learning architecture, attention pooling, and metrics."""

import numpy as np
import pytest
import torch

from src.deep_learning.dataset import collate_audio_batch
from src.deep_learning.model import TemporalAttentionPooling
from src.deep_learning.train import (
    aggregate_by_participant,
    compute_metrics,
    find_best_participant_threshold,
)


def test_temporal_attention_pooling_shapes():
    """Verify temporal attention pooling preserves batch dimension and aggregates time."""
    batch_size = 4
    time_steps = 100
    hidden_dim = 768
    attention_dim = 128

    pool = TemporalAttentionPooling(hidden_dim=hidden_dim, attention_dim=attention_dim)
    dummy_input = torch.randn(batch_size, time_steps, hidden_dim)

    pooled, weights = pool(dummy_input)

    assert pooled.shape == (batch_size, hidden_dim), f"Expected {(batch_size, hidden_dim)}, got {pooled.shape}"
    assert weights.shape == (batch_size, time_steps), f"Expected {(batch_size, time_steps)}, got {weights.shape}"

    # Verify attention weights are a valid probability distribution (sum to 1.0)
    sums = torch.sum(weights, dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), f"Weights did not sum to 1: {sums}"


def test_collate_audio_batch():
    """Verify collator batches audio tensors and preserves metadata lists."""
    mock_batch = [
        {
            "input_values": torch.randn(80000),
            "label": torch.tensor(1, dtype=torch.long),
            "row_id": "1001",
            "health_code": "hc_01",
            "label_str": "PD",
        },
        {
            "input_values": torch.randn(80000),
            "label": torch.tensor(0, dtype=torch.long),
            "row_id": "1002",
            "health_code": "hc_02",
            "label_str": "CONTROL",
        },
    ]

    batched = collate_audio_batch(mock_batch)

    assert batched["input_values"].shape == (2, 80000)
    assert batched["labels"].shape == (2,)
    assert batched["labels"].tolist() == [1, 0]
    assert batched["row_ids"] == ["1001", "1002"]
    assert batched["health_codes"] == ["hc_01", "hc_02"]


def test_compute_metrics():
    """Verify metric calculations for balanced accuracy, sensitivity, and specificity."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.2, 0.8])

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    # Predictions: [1, 0, 0, 1]
    # TP=1, FN=1 -> sensitivity = 0.5
    # TN=1, FP=1 -> specificity = 0.5
    # balanced accuracy = 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_participant_aggregation():
    """Verify multiple recordings per participant are properly averaged."""
    health_codes = ["p1", "p1", "p2", "p2", "p3"]
    y_true = np.array([1, 1, 0, 0, 1])
    y_prob = np.array([0.8, 0.6, 0.2, 0.4, 0.9])

    p_true, p_prob = aggregate_by_participant(health_codes, y_true, y_prob)

    assert len(p_true) == 3
    assert len(p_prob) == 3
    # p1 mean = (0.8 + 0.6)/2 = 0.7
    # p2 mean = (0.2 + 0.4)/2 = 0.3
    # p3 mean = 0.9
    expected_probs = [0.7, 0.3, 0.9]
    np.testing.assert_allclose(p_prob, expected_probs, atol=1e-5)
    np.testing.assert_array_equal(p_true, [1, 0, 1])


def test_threshold_search():
    """Verify threshold search finds the optimal threshold for balanced accuracy."""
    p_true = np.array([1, 1, 0, 0])
    p_prob = np.array([0.7, 0.65, 0.6, 0.2])

    # At threshold 0.50: preds=[1, 1, 1, 0] -> bal_acc = (1.0 + 0.5)/2 = 0.75
    # At threshold 0.62: preds=[1, 1, 0, 0] -> bal_acc = (1.0 + 1.0)/2 = 1.0
    best_thresh, best_bal_acc, best_metrics = find_best_participant_threshold(p_true, p_prob)

    assert best_bal_acc == 1.0
    assert 0.60 < best_thresh <= 0.65

