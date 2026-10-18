"""Automated unit tests for Gated Attention-MIL pipeline."""

import numpy as np
import pytest
import torch

from src.mil.model import GatedAttentionLayer, GatedAttentionMIL
from src.mil.train import compute_bootstrap_ci, compute_metrics, find_optimal_threshold


def test_gated_attention_layer_shapes():
    """Verify GatedAttentionLayer produces correct output dimensions and normalized weights."""
    layer = GatedAttentionLayer(in_dim=768, attention_dim=128)
    h = torch.randn(15, 768)  # 15 crops in a participant bag
    bag_emb, weights = layer(h)

    assert bag_emb.shape == (768,)
    assert weights.shape == (15,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert (weights >= 0.0).all()
    assert (weights <= 1.0).all()


def test_attention_permutation_equivariance():
    """Verify that permuting input instances permutes attention weights identically."""
    layer = GatedAttentionLayer(in_dim=768, attention_dim=128)
    layer.eval()

    h = torch.randn(10, 768)
    perm = torch.randperm(10)
    h_perm = h[perm]

    with torch.no_grad():
        _, weights = layer(h)
        _, weights_perm = layer(h_perm)

    # Weights for permuted inputs should match permuted original weights
    assert torch.allclose(weights[perm], weights_perm, atol=1e-5)


def test_variable_bag_sizes():
    """Verify Attention-MIL handles variable-sized participant bags (1 crop to 50 crops)."""
    model = GatedAttentionMIL(in_dim=768, attention_dim=128, hidden_dim=128)
    model.eval()

    for n_crops in [1, 5, 12, 25, 50]:
        h = torch.randn(n_crops, 768)
        with torch.no_grad():
            out = model(h)

        assert out["logit"].shape == (1,)
        assert 0.0 <= out["probability"].item() <= 1.0
        assert out["attention_weights"].shape == (n_crops,)
        assert torch.isclose(out["attention_weights"].sum(), torch.tensor(1.0), atol=1e-5)


def test_gated_attention_mil_gradient_flow():
    """Verify backward pass computes non-zero gradients for all attention and classifier layers."""
    model = GatedAttentionMIL(in_dim=768, attention_dim=128, hidden_dim=128)
    model.train()

    h = torch.randn(8, 768)
    target = torch.tensor([1.0])

    out = model(h)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(out["logit"], target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"
        assert torch.isfinite(param.grad).all(), f"Gradient has non-finite values in {name}"


def test_residual_mean_blend():
    """Verify optional residual mean mechanism functions correctly."""
    model = GatedAttentionMIL(in_dim=768, attention_dim=128, hidden_dim=128, use_residual_mean=True)
    h = torch.randn(6, 768)
    out = model(h)
    assert out["bag_embedding"].shape == (768,)
    assert 0.0 <= out["probability"].item() <= 1.0


def test_zero_leakage_kfold_isolation():
    """Verify that participant splits strictly isolate participants across train and validation."""
    from sklearn.model_selection import StratifiedKFold
    participants = [f"p_{i}" for i in range(100)]
    labels = np.array([1 if i < 30 else 0 for i in range(100)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(participants, labels)):
        train_set = set(participants[i] for i in train_idx)
        val_set = set(participants[i] for i in val_idx)
        assert len(train_set.intersection(val_set)) == 0, f"Participant leakage in fold {fold}"


def test_bootstrap_ci_coverage():
    """Verify bootstrap CI produces valid ordered intervals."""
    y_true = np.array([1]*20 + [0]*60)
    y_prob = np.array([0.8]*18 + [0.2]*2 + [0.1]*55 + [0.7]*5)
    ci = compute_bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstraps=100)

    for metric in ["balanced_accuracy_ci", "roc_auc_ci", "sensitivity_ci", "specificity_ci"]:
        low, high = ci[metric]
        assert 0.0 <= low <= high <= 1.0, f"Invalid CI interval for {metric}: [{low}, {high}]"


def test_exported_production_mil_head():
    """Verify exported production mil_head.pt loads and matches calibrated threshold."""
    from pathlib import Path
    mil_path = Path("artifacts/models/deep_learning/wavelm/mil_head.pt")
    assert mil_path.exists(), f"Missing production MIL head at {mil_path}"

    ckpt = torch.load(mil_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "optimal_threshold" in ckpt
    assert abs(ckpt["optimal_threshold"] - 0.300) < 1e-4

    valid_keys = {"in_dim", "attention_dim", "hidden_dim", "dropout", "use_residual_mean"}
    clean_cfg = {k: v for k, v in ckpt.get("config", {}).items() if k in valid_keys}
    model = GatedAttentionMIL(**clean_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_h = torch.randn(6, 768)
    with torch.no_grad():
        out = model(dummy_h)
    assert 0.0 <= out["probability"].item() <= 1.0
    assert out["attention_weights"].shape == (6,)
    assert torch.isclose(out["attention_weights"].sum(), torch.tensor(1.0), atol=1e-5)


def test_single_recording_inference_uses_mil():
    """Verify single-recording bag runs MIL head on the 3 crop representations and blends (Issue 1)."""
    from pathlib import Path
    from src.mil.model import score_production_bag

    mil_path = Path("artifacts/models/deep_learning/wavelm/mil_head.pt")
    ckpt = torch.load(mil_path, map_location="cpu", weights_only=False)
    valid_keys = {"in_dim", "attention_dim", "hidden_dim", "dropout", "use_residual_mean"}
    clean_cfg = {k: v for k, v in ckpt.get("config", {}).items() if k in valid_keys}
    model = GatedAttentionMIL(**clean_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1 recording with 3 crops
    embs = torch.randn(3, 768)
    crop_probs = torch.tensor([0.20, 0.25, 0.24])  # mean = 0.23

    res = score_production_bag(
        mil_model=model,
        bag_embeddings=embs,
        crop_probabilities=crop_probs,
        num_recordings=1,
        recording_threshold=0.310,
        crops_per_rec=3,
    )

    # Must have 3 crop weights summing to 1.0
    assert len(res["crop_weights"]) == 3
    assert abs(sum(res["crop_weights"]) - 1.0) < 1e-4

    # Recording weight must be 1.0
    assert len(res["rec_weights"]) == 1
    assert abs(res["rec_weights"][0] - 1.0) < 1e-4

    # Recording prob equals mean of crops
    assert abs(res["rec_probs"][0] - 0.23) < 1e-4

    # Participant score is blended
    expected_blend = 0.5 * 0.23 + 0.5 * res["mil_prob"]
    # Since rec_prob (0.23) < recording_threshold (0.310), guardrail applies: min(blend, 0.23)
    expected_final = min(expected_blend, 0.23)
    assert abs(res["participant_score"] - expected_final) < 1e-4


def test_scoring_rule_consistency_across_bag_sizes():
    """Verify identical scoring rule applies to 1-recording and multi-recording bags (Issue 1 & 2)."""
    from pathlib import Path
    from src.mil.model import score_production_bag

    mil_path = Path("artifacts/models/deep_learning/wavelm/mil_head.pt")
    ckpt = torch.load(mil_path, map_location="cpu", weights_only=False)
    clean_cfg = {k: v for k, v in ckpt.get("config", {}).items() if k in {"in_dim", "attention_dim", "hidden_dim", "dropout", "use_residual_mean"}}
    model = GatedAttentionMIL(**clean_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for n_rec in [1, 2, 4]:
        embs = torch.randn(n_rec * 3, 768)
        crop_probs = torch.full((n_rec * 3,), 0.45)
        res = score_production_bag(
            mil_model=model,
            bag_embeddings=embs,
            crop_probabilities=crop_probs,
            num_recordings=n_rec,
            recording_threshold=0.310,
        )
        assert "participant_score" in res
        assert "mil_prob" in res
        assert "weighted_p" in res
        assert len(res["rec_weights"]) == n_rec
        assert abs(sum(res["rec_weights"]) - 1.0) < 1e-4


def test_threshold_separation_and_documentation():
    """Verify recording-level (0.310) and participant-level (0.300) thresholds are separated (Issue 4)."""
    from src.deep_learning.predict_patient import (
        ACOUSTIC_OPTIMAL_THRESHOLD,
        MIL_PARTICIPANT_THRESHOLD,
        WAVELM_RECORDING_THRESHOLD,
    )

    assert WAVELM_RECORDING_THRESHOLD == 0.310
    assert MIL_PARTICIPANT_THRESHOLD == 0.300
    assert ACOUSTIC_OPTIMAL_THRESHOLD == 0.310


def test_consensus_guardrail_guarantees_negative_classification():
    """Verify consensus guardrail strictly prevents positive classification when all recordings < tau_rec.

    Edge case: all recordings < 0.310, but the highest is 0.305 (in [0.300, 0.310)).
    Even with high raw MIL probability (0.95), the score must be capped strictly below 0.300.
    """
    from src.mil.model import score_production_bag

    class HighProbMIL(torch.nn.Module):
        def forward(self, h):
            n = h.shape[0]
            return {
                "probability": torch.tensor(0.95),
                "attention_weights": torch.full((n,), 1.0 / n),
            }

    mock_model = HighProbMIL()
    embs = torch.randn(6, 768)
    # Both recordings are < 0.310 (normal phonation), but rec1 = 0.305 >= 0.300
    crop_probs = torch.tensor([0.305, 0.305, 0.305, 0.280, 0.280, 0.280])

    res = score_production_bag(
        mil_model=mock_model,
        bag_embeddings=embs,
        crop_probabilities=crop_probs,
        num_recordings=2,
        recording_threshold=0.310,
        participant_threshold=0.300,
    )

    assert res["guardrail_active"] is True
    # Guaranteed strictly less than participant_threshold (0.300)
    assert res["participant_score"] < 0.300
    assert res["participant_score"] == pytest.approx(0.300 - 1e-4, abs=1e-6)
    # Enforces negative diagnosis
    assert (res["participant_score"] >= 0.300) is False


def test_consensus_guardrail_inactive_when_abnormal_recording_exists():
    """Verify consensus guardrail does NOT activate when at least one recording >= tau_rec."""
    from src.mil.model import score_production_bag

    class SimpleMIL(torch.nn.Module):
        def forward(self, h):
            n = h.shape[0]
            return {
                "probability": torch.tensor(0.60),
                "attention_weights": torch.full((n,), 1.0 / n),
            }

    mock_model = SimpleMIL()
    embs = torch.randn(6, 768)
    # Rec 1 is abnormal (0.315 >= 0.310), Rec 2 is normal (0.200 < 0.310)
    crop_probs = torch.tensor([0.315, 0.315, 0.315, 0.200, 0.200, 0.200])

    res = score_production_bag(
        mil_model=mock_model,
        bag_embeddings=embs,
        crop_probabilities=crop_probs,
        num_recordings=2,
        recording_threshold=0.310,
        participant_threshold=0.300,
    )

    assert res["guardrail_active"] is False
    assert res["participant_score"] == res["blend_p"]


def test_scoring_path_parity_between_predictor_and_helper(tmp_path):
    """Verify PatientPredictor and score_production_bag produce identical scores without divergence."""
    import soundfile as sf
    from src.deep_learning.predict_patient import PatientPredictor
    from src.mil.model import score_production_bag, GatedAttentionMIL

    # Generate two synthetic 3-second audio files
    f1 = tmp_path / "take1.wav"
    f2 = tmp_path / "take2.wav"
    sr = 16000
    t = np.linspace(0, 3, sr * 3)
    sf.write(str(f1), (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sr)
    sf.write(str(f2), (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), sr)

    predictor = PatientPredictor.__new__(PatientPredictor)
    predictor.device = torch.device("cpu")
    predictor.sample_rate = sr
    predictor.duration = 8.0
    predictor.target_length = int(8.0 * sr)
    predictor.num_crops = 3
    predictor.use_attention_mil = True
    predictor.recording_threshold = 0.310
    predictor.optimal_threshold = 0.300

    class DeterministicAcoustic(torch.nn.Module):
        def forward(self, input_values):
            b = input_values.shape[0]
            # Deterministic embeddings and probabilities
            torch.manual_seed(123)
            embs = torch.randn(b, 768)
            probs = torch.tensor([[0.72, 0.28]] * b)
            return {"probabilities": probs, "embeddings": embs}

    torch.manual_seed(42)
    predictor.model = DeterministicAcoustic()
    predictor.mil_model = GatedAttentionMIL(768, 128, 128, 0.25)
    predictor.mil_model.eval()

    # Predict via PatientPredictor
    pred_res = predictor.predict_patient([f1, f2])

    # Independently extract embs and crop_probs using predictor methods
    w1 = predictor.load_audio(f1)
    w2 = predictor.load_audio(f2)
    _, e1, c1 = predictor.infer_single_recording(w1, return_embeddings=True, return_crop_probs=True)
    _, e2, c2 = predictor.infer_single_recording(w2, return_embeddings=True, return_crop_probs=True)

    bag_tensor = torch.cat([e1, e2], dim=0)
    crop_tensor = torch.cat([c1, c2], dim=0)

    score_res = score_production_bag(
        mil_model=predictor.mil_model,
        bag_embeddings=bag_tensor,
        crop_probabilities=crop_tensor,
        num_recordings=2,
        recording_threshold=0.310,
        participant_threshold=0.300,
        crops_per_rec=3,
    )

    # Assert exact score parity
    assert abs(pred_res["raw_participant_score"] - score_res["participant_score"]) < 1e-7
    assert pred_res["participant_mean_score"] == round(score_res["participant_score"], 4)
    assert pred_res["mil_probability"] == round(score_res["mil_prob"], 4)

    # Assert per-recording details parity
    assert len(pred_res["recordings_detail"]) == 2
    for idx, rec_detail in enumerate(pred_res["recordings_detail"]):
        assert rec_detail["probability_PD"] == round(score_res["rec_probs"][idx], 4)
        assert rec_detail["attention_weight"] == round(score_res["rec_weights"][idx], 4)


