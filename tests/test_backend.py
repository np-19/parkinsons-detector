"""Unit and integration tests for FastAPI backend routes and schemas."""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import (
    MetaStatsResponse,
    PredictionResponse,
    RecordingResult,
    StatisticalDistribution,
)
from backend.services.predictor_service import PredictorService


@pytest.fixture
def client():
    # Use TestClient with lifespan context manager disabled for lightweight route testing
    with TestClient(app) as test_client:
        yield test_client


def test_schema_serialization():
    """Verify PredictionResponse schema validates and serializes correctly."""
    rec = RecordingResult(
        filename="take_1.wav",
        duration_seconds=10.2,
        probability_pd=0.1111,
        recording_status="Normal Phonation",
    )
    dist = StatisticalDistribution(
        mean_probability=0.1111,
        top2_probability=0.1111,
        max_probability=0.1111,
        min_probability=0.1111,
        std_deviation=0.0000,
        recordings_analyzed=1,
    )
    resp = PredictionResponse(
        diagnosis="NEGATIVE (Normal Phonation Dynamics)",
        is_pd=False,
        confidence_level="High Confidence Negative",
        meta_classifier_score=0.0000,
        threshold=0.29,
        recordings_analyzed=1,
        distribution=dist,
        recordings=[rec],
        failed_recordings=[],
        clinical_warnings=["Single recording analyzed."],
    )
    assert resp.is_pd is False
    assert resp.meta_classifier_score == 0.0000
    assert len(resp.recordings) == 1
    assert resp.recordings[0].recording_status == "Normal Phonation"


def test_root_endpoint(client):
    """Verify root endpoint returns operational metadata."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"
    assert "/docs" in data["documentation"]


def test_meta_stats_endpoint(client):
    """Verify meta-stats endpoint returns valid 10-moment weights and active threshold."""
    res = client.get("/api/v1/meta-stats")
    assert res.status_code == 200
    data = res.json()
    assert data["threshold"] in (0.30, 0.31, 0.36, 0.39)
    assert len(data["weights"]) == 10
    assert len(data["reference_means"]) == 10
    assert len(data["reference_stds"]) == 10


def test_predict_rejects_unsupported_format(client):
    """Verify API rejects non-audio formats (.txt, .exe, etc.) with 415."""
    files = [("files", ("test.txt", b"hello world", "text/plain"))]
    res = client.post("/api/v1/predict", files=files)
    assert res.status_code == 415
    assert "Unsupported audio format" in res.json()["detail"]


def test_predict_rejects_empty_file(client):
    """Verify API rejects 0-byte audio uploads with 400 Bad Request."""
    files = [("files", ("test.wav", b"", "audio/wav"))]
    res = client.post("/api/v1/predict", files=files)
    assert res.status_code == 400
    assert "0 bytes" in res.json()["detail"]
