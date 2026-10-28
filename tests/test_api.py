import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_liveness_without_data(tmp_path, monkeypatch):
    monkeypatch.setenv("MONITORING_ROOT", str(tmp_path))
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/metrics").status_code == 503


def test_metrics_and_status_expose_only_aggregates(tmp_path, monkeypatch):
    monkeypatch.setenv("MONITORING_ROOT", str(tmp_path))
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics/latest_run.json").write_text(json.dumps({
        "run_id": "abc", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "SUCCESS",
        "duration_seconds": 0.3, "reference_rows": 5000, "production_rows": 5000,
        "features_monitored": 2, "drift_features": 2, "alert_count": 2,
        "alerts": [{"message": "identificador-secreto"}],
    }), encoding="utf-8")
    assert client.get("/ready").status_code == 200
    status = client.get("/status")
    assert status.status_code == 200
    assert "alerts" not in status.json()
    result = client.get("/metrics")
    assert result.status_code == 200
    assert 'credit_pipeline_success 1.0' in result.text
    assert 'credit_drift_features 2.0' in result.text
    assert "identificador-secreto" not in result.text


def test_stale_monitoring_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("MONITORING_ROOT", str(tmp_path))
    monkeypatch.setenv("MONITORING_MAX_AGE_SECONDS", "3600")
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics/latest_run.json").write_text(json.dumps({
        "status": "SUCCESS", "run_id": "abc", "timestamp_utc": "2020-01-01T00:00:00+00:00",
        "duration_seconds": 0.2, "drift_features": 0, "alert_count": 0,
    }), encoding="utf-8")
    assert client.get("/ready").status_code == 503
