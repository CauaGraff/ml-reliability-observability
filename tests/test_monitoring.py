import json
from pathlib import Path

import pytest

from src.monitoring.alerts import build_alerts
from src.monitoring.metrics import load_drift_metrics
from src.monitoring.pipeline import _tracking_uri, record_pipeline_failure, run_monitoring


def sample_metrics():
    return {
        "reference_rows": 5000, "production_rows": 5000,
        "features": {
            "AMT_INCOME_TOTAL": {"ks_statistic": 0.332, "p_value": 2.37e-244, "drift_detected": True},
            "AMT_CREDIT": {"ks_statistic": 0.236, "p_value": 1.67e-122, "drift_detected": True},
        },
    }


def setup_artifacts(root: Path):
    (root / "reports").mkdir(parents=True)
    (root / "data/production").mkdir(parents=True)
    (root / "reports/drift_metrics.json").write_text(json.dumps(sample_metrics()), encoding="utf-8")
    (root / "reports/drift_report.html").write_text("<html>relatorio</html>", encoding="utf-8")
    (root / "data/production/credit_predictions.csv").write_text("prediction\n1\n", encoding="utf-8")


def test_alertas_para_duas_features():
    alerts = build_alerts(sample_metrics())
    assert len(alerts) == 2
    assert all(a["type"] == "DATA_DRIFT" for a in alerts)


def test_rejeita_metrica_invalida(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text('{"reference_rows": 5, "production_rows": 5, "features": {"x": {"p_value": 1.4, "ks_statistic": 0.2, "drift_detected": false}}}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_drift_metrics(path)


def test_pipeline_com_drift(tmp_path):
    setup_artifacts(tmp_path)
    record = run_monitoring(tmp_path)
    assert record["status"] == "SUCCESS"
    assert record["drift_features"] == 2
    history = (tmp_path / "monitoring/metrics/history.jsonl").read_text(encoding="utf-8")
    assert len(history.splitlines()) == 1
    assert len(json.loads((tmp_path / "monitoring/alerts/latest_alerts.json").read_text())["alerts"]) == 2


def test_pipeline_falha_sem_artefatos(tmp_path):
    record = run_monitoring(tmp_path)
    assert record["status"] == "FAILED"
    assert record["alerts"][0]["type"] == "PIPELINE_FAILURE"


def test_orquestracao_falhou_nao_mantem_success_anterior(tmp_path):
    setup_artifacts(tmp_path)
    assert run_monitoring(tmp_path)["status"] == "SUCCESS"
    record_pipeline_failure(tmp_path, "scripts.train", RuntimeError("conteudo sensivel"))
    latest = json.loads((tmp_path / "monitoring/metrics/latest_run.json").read_text())
    assert latest["status"] == "FAILED"
    assert "conteudo sensivel" not in (tmp_path / "monitoring/logs/monitoring.log").read_text()


def test_tracking_uri_sqlite_linux_e_env(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert _tracking_uri(tmp_path).startswith("sqlite:////")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert _tracking_uri(tmp_path) == "http://mlflow:5000"
