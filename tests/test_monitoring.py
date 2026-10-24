import json
from pathlib import Path
import pytest
from src.monitoring.alerts import build_alerts
from src.monitoring.metrics import load_drift_metrics
from src.monitoring.pipeline import run_monitoring


def sample_metrics():
    return {"reference_rows": 5000, "production_rows": 5000, "features": {
        "AMT_INCOME_TOTAL": {"ks_statistic": 0.332, "p_value": 2.37e-244, "drift_detected": True},
        "AMT_CREDIT": {"ks_statistic": 0.236, "p_value": 1.67e-122, "drift_detected": True},
    }}


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
    file = tmp_path / "metrics.json"
    file.write_text('{"features": {"x": {"p_value": 1.4, "ks_statistic": 0.2, "drift_detected": false}}}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_drift_metrics(file)


def test_pipeline_com_drift(tmp_path):
    setup_artifacts(tmp_path)
    result = run_monitoring(tmp_path)
    assert result["status"] == "SUCCESS"
    assert result["drift_features"] == 2
    assert (tmp_path / "monitoring/metrics/history.jsonl").is_file()
    assert len(json.loads((tmp_path / "monitoring/alerts/latest_alerts.json").read_text())["alerts"]) == 2


def test_pipeline_falha_sem_artefatos(tmp_path):
    result = run_monitoring(tmp_path)
    assert result["status"] == "FAILED"
    assert result["alerts"][0]["type"] == "PIPELINE_FAILURE"
