"""Monitoramento de lote concluido: logs, indicadores, alertas e MLflow."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.monitoring.alerts import build_alerts
from src.monitoring.logger import configure_logger
from src.monitoring.metrics import append_history, load_drift_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _tracking_uri(root: Path) -> str:
    """sqlite:///C:/... no Windows; sqlite:////app/... no Linux."""
    configured = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if configured:
        return configured
    db_file = (root / "monitoring" / "mlflow.db").resolve()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + db_file.as_posix()


def _log_mlflow(root: Path, record: dict, alpha: float) -> None:
    import mlflow

    mlflow.set_tracking_uri(_tracking_uri(root))
    mlflow.set_experiment("fiap-fase4-observabilidade")
    with mlflow.start_run(run_name=f"monitoramento-{record['run_id'][:8]}"):
        mlflow.log_params({"ks_alpha": alpha, "monitoring_type": "batch"})
        mlflow.log_metrics({
            "duration_seconds": record["duration_seconds"],
            "features_monitored": record["features_monitored"],
            "drift_features": record["drift_features"],
            "alert_count": record["alert_count"],
            "pipeline_success": int(record["status"] == "SUCCESS"),
            "production_rows": record["production_rows"] or 0,
        })
        mlflow.set_tags({"pipeline_status": record["status"], "source": "fiap-fase4"})


def _persist(root: Path, record: dict) -> None:
    _write_json(root / "monitoring/metrics/latest_run.json", record)
    _write_json(root / "monitoring/alerts/latest_alerts.json", {
        "run_id": record["run_id"], "timestamp_utc": record["timestamp_utc"], "alerts": record["alerts"]
    })
    append_history(root / "monitoring/metrics/history.jsonl", record)


def record_pipeline_failure(root: Path, step: str, exception: Exception) -> dict:
    """Marca falha de orquestracao, evitando reaproveitar um SUCCESS antigo."""
    root = Path(root)
    logger = configure_logger(root / "monitoring/logs/monitoring.log")
    # Evitamos expor caminho absoluto/nomes de dados individuais no log persistido.
    message = f"Falha na etapa {step}: {type(exception).__name__}"
    logger.error("%s", message)
    record = {
        "run_id": uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAILED",
        "duration_seconds": 0.0,
        "reference_rows": None,
        "production_rows": None,
        "features_monitored": 0,
        "drift_features": 0,
        "alert_count": 1,
        "alerts": [{"level": "CRITICAL", "type": "PIPELINE_FAILURE", "message": message}],
        "error": message,
    }
    _persist(root, record)
    return record


def run_monitoring(root: Path, *, alpha: float = 0.05, use_mlflow: bool = False) -> dict:
    """Consulta os artefatos mais recentes, sem executar novamente inferencia."""
    root = Path(root)
    logger = configure_logger(root / "monitoring/logs/monitoring.log")
    start = time.perf_counter()
    now = datetime.now(timezone.utc).isoformat()
    run_id = uuid4().hex
    status = "SUCCESS"
    alerts: list[dict] = []
    error = None
    data = None
    logger.info("Iniciando monitoramento | run_id=%s", run_id)
    try:
        expected = [
            root / "reports/drift_metrics.json",
            root / "reports/drift_report.html",
            root / "data/production/credit_predictions.csv",
        ]
        missing = [str(p.relative_to(root)) for p in expected if not p.is_file() or p.stat().st_size == 0]
        if missing:
            raise FileNotFoundError("Artefatos ausentes/vazios: " + ", ".join(missing))
        data = load_drift_metrics(expected[0])
        alerts = build_alerts(data, alpha=alpha)
        for alert in alerts:
            logger.warning(
                "Alerta %s | feature=%s | KS=%.4f | p=%.3g",
                alert["type"], alert["feature"], alert["ks_statistic"], alert["p_value"]
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        status = "FAILED"
        error = f"Falha ao verificar artefatos ou metricas: {type(exc).__name__}"
        alerts = [{"level": "CRITICAL", "type": "PIPELINE_FAILURE", "message": error}]
        logger.error("%s", error)

    record = {
        "run_id": run_id,
        "timestamp_utc": now,
        "status": status,
        "duration_seconds": round(time.perf_counter() - start, 4),
        "reference_rows": data.get("reference_rows") if data else None,
        "production_rows": data.get("production_rows") if data else None,
        "features_monitored": len(data["features"]) if data else 0,
        "drift_features": sum(1 for a in alerts if a["type"] == "DATA_DRIFT"),
        "alert_count": len(alerts),
        "alerts": alerts,
        "error": error,
    }
    if use_mlflow:
        try:
            _log_mlflow(root, record, alpha)
        except Exception as exc:
            logger.error("Falha no MLflow: %s", type(exc).__name__)
            record["status"] = "FAILED"
            record["error"] = f"MLflow indisponivel: {type(exc).__name__}"
            record["alerts"].append({
                "level": "CRITICAL", "type": "PIPELINE_FAILURE", "message": record["error"]
            })
            record["alert_count"] = len(record["alerts"])
    record["duration_seconds"] = round(time.perf_counter() - start, 4)
    _persist(root, record)
    logger.info("Monitoramento finalizado | status=%s | alertas=%s", record["status"], record["alert_count"])
    return record
