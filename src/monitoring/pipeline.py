"""Executa observabilidade usando as saídas já geradas pela Etapa 2."""
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
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def run_monitoring(root: Path, *, alpha: float = 0.05, use_mlflow: bool = False) -> dict:
    """Não reexecuta a inferência: monitora o último lote da Etapa 2."""
    root = Path(root)
    logger = configure_logger(root / "monitoring/logs/monitoring.log")
    start = time.perf_counter()
    now = datetime.now(timezone.utc).isoformat()
    run_id = uuid4().hex
    status = "SUCCESS"
    alerts = []
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
            raise FileNotFoundError("Artefatos ausentes ou vazios: " + ", ".join(missing))
        data = load_drift_metrics(expected[0])
        alerts = build_alerts(data, alpha=alpha)
        for alert in alerts:
            logger.warning("Alerta %s | feature=%s | KS=%.4f | p=%.3g", alert["type"], alert["feature"], alert["ks_statistic"], alert["p_value"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        status = "FAILED"
        error = f"{type(exc).__name__}: {exc}"
        alerts = [{"level": "CRITICAL", "type": "PIPELINE_FAILURE", "message": error}]
        logger.error("Falha no monitoramento | %s", error)

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
    _write_json(root / "monitoring/metrics/latest_run.json", record)
    _write_json(root / "monitoring/alerts/latest_alerts.json", {"run_id": run_id, "timestamp_utc": now, "alerts": alerts})
    append_history(root / "monitoring/metrics/history.jsonl", record)
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri((root / "monitoring/mlruns").resolve().as_uri())
            tracking_db = Path(root) / "monitoring" / "mlflow.db"
            tracking_db.parent.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_db.resolve().as_uri().replace("file:///", "sqlite:///"))
            mlflow.set_experiment("fiap-fase4-observabilidade")
            with mlflow.start_run(run_name=f"monitoramento-{run_id[:8]}"):
                mlflow.log_params({"ks_alpha": alpha, "monitoring_type": "batch"})
                mlflow.log_metrics({
                    "duration_seconds": record["duration_seconds"],
                    "features_monitored": record["features_monitored"],
                    "drift_features": record["drift_features"],
                    "alert_count": record["alert_count"],
                    "pipeline_success": int(status == "SUCCESS"),
                })
                mlflow.set_tags({"pipeline_status": status, "source": "fiap-fase4"})
        except Exception:
            logger.exception("Falha ao registrar métricas no MLflow")
            raise
    logger.info("Monitoramento finalizado | status=%s | alertas=%s | duração=%.3fs", status, len(alerts), record["duration_seconds"])
    return record
