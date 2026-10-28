"""Exposicao pull de metricas agregadas; sem registros individuais."""
from datetime import datetime

from prometheus_client import CollectorRegistry, Gauge, generate_latest


def render_prometheus(record: dict) -> bytes:
    registry = CollectorRegistry()

    def gauge(name: str, desc: str, value: float) -> None:
        Gauge(name, desc, registry=registry).set(value)

    gauge("credit_pipeline_success", "1 quando a ultima verificacao terminou com sucesso", float(record["status"] == "SUCCESS"))
    gauge("credit_drift_features", "Quantidade de features com drift na ultima verificacao", float(record["drift_features"]))
    gauge("credit_active_alerts", "Quantidade de alertas da ultima verificacao", float(record["alert_count"]))
    gauge("credit_pipeline_duration_seconds", "Duracao da ultima verificacao de monitoramento, segundos", float(record["duration_seconds"]))
    gauge("credit_production_rows", "Registros do ultimo lote analisado", float(record.get("production_rows") or 0))
    dt = datetime.fromisoformat(record["timestamp_utc"].replace("Z", "+00:00"))
    gauge("credit_pipeline_last_run_timestamp_seconds", "Instante Unix da ultima verificacao", dt.timestamp())
    return generate_latest(registry)
