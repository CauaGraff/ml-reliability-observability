"""API apenas de leitura para saude operacional e Prometheus."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST

from app.metrics import render_prometheus

app = FastAPI(title="FIAP - Observabilidade de Credito", version="0.2.0")


def _latest_run() -> dict | None:
    path = Path(os.getenv("MONITORING_ROOT", "monitoring")) / "metrics/latest_run.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        required = ("status", "run_id", "timestamp_utc", "duration_seconds", "drift_features", "alert_count")
        if not isinstance(data, dict) or any(key not in data for key in required):
            return None
        return data
    except (OSError, ValueError):
        return None


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: o servidor HTTP esta respondendo."""
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness: última execução válida, bem-sucedida e recente."""
    result = _latest_run()
    if result is None or result["status"] != "SUCCESS":
        raise HTTPException(503, detail="Monitoramento ausente ou com falha")
    try:
        stamp = datetime.fromisoformat(result["timestamp_utc"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("timestamp sem fuso")
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
        max_age = int(os.getenv("MONITORING_MAX_AGE_SECONDS", "86400"))
        if not 0 <= age <= max_age:
            raise ValueError("monitoramento desatualizado")
    except (TypeError, ValueError):
        raise HTTPException(503, detail="Monitoramento antigo ou sem data valida") from None
    return {"status": "ready"}


@app.get("/status")
def status() -> dict:
    result = _latest_run()
    if result is None:
        raise HTTPException(503, detail="Nenhuma execucao valida registrada")
    # Lista branca para nao expor erros brutos, IDs de clientes ou payloads.
    fields = (
        "run_id", "timestamp_utc", "status", "duration_seconds", "reference_rows",
        "production_rows", "features_monitored", "drift_features", "alert_count",
    )
    return {field: result.get(field) for field in fields}


@app.get("/metrics")
def metrics() -> Response:
    result = _latest_run()
    if result is None:
        return Response("# Nenhuma execucao de monitoramento encontrada\n", status_code=503, media_type="text/plain")
    return Response(render_prometheus(result), media_type=CONTENT_TYPE_LATEST)
