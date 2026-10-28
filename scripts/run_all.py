"""Orquestracao batch: validar, treinar, simular, detectar e monitorar."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.monitoring.logger import configure_logger
from src.monitoring.pipeline import record_pipeline_failure

ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    "scripts.validate_data",
    "scripts.train",
    "scripts.simulate_production",
    "scripts.detect_drift",
)


def _register_step(root: Path, step: str, status: str, duration: float) -> None:
    """Historico de duracao e status, sem dados pessoais."""
    path = root / "monitoring/metrics/stages_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "status": status,
        "duration_seconds": round(duration, 3),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa pipeline completo do Tech Challenge 4")
    parser.add_argument("--mlflow", action="store_true", help="Registra ultima etapa no MLflow")
    args = parser.parse_args()
    logger = configure_logger(ROOT / "monitoring/logs/monitoring.log")
    for module in STEPS:
        print(f"\n=== Executando {module} ===", flush=True)
        logger.info("Iniciando etapa %s", module)
        started = time.perf_counter()
        try:
            subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            duration = time.perf_counter() - started
            _register_step(ROOT, module, "FAILED", duration)
            record_pipeline_failure(ROOT, module, exc)
            print(f"Pipeline bloqueado por falha em {module}", file=sys.stderr)
            return 1
        duration = time.perf_counter() - started
        _register_step(ROOT, module, "SUCCESS", duration)
        logger.info("Etapa concluida %s | duracao=%.2fs", module, duration)

    command = [sys.executable, "-m", "scripts.run_monitoring"]
    if args.mlflow:
        command.append("--mlflow")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
