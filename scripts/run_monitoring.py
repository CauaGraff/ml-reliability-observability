"""Uso: py -m poetry run python -m scripts.run_monitoring [--mlflow]"""
import argparse
from pathlib import Path
from src.monitoring.pipeline import run_monitoring

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitora a execução da Etapa 2")
    parser.add_argument("--mlflow", action="store_true", help="Registra métricas também no MLflow (opcional)")
    args = parser.parse_args()
    record = run_monitoring(ROOT, use_mlflow=args.mlflow)
    print(f"Status: {record['status']} | Alertas: {record['alert_count']} | Drift: {record['drift_features']} feature(s)")
    print("Logs: monitoring/logs/monitoring.log")
    print("Métricas: monitoring/metrics/latest_run.json")
    print("Histórico: monitoring/metrics/history.jsonl")
    print("Alertas: monitoring/alerts/latest_alerts.json")
    return 0 if record["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
