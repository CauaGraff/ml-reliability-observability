"""Leitura valida de metricas da Etapa 2 e historico do monitoramento."""
import json
import math
from pathlib import Path


def load_drift_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("features"), dict) or not data["features"]:
        raise ValueError("drift_metrics.json sem 'features' preenchido")
    for name, metric in data["features"].items():
        if not isinstance(metric, dict):
            raise ValueError(f"Metrica invalida: {name}")
        for key in ("ks_statistic", "p_value", "drift_detected"):
            if key not in metric:
                raise ValueError(f"Campo ausente: features.{name}.{key}")
        if type(metric["drift_detected"]) is not bool:
            raise ValueError(f"drift_detected precisa ser booleano: {name}")
        for key in ("ks_statistic", "p_value"):
            val = metric[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or not 0 <= val <= 1:
                raise ValueError(f"Metrica fora de [0,1]: {name}.{key}")
    for key in ("reference_rows", "production_rows"):
        count = data.get(key)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"Contagem invalida: {key}")
    return data


def append_history(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
