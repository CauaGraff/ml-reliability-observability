"""Regras simples e auditaveis para sinais de drift."""


def build_alerts(metrics: dict, alpha: float = 0.05) -> list[dict]:
    if not 0 < alpha < 1:
        raise ValueError("alpha deve estar entre 0 e 1")
    alerts = []
    for feature, result in metrics["features"].items():
        if result["p_value"] < alpha:
            alerts.append({
                "level": "WARNING",
                "type": "DATA_DRIFT",
                "feature": feature,
                "message": f"Diferenca estatistica detectada: {feature}",
                "ks_statistic": result["ks_statistic"],
                "p_value": result["p_value"],
            })
    return alerts
