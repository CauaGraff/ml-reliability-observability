"""Alertas locais de drift e falha operacional; sem envio externo."""


def build_alerts(metrics: dict, alpha: float = 0.05) -> list[dict]:
    alerts = []
    for feature, result in metrics["features"].items():
        if result["p_value"] < alpha:
            alerts.append({
                "level": "WARNING",
                "type": "DATA_DRIFT",
                "feature": feature,
                "message": f"Mudança estatística detectada em {feature}",
                "ks_statistic": result["ks_statistic"],
                "p_value": result["p_value"],
            })
    return alerts
