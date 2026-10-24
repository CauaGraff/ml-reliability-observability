"""Métricas de drift independentes do formato do relatório Evidently."""
import pandas as pd
from scipy.stats import ks_2samp


def ks_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: tuple[str, ...], alpha: float = 0.05) -> dict:
    results = {}
    for column in columns:
        a = pd.to_numeric(reference[column], errors="coerce").dropna()
        b = pd.to_numeric(current[column], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            raise ValueError(f"Dados insuficientes para KS: {column}")
        statistic, pvalue = ks_2samp(a, b)
        results[column] = {
            "ks_statistic": float(statistic),
            "p_value": float(pvalue),
            "drift_detected": bool(pvalue < alpha),
            "reference_count": int(len(a)),
            "production_count": int(len(b)),
            "reference_median": float(a.median()),
            "production_median": float(b.median()),
        }
    return results


def save_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path) -> None:
    """API Evidently 0.7.x: Report.run retorna um snapshot exportável."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    snapshot.save_html(str(path))
