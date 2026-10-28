"""Predicoes, KS e relatorio visual Evidently com features anonimizadas."""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data.loader import load_credit_data
from src.drift.detector import ks_drift, save_evidently_report
from src.drift.simulator import DRIFT_COLUMNS
from src.models.preprocessing import make_features
from src.validation.contracts import validate_credit_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    reference_path = ROOT / "data/production/credit_reference.csv"
    production_path = ROOT / "data/production/credit_production.csv"
    model_path = ROOT / "models/credit_model.joblib"
    for path in (reference_path, production_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo ausente: {path}. Execute treino/simulacao antes")

    reference = validate_credit_data(load_credit_data(reference_path, require_target=False), require_target=False)
    production = validate_credit_data(load_credit_data(production_path, require_target=False), require_target=False)
    # ATENCAO: joblib.load usa pickle. Carregue apenas modelo proprio e confiavel.
    model = joblib.load(model_path)
    x_prod = make_features(production)
    probabilities = model.predict_proba(x_prod)[:, 1]
    predictions = model.predict(x_prod)
    metrics = ks_drift(reference, production, DRIFT_COLUMNS)

    out = ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    # Predicoes individuais sao locais e NAO versionadas no Git.
    pd.DataFrame({"prediction": predictions, "risk_probability": probabilities}).to_csv(
        ROOT / "data/production/credit_predictions.csv", index=False
    )
    # Report usa apenas 5 features numericas; nunca IDs, alvo ou PII textual.
    save_evidently_report(make_features(reference), x_prod, out / "drift_report.html")

    summary = {
        "scenario": "drift sintetico: renda +35% e credito +50%",
        "reference_rows": int(len(reference)),
        "production_rows": int(len(production)),
        "ks_alpha": 0.05,
        "features": metrics,
        "predicted_positive_rate": float(predictions.mean()),
        "mean_predicted_risk": float(probabilities.mean()),
        "note": "Drift de dados simulado; nao comprova concept drift ou queda de performance real.",
    }
    (out / "drift_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    for feature, value in metrics.items():
        print(f"{feature}: KS={value['ks_statistic']:.4f} | p={value['p_value']:.3g} | drift={value['drift_detected']}")
    print(f"Relatorio HTML: {out / 'drift_report.html'}")
    print("Metricas JSON: reports/drift_metrics.json")
    print("Predicoes CSV: data/production/credit_predictions.csv")


if __name__ == "__main__":
    main()
