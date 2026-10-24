"""Executa inferência, calcula KS e exporta relatório HTML Evidently."""
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


def main():
    reference_path = ROOT / "data/production/credit_reference.csv"
    production_path = ROOT / "data/production/credit_production.csv"
    model_path = ROOT / "models/credit_model.joblib"
    for path in (reference_path, production_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Arquivo ausente: {path}. Execute treinamento/simulação antes.")

    reference = validate_credit_data(load_credit_data(reference_path))
    production = validate_credit_data(load_credit_data(production_path))
    model = joblib.load(model_path)  # Carregue somente modelos criados por você / fontes confiáveis.
    probabilities = model.predict_proba(make_features(production))[:, 1]
    predictions = model.predict(make_features(production))
    outputs = ROOT / "reports"
    outputs.mkdir(parents=True, exist_ok=True)
    prediction_table = pd.DataFrame({
        "SK_ID_CURR": production["SK_ID_CURR"].to_numpy(),
        "prediction": predictions,
        "risk_probability": probabilities,
    })
    prediction_table.to_csv(ROOT / "data/production/credit_predictions.csv", index=False)

    drift_metrics = ks_drift(reference, production, DRIFT_COLUMNS)
    # Apenas as features selecionadas: evita comparar IDs e o rótulo TARGET.
    ref_features = make_features(reference)
    prod_features = make_features(production)
    report_path = outputs / "drift_report.html"
    save_evidently_report(ref_features, prod_features, report_path)

    summary = {
        "scenario": "drift sintético: renda +35% e crédito +50%",
        "reference_rows": int(len(reference)),
        "production_rows": int(len(production)),
        "ks_alpha": 0.05,
        "features": drift_metrics,
        "predicted_positive_rate": float(predictions.mean()),
        "mean_predicted_risk": float(probabilities.mean()),
        "note": "Drift sintético não demonstra degradação real do modelo nem concept drift.",
    }
    (outputs / "drift_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for feature, metrics in drift_metrics.items():
        print(f"{feature}: KS={metrics['ks_statistic']:.4f} | p={metrics['p_value']:.3g} | drift={metrics['drift_detected']}")
    print(f"Relatório HTML: {report_path}")
    print("Métricas JSON: reports/drift_metrics.json")
    print("Previsões CSV: data/production/credit_predictions.csv")


if __name__ == "__main__":
    main()
