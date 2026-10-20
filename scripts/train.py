"""Treina baseline SOMENTE depois da validação do contrato."""
import json
from pathlib import Path
import joblib
from pandera.errors import SchemaErrors
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from src.data.loader import load_credit_data
from src.models.preprocessing import make_features
from src.models.training import build_pipeline
from src.validation.contracts import validate_credit_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = load_credit_data(ROOT / "data/raw/application_train.csv")
    print(f"Validando lote de {len(df)} linhas...")
    try:
        df = validate_credit_data(df)
    except SchemaErrors as exc:
        print("TREINAMENTO BLOQUEADO: contrato de dados reprovado.")
        print(exc.failure_cases.head(15).to_string(index=False))
        raise SystemExit(1) from exc
    x, y = make_features(df), df["TARGET"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    model = build_pipeline()
    print(f"Treino: {len(x_train)} | Teste: {len(x_test)}")
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "train_rows": len(x_train), "test_rows": len(x_test),
        "features": list(x.columns),
    }
    print(classification_report(y_test, predicted, zero_division=0))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/credit_model.joblib")
    (ROOT / "reports/baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Modelo salvo em models/credit_model.joblib")
    print("Métricas salvas em reports/baseline_metrics.json")


if __name__ == "__main__":
    main()
