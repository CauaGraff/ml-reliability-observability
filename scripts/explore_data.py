from pathlib import Path
from src.data.loader import load_credit_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = load_credit_data(ROOT / "data/raw/application_train.csv")
    print("Dimensões:", df.shape)
    print("TARGET (contagem):\n", df["TARGET"].value_counts(dropna=False))
    print("TARGET (%):\n", df["TARGET"].value_counts(normalize=True, dropna=False).mul(100).round(2))
    print("Top 10 ausentes:\n", df.isna().sum().sort_values(ascending=False).head(10))
    print("IDs duplicados:", df["SK_ID_CURR"].duplicated().sum())
    print("Idade (anos):\n", (-df["DAYS_BIRTH"] / 365.25).describe())


if __name__ == "__main__":
    main()
