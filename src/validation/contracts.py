"""Data contracts em Pandera. O alvo nao e exigido na inferencia."""
import pandas as pd
import pandera.pandas as pa
from pandera import Check

BASE_COLUMNS = {
    "SK_ID_CURR": pa.Column(int, nullable=False, unique=True, coerce=True),
    "AMT_INCOME_TOTAL": pa.Column(float, checks=Check.ge(0), nullable=False, coerce=True),
    "AMT_CREDIT": pa.Column(float, checks=Check.gt(0), nullable=False, coerce=True),
    "AMT_ANNUITY": pa.Column(float, checks=Check.gt(0), nullable=True, coerce=True),
    "DAYS_BIRTH": pa.Column(
        int, checks=[Check.ge(-36525), Check.le(-6570)], nullable=False, coerce=True
    ),
    "DAYS_EMPLOYED": pa.Column(float, nullable=True, coerce=True),
}

credit_schema = pa.DataFrameSchema(
    {**BASE_COLUMNS, "TARGET": pa.Column(int, checks=Check.isin([0, 1]), nullable=False, coerce=True)},
    strict=False,
)

inference_schema = pa.DataFrameSchema(BASE_COLUMNS, strict=False)


def validate_credit_data(df: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    """Interrompe o processamento se alguma regra falhar."""
    schema = credit_schema if require_target else inference_schema
    validated = schema.validate(df, lazy=True)
    if not require_target and "TARGET" in validated.columns:
        # Caso um lote inclua TARGET, nao afirmamos que seja conhecido no instante da previsao.
        validated = validated.drop(columns=["TARGET"])
    return validated
