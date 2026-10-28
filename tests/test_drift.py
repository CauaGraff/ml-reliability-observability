import pandas as pd
import pytest

from src.drift.detector import ks_drift
from src.drift.simulator import PRODUCTION_COLUMNS, inject_drift, split_reference_production


def production_sample(rows=100):
    return pd.DataFrame({
        "SK_ID_CURR": list(range(1000, 1000 + rows)),
        "AMT_INCOME_TOTAL": list(range(1, 1 + rows)),
        "AMT_CREDIT": list(range(101, 101 + rows)),
        "AMT_ANNUITY": [300.0] * rows,
        "DAYS_BIRTH": [-10000] * rows,
        "DAYS_EMPLOYED": [-500] * rows,
        "TARGET": [i % 2 for i in range(rows)],
        "NAME_INCOME_TYPE": ["Working"] * rows,
    })


def test_simulation_is_reproducible_and_disjoint():
    df = production_sample(100)
    ref, prod = split_reference_production(df, sample_size=30, seed=42)
    ref2, prod2 = split_reference_production(df, sample_size=30, seed=42)
    assert set(ref.SK_ID_CURR).isdisjoint(set(prod.SK_ID_CURR))
    assert set(ref.columns) == set(PRODUCTION_COLUMNS)
    assert "TARGET" not in ref.columns
    pd.testing.assert_frame_equal(ref, ref2)
    pd.testing.assert_frame_equal(prod, prod2)


def test_injection_changes_only_selected_features():
    df = production_sample(2)
    original_credit = df["AMT_CREDIT"].copy()
    changed = inject_drift(df)
    assert changed["AMT_INCOME_TOTAL"].tolist() == pytest.approx([1.35, 2.7])
    assert changed["AMT_CREDIT"].tolist() == pytest.approx([151.5, 153])
    assert changed["TARGET"].equals(df["TARGET"])
    assert df["AMT_CREDIT"].equals(original_credit)


def test_ks_detects_large_shift():
    ref = pd.DataFrame({"x": list(range(100))})
    prod = pd.DataFrame({"x": list(range(1000, 1100))})
    assert ks_drift(ref, prod, ("x",))["x"]["drift_detected"] is True
