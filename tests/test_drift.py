import pandas as pd
import pytest
from src.drift.detector import ks_drift
from src.drift.simulator import inject_drift, split_reference_production


def test_simulation_is_reproducible_and_disjoint():
    df = pd.DataFrame({"SK_ID_CURR": range(100), "AMT_INCOME_TOTAL": range(1, 101), "AMT_CREDIT": range(101, 201)})
    ref, prod = split_reference_production(df, sample_size=30, seed=42)
    ref2, prod2 = split_reference_production(df, sample_size=30, seed=42)
    assert set(ref.SK_ID_CURR).isdisjoint(set(prod.SK_ID_CURR))
    pd.testing.assert_frame_equal(ref, ref2)
    pd.testing.assert_frame_equal(prod, prod2)


def test_injection_changes_only_selected_features():
    df = pd.DataFrame({"SK_ID_CURR": [1, 2], "AMT_INCOME_TOTAL": [100., 200.], "AMT_CREDIT": [1000., 2000.], "TARGET": [0, 1]})
    changed = inject_drift(df)
    assert changed["AMT_INCOME_TOTAL"].tolist() == pytest.approx([135., 270.])
    assert changed["AMT_CREDIT"].tolist() == pytest.approx([1500., 3000.])
    assert changed["TARGET"].equals(df["TARGET"])
    assert df["AMT_INCOME_TOTAL"].tolist() == [100., 200.]


def test_ks_detects_large_shift():
    ref = pd.DataFrame({"x": list(range(100))})
    prod = pd.DataFrame({"x": list(range(1000, 1100))})
    assert ks_drift(ref, prod, ("x",))["x"]["drift_detected"] is True
