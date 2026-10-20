import numpy as np
from src.models.preprocessing import make_features
from tests.test_contracts import sample


def test_age_and_employment():
    features = make_features(sample())
    assert 27 < features.loc[0, "AGE_YEARS"] < 28
    assert np.isnan(features.loc[1, "EMPLOYED_YEARS"])
    assert "DAYS_BIRTH" not in features.columns
