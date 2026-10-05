import pandas as pd
import pytest

from labs.lab05_linear_regression import canada_income


def make_synthetic_data() -> pd.DataFrame:
    """Build income data with a known linear relationship."""
    return pd.DataFrame(
        {"year": [1970, 1980, 1990, 2000, 2010], "per capita income (US$)": [10, 30, 50, 70, 90]}
    )


def test_split_xy_keeps_feature_names() -> None:
    features, target = canada_income.split_xy(make_synthetic_data())

    assert list(features.columns) == canada_income.FEATURES
    assert target.name == canada_income.TARGET


def test_model_evaluation_and_year_prediction() -> None:
    features, target = canada_income.split_xy(make_synthetic_data())
    model = canada_income.build_model().fit(features, target)
    results = canada_income.evaluate(model, features, target)

    assert results["r2"] == pytest.approx(1)
    assert results["mae"] == pytest.approx(0)
    assert canada_income.predict_year(model, 2020) == pytest.approx(110)


def test_load_data_reports_documentation_for_missing_file(tmp_path: object) -> None:
    with pytest.raises(FileNotFoundError, match="data/README\\.md"):
        canada_income.load_data(tmp_path / "missing.csv")  # type: ignore[arg-type]
