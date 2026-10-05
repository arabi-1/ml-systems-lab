import pandas as pd
import pytest

from labs.lab05_linear_regression import salary


def make_synthetic_data() -> pd.DataFrame:
    """Build salary data with a known linear relationship."""
    return pd.DataFrame({"YearsExperience": [1, 2, 3, 4, 5, 6], "Salary": [3, 5, 7, 9, 11, 13]})


def test_split_xy_keeps_feature_names() -> None:
    features, target = salary.split_xy(make_synthetic_data())

    assert list(features.columns) == salary.FEATURES
    assert target.name == salary.TARGET


def test_model_and_evaluation() -> None:
    features, target = salary.split_xy(make_synthetic_data())
    model = salary.build_model().fit(features, target)
    results = salary.evaluate(model, features, target)

    assert model.coef_[0] == pytest.approx(2)
    assert model.intercept_ == pytest.approx(1)
    assert results["r2"] == pytest.approx(1)
    assert results["mae"] == pytest.approx(0)


def test_required_split_is_reproducible() -> None:
    features, target = salary.split_xy(make_synthetic_data())
    first = salary.train_test_split(features, target, test_size=1 / 3, random_state=0)
    second = salary.train_test_split(features, target, test_size=1 / 3, random_state=0)

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_series_equal(first[2], second[2])
    pd.testing.assert_series_equal(first[3], second[3])


def test_load_data_reports_documentation_for_missing_file(tmp_path: object) -> None:
    with pytest.raises(FileNotFoundError, match="data/README\\.md"):
        salary.load_data(tmp_path / "missing.csv")  # type: ignore[arg-type]
