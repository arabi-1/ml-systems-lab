import numpy as np
import pandas as pd

from labs.lab04_decision_trees.titanic import FEATURES, TARGET, build_model, evaluate, split_xy


def make_synthetic_data() -> pd.DataFrame:
    """Build a small Titanic-shaped dataset for tests."""
    rows = []
    for index in range(30):
        rows.append(
            {
                "Pclass": (index % 3) + 1,
                "Age": np.nan if index % 7 == 0 else 20 + index,
                "SibSp": index % 2,
                "Parch": index % 3,
                "Fare": 7.25 + index,
                "Sex": "female" if index % 2 else "male",
                "Embarked": np.nan if index % 6 == 0 else ("S" if index % 2 else "C"),
                "Survived": index % 2,
            }
        )
    return pd.DataFrame(rows)


def test_split_xy_returns_features_and_target() -> None:
    data = make_synthetic_data()

    features, target = split_xy(data)

    assert list(features.columns) == FEATURES
    assert target.name == TARGET
    assert len(features) == len(target) == 30


def test_model_predicts_one_binary_value_per_row() -> None:
    features, target = split_xy(make_synthetic_data())

    predictions = build_model().fit(features, target).predict(features)

    assert len(predictions) == len(features)
    assert set(predictions).issubset({0, 1})


def test_model_handles_missing_values() -> None:
    features, target = split_xy(make_synthetic_data())

    build_model().fit(features, target)


def test_evaluate_returns_accuracy_between_zero_and_one() -> None:
    features, target = split_xy(make_synthetic_data())
    model = build_model().fit(features, target)

    results = evaluate(model, features, target)

    assert 0 <= results["accuracy"] <= 1
