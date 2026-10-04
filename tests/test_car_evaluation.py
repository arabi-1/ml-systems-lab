import pandas as pd

from labs.lab04_decision_trees.car_evaluation import (
    CLASS_LABELS,
    FEATURES,
    TARGET,
    build_model,
    evaluate,
    split_xy,
)


def make_synthetic_data() -> pd.DataFrame:
    """Build a small Car Evaluation-shaped dataset for tests."""
    rows = []
    categories = [
        ["vhigh", "high", "med", "low"],
        ["vhigh", "high", "med", "low"],
        ["2", "3", "4", "5more"],
        ["2", "4", "more"],
        ["small", "med", "big"],
        ["low", "med", "high"],
    ]
    for index in range(32):
        row = {
            feature: categories[position][index % len(categories[position])]
            for position, feature in enumerate(FEATURES)
        }
        row[TARGET] = CLASS_LABELS[index % len(CLASS_LABELS)]
        rows.append(row)
    return pd.DataFrame(rows)


def test_split_xy_returns_features_and_target() -> None:
    data = make_synthetic_data()

    features, target = split_xy(data)

    assert list(features.columns) == FEATURES
    assert target.name == TARGET
    assert len(features) == len(target) == 32


def test_model_predicts_known_class_for_each_row() -> None:
    features, target = split_xy(make_synthetic_data())

    predictions = build_model().fit(features, target).predict(features)

    assert len(predictions) == len(features)
    assert set(predictions).issubset(set(CLASS_LABELS))


def test_model_predicts_extreme_category_values() -> None:
    data = pd.DataFrame(
        [
            {
                "buying": "low",
                "maint": "low",
                "doors": "5more",
                "persons": "more",
                "lug_boot": "big",
                "safety": "high",
                "class": "vgood",
            }
        ]
    )
    features, target = split_xy(pd.concat([make_synthetic_data(), data], ignore_index=True))
    model = build_model().fit(features, target)

    predictions = model.predict(data[FEATURES])

    assert len(predictions) == 1
    assert predictions[0] in CLASS_LABELS


def test_evaluate_returns_accuracy_between_zero_and_one() -> None:
    features, target = split_xy(make_synthetic_data())
    model = build_model().fit(features, target)

    results = evaluate(model, features, target)

    assert 0 <= results["accuracy"] <= 1
