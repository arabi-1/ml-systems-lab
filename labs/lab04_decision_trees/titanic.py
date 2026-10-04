"""Task 1: predict Titanic survival with a decision tree."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).parent / "data" / "raw" / "train.csv"
NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Survived"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Kaggle Titanic training CSV."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See labs/lab04_decision_trees/data/README.md")
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature columns from the survival label."""
    return df[FEATURES].copy(), df[TARGET].copy()


def build_model(max_depth: int | None = 4, random_state: int = 42) -> Pipeline:
    """Imputation, one-hot encoding, and a decision tree, fitted as one unit.

    Imputers learn their fill values from the training split only, which avoids
    leaking test-set information into the model.
    """
    preprocess = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("tree", DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)),
        ]
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    """Return test accuracy, a classification report, and a confusion matrix."""
    predictions = model.predict(X_test)
    return {
        "accuracy": model.score(X_test, y_test),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


def main() -> None:
    """Train and evaluate the Titanic survival model."""
    data = load_data()
    X, y = split_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)
    results = evaluate(model, X_test, y_test)

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["tree"].feature_importances_
    top_features = sorted(zip(importances, feature_names, strict=True), reverse=True)[:5]

    print(f"Dataset shape: {data.shape}")
    print(f"Train/test sizes: {len(X_train)}/{len(X_test)}")
    print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
    print(f"TEST accuracy (model score): {results['accuracy']:.3f}")
    print("Classification report:")
    print(results["classification_report"])
    print("Confusion matrix:")
    print(results["confusion_matrix"])
    print("Top 5 feature importances:")
    for importance, feature_name in top_features:
        print(f"  {feature_name}: {importance:.3f}")


if __name__ == "__main__":
    main()
