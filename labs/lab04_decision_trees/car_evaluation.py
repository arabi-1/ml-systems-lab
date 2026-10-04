"""Task 2: predict car safety with a decision tree."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).parent / "data" / "raw" / "car.data"
COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
FEATURES = COLUMNS[:-1]
TARGET = "class"
CLASS_LABELS = ["unacc", "acc", "good", "vgood"]
CATEGORY_ORDERS = [
    ["vhigh", "high", "med", "low"],
    ["vhigh", "high", "med", "low"],
    ["2", "3", "4", "5more"],
    ["2", "4", "more"],
    ["small", "med", "big"],
    ["low", "med", "high"],
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the headerless UCI Car Evaluation data."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See labs/lab04_decision_trees/data/README.md")
    return pd.read_csv(path, header=None, names=COLUMNS)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the car features from the safety label."""
    return df[FEATURES].copy(), df[TARGET].copy()


def build_model(max_depth: int | None = None, random_state: int = 42) -> Pipeline:
    """Ordinally encode the ordered categories and fit a decision tree."""
    preprocess = ColumnTransformer(
        [("ordinal", OrdinalEncoder(categories=CATEGORY_ORDERS), FEATURES)]
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
        "classification_report": classification_report(
            y_test, predictions, labels=CLASS_LABELS, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CLASS_LABELS),
    }


def main() -> None:
    """Train and evaluate the Car Evaluation model."""
    data = load_data()
    X, y = split_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)
    results = evaluate(model, X_test, y_test)
    majority_class = y_train.mode()[0]
    baseline_accuracy = (y_test == majority_class).mean()
    tree = model.named_steps["tree"]
    importances = sorted(zip(tree.feature_importances_, FEATURES, strict=True), reverse=True)

    print(f"Dataset shape: {data.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(f"Train/test sizes: {len(X_train)}/{len(X_test)}")
    print(f"Majority-class baseline accuracy: {baseline_accuracy:.3f} ({majority_class})")
    print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
    print(f"TEST accuracy (model score): {results['accuracy']:.3f}")
    print("Classification report:")
    print(results["classification_report"])
    print("Confusion matrix (rows=true, columns=predicted; order: unacc, acc, good, vgood):")
    print(pd.DataFrame(results["confusion_matrix"], index=CLASS_LABELS, columns=CLASS_LABELS))
    print(f"Tree depth/leaves: {tree.get_depth()}/{tree.get_n_leaves()}")
    print("Feature importances:")
    for importance, feature_name in importances:
        print(f"  {feature_name}: {importance:.3f}")


if __name__ == "__main__":
    main()
