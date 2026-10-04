"""Task 1: predict Titanic survival with a decision tree."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
