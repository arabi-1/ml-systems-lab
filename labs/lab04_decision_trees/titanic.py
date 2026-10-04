"""Task 1: predict Titanic survival with a decision tree."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "raw" / "train.csv"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = "Survived"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Kaggle Titanic training CSV."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See labs/lab04_decision_trees/data/README.md")
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature columns from the survival label."""
    return df[FEATURES].copy(), df[TARGET].copy()
