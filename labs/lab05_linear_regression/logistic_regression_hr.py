"""Logistic regression for employee retention prediction."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "data" / "HR_comma_sep.csv"
FEATURES = [
    "satisfaction_level",
    "average_montly_hours",
    "promotion_last_5years",
    "salary",
]
TARGET = "left"
SALARY_VALUES = {"low": 0, "medium": 1, "high": 2}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the HR dataset from disk."""
    return pd.read_csv(path)


def save_eda_plots(data: pd.DataFrame, output_directory: Path = Path.cwd()) -> None:
    """Save retention charts grouped by salary and department."""
    salary_retention = pd.crosstab(data["salary"], data[TARGET])
    salary_retention = salary_retention.rename(columns={0: "Not left", 1: "Left"})
    salary_retention.plot(kind="bar", title="Employee Retention by Salary")
    plt.xlabel("Salary")
    plt.ylabel("Number of employees")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_directory / "salary_retention.png")
    plt.close()

    department_retention = pd.crosstab(data["Department"], data[TARGET])
    department_retention = department_retention.rename(columns={0: "Not left", 1: "Left"})
    department_retention.plot(kind="bar", title="Employee Retention by Department")
    plt.xlabel("Department")
    plt.ylabel("Number of employees")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_directory / "department_retention.png")
    plt.close()


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select model features and encode salary as an ordinal value."""
    features = data[FEATURES].copy()
    features["salary"] = features["salary"].map(SALARY_VALUES)
    return features, data[TARGET].copy()


def main() -> None:
    """Create EDA plots, train the model, and report test accuracy."""
    data = load_data()
    save_eda_plots(data)
    X, y = prepare_features(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
