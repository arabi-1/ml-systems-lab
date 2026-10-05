"""Simple linear regression for salary prediction from years of experience."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "data" / "raw" / "Salary_Data.csv"
FEATURES = ["YearsExperience"]
TARGET = "Salary"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the salary dataset."""
    if not path.exists():
        message = f"{path} not found. See labs/lab05_linear_regression/data/README.md"
        raise FileNotFoundError(message)
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate salary features and target."""
    return df[FEATURES].copy(), df[TARGET].copy()


def build_model() -> LinearRegression:
    """Create an unconfigured linear regression model."""
    return LinearRegression()


def evaluate(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Return R2 and mean absolute error for a test split."""
    predictions = model.predict(X_test)
    return {"r2": r2_score(y_test, predictions), "mae": mean_absolute_error(y_test, predictions)}


def plot_results(
    model: LinearRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Plot the fitted line against the training and test observations."""
    feature = FEATURES[0]
    train_line = X_train.sort_values(feature)
    test_line = X_test.sort_values(feature)

    for X_values, y_values, line_values in (
        (X_train, y_train, train_line),
        (X_test, y_test, test_line),
    ):
        plt.figure()
        plt.scatter(X_values[feature], y_values, color="red")
        plt.plot(line_values[feature], model.predict(line_values), color="blue")
        plt.title("Salary vs Experience")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")


def main() -> None:
    """Train, evaluate, and plot the salary regression model."""
    data = load_data()
    X, y = split_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
    model = build_model().fit(X_train, y_train)
    results = evaluate(model, X_test, y_test)
    baseline_predictions = [y_train.mean()] * len(y_test)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    prediction = model.predict(pd.DataFrame({"YearsExperience": [12]}))[0]

    print(f"Train R2: {model.score(X_train, y_train):.3f}")
    print(f"TEST R2: {results['r2']:.3f}")
    print(f"TEST MAE: {results['mae']:.2f}")
    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"Coefficient: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Prediction for 12 years: {prediction:.2f}")
    plot_results(model, X_train, y_train, X_test, y_test)
    plt.show()


if __name__ == "__main__":
    main()
