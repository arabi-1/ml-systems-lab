"""Linear regression for Canada's per capita income."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "data" / "raw" / "canada_per_capita_income.csv"
FEATURES = ["year"]
TARGET = "per capita income (US$)"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load Canada's per capita income dataset."""
    if not path.exists():
        message = f"{path} not found. See labs/lab05_linear_regression/data/README.md"
        raise FileNotFoundError(message)
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate years and per capita income."""
    return df[FEATURES].copy(), df[TARGET].copy()


def build_model() -> LinearRegression:
    """Create an unconfigured linear regression model."""
    return LinearRegression()


def evaluate(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Return R2 and mean absolute error for a test split."""
    predictions = model.predict(X_test)
    return {"r2": r2_score(y_test, predictions), "mae": mean_absolute_error(y_test, predictions)}


def predict_year(model: LinearRegression, year: int | float) -> float:
    """Predict per capita income for a year."""
    return float(model.predict(pd.DataFrame({"year": [year]}))[0])


def plot_results(model: LinearRegression, X: pd.DataFrame, y: pd.Series) -> None:
    """Plot all observations and the fitted income trend."""
    feature = FEATURES[0]
    line = pd.DataFrame({feature: sorted(X[feature])})
    plt.figure()
    plt.scatter(X[feature], y, color="red")
    plt.plot(line[feature], model.predict(line), color="blue")
    plt.title("Canada Per Capita Income")
    plt.xlabel("Year")
    plt.ylabel(TARGET)


def main() -> None:
    """Train, evaluate, and plot Canada's income regression model."""
    data = load_data()
    X, y = split_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model().fit(X_train, y_train)
    results = evaluate(model, X_test, y_test)
    baseline_predictions = [y_train.mean()] * len(y_test)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    # 2020 is outside the observed 1970 to 2016 training range.
    prediction = predict_year(model, 2020)

    print(f"Train R2: {model.score(X_train, y_train):.3f}")
    print(f"TEST R2: {results['r2']:.3f}")
    print(f"TEST MAE: {results['mae']:.2f}")
    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"Coefficient: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print("Note: 2020 is outside the training range (1970 to 2016); this is an extrapolation.")
    print(f"Predicted per capita income for 2020: {prediction:.2f}")
    plot_results(model, X, y)
    plt.show()


if __name__ == "__main__":
    main()
