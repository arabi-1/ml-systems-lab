"""Compare linear and polynomial regression for position salaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


DATA_PATH = Path(__file__).parent / "data" / "Position_salaries.csv"
PLOT_PATH = Path.cwd() / "polynomial_vs_linear.png"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
	"""Load the position salary dataset."""
	return pd.read_csv(path)


def main() -> None:
	"""Train both models, save their comparison plot, and make predictions."""
	data = load_data()
	X = data[["Level"]]
	y = data["Salary"]

	linear_model = LinearRegression().fit(X, y)

	polynomial_features = PolynomialFeatures(degree=4)
	X_polynomial = polynomial_features.fit_transform(X)
	polynomial_model = LinearRegression().fit(X_polynomial, y)

	level_grid = pd.DataFrame(
		{"Level": np.linspace(X["Level"].min(), X["Level"].max(), 300)}
	)
	linear_predictions = linear_model.predict(level_grid)
	polynomial_predictions = polynomial_model.predict(
		polynomial_features.transform(level_grid)
	)

	plt.figure(figsize=(9, 6))
	plt.scatter(X["Level"], y, color="blue", label="Actual data")
	plt.plot(level_grid["Level"], linear_predictions, color="red", label="Linear Regression")
	plt.plot(
		level_grid["Level"],
		polynomial_predictions,
		color="green",
		label="Polynomial Regression (degree 4)",
	)
	plt.title("Polynomial vs Linear Regression")
	plt.xlabel("Level")
	plt.ylabel("Salary")
	plt.legend()
	plt.tight_layout()
	plt.savefig(PLOT_PATH)
	plt.close()

	candidate = pd.DataFrame({"Level": [6.5]})
	linear_salary = linear_model.predict(candidate)[0]
	polynomial_salary = polynomial_model.predict(polynomial_features.transform(candidate))[0]
	print(f"Predicted salary for level 6.5 (Linear Regression): ${linear_salary:,.2f}")
	print(f"Predicted salary for level 6.5 (Polynomial Regression): ${polynomial_salary:,.2f}")
	print(f"Plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
	main()
