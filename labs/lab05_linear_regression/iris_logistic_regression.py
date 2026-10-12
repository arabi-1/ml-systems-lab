"""Multiclass logistic regression for the Iris dataset."""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    """Train, evaluate, and demonstrate an Iris classification model."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="flower_category")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    predictions = model.predict(X_test.iloc[:5])
    print("Sample predictions:")
    for sample_number, (predicted, actual) in enumerate(
        zip(predictions, y_test.iloc[:5], strict=True), start=1
    ):
        predicted_name = iris.target_names[predicted]
        actual_name = iris.target_names[actual]
        print(f"Sample {sample_number}: predicted={predicted_name}, actual={actual_name}")


if __name__ == "__main__":
    main()
