from pathlib import Path

import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

COLUMNS = [
    "pregnant",
    "glucose",
    "bp",
    "skin",
    "insulin",
    "bmi",
    "pedigree",
    "age",
    "label",
]
K_VALUES = [3, 5, 7, 9]


def load_diabetes_data(csv_path: Path) -> pd.DataFrame:
    """Load either supported diabetes CSV format into canonical columns."""
    raw_data = pd.read_csv(csv_path, header=None)
    first_row = [str(value).strip().lower() for value in raw_data.iloc[0]]
    expected_header = [
        "pregnancies",
        "glucose",
        "bloodpressure",
        "skinthickness",
        "insulin",
        "bmi",
        "diabetespedigreefunction",
        "age",
        "outcome",
    ]

    # Remove the header row when the file uses Kaggle-style column names.
    if first_row == expected_header:
        data = raw_data.iloc[1:].copy()
    else:
        data = raw_data

    data.columns = COLUMNS
    return data.apply(pd.to_numeric)


def main() -> None:
    # Load the dataset relative to this script so it works from any directory.
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "raw" / "diabetes.csv"
    data = load_diabetes_data(data_path)

    # Select the requested predictors and split the data into training and test sets.
    feature_columns = ["pregnant", "glucose", "bp", "insulin", "bmi", "age"]
    x_train, x_test, y_train, y_test = train_test_split(
        data[feature_columns],
        data["label"],
        test_size=0.3,
        random_state=1,
    )

    # Train and evaluate the default and pre-pruned decision trees.
    default_tree = DecisionTreeClassifier()
    default_tree.fit(x_train, y_train)
    default_accuracy = metrics.accuracy_score(y_test, default_tree.predict(x_test))
    print(f"Default Decision Tree accuracy: {default_accuracy:.4f}")

    pre_pruned_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    pre_pruned_tree.fit(x_train, y_train)
    pre_pruned_accuracy = metrics.accuracy_score(y_test, pre_pruned_tree.predict(x_test))
    print(f"Pre-Pruned Decision Tree accuracy: {pre_pruned_accuracy:.4f}")

    # Train each requested k-NN model and retain its test accuracy.
    knn_accuracies = {}
    for k in K_VALUES:
        knn_model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn_model.fit(x_train, y_train)
        accuracy = metrics.accuracy_score(y_test, knn_model.predict(x_test))
        knn_accuracies[k] = accuracy
        print(f"k-NN (k={k}) accuracy: {accuracy:.4f}")

    # Retrain the best k-NN model and report its detailed test results.
    best_k = min(K_VALUES, key=lambda value: (-knn_accuracies[value], value))
    best_knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    best_knn.fit(x_train, y_train)
    best_knn_predictions = best_knn.predict(x_test)
    best_knn_accuracy = metrics.accuracy_score(y_test, best_knn_predictions)
    print(f"\nBest k-NN k: {best_k}")
    print("Best k-NN confusion matrix:")
    print(confusion_matrix(y_test, best_knn_predictions))
    print("Best k-NN classification report:")
    print(classification_report(y_test, best_knn_predictions))

    # Compare the pre-pruned tree with the best k-NN model.
    if pre_pruned_accuracy > best_knn_accuracy:
        conclusion = "The Pre-Pruned Decision Tree achieved higher accuracy."
    elif best_knn_accuracy > pre_pruned_accuracy:
        conclusion = "The best k-NN achieved higher accuracy."
    else:
        conclusion = "The Pre-Pruned Decision Tree and best k-NN had equal accuracy."
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
