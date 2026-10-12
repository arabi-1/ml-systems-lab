"""Multiple linear regression for employee salary prediction."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression


DATA_PATH = Path(__file__).parent / "data" / "hiring.csv"
FEATURES = ["experience", "test_score", "interview_score"]
TARGET = "salary"
EXPERIENCE_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and preprocess the hiring dataset."""
    data = pd.read_csv(path)
    data["experience"] = data["experience"].fillna(0)
    data["test_score"] = data["test_score"].fillna(data["test_score"].median())
    data["experience"] = data["experience"].replace(EXPERIENCE_WORDS).astype(int)
    return data


def main() -> None:
    """Train the model on all rows and predict salaries for two candidates."""
    data = load_data()
    model = LinearRegression().fit(data[FEATURES], data[TARGET])

    candidates = pd.DataFrame(
        {
            "experience": [2, 12],
            "test_score": [9, 10],
            "interview_score": [6, 10],
        }
    )
    predictions = model.predict(candidates)

    for number, salary in enumerate(predictions, start=1):
        print(f"Candidate {number} predicted salary: ${salary:,.2f}")


if __name__ == "__main__":
    main()