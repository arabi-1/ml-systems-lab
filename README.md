# ml-systems-lab

Machine learning coursework implemented as tested, CI-checked Python code.

![CI](https://github.com/arabi-1/ml-systems-lab/actions/workflows/ci.yml/badge.svg)

## Labs

| Lab | Topic | Result |
|-----|-------|--------|
| [Lab 04, Task 1](labs/lab04_decision_trees/titanic.py) | Decision tree: Titanic survival | Test accuracy 0.793 |
| [Lab 04, Task 2](labs/lab04_decision_trees/car_evaluation.py) | Decision tree: UCI Car Evaluation | Test accuracy 0.991 (baseline 0.699) |
| [Lab 05, Task 1](labs/lab05_linear_regression/salary.py) | Linear regression: Salary vs Experience | Test R2 0.975 |
| [Lab 05, Task 2](labs/lab05_linear_regression/canada_income.py) | Linear regression: Canada per capita income | Test R2 0.875, test MAE 3,240.91 (baseline 8,428.37); 2020 prediction 41,027.68 (extrapolation beyond 1970-2016 training range) |

## Quick start

```bash
git clone https://github.com/arabi-1/ml-systems-lab.git
cd ml-systems-lab
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

Datasets are not committed. See the data READMEs for download instructions
([Lab 04](labs/lab04_decision_trees/data/README.md),
[Lab 05](labs/lab05_linear_regression/data/README.md)), then run a lab:

```bash
python -m labs.lab04_decision_trees.titanic
python -m labs.lab04_decision_trees.car_evaluation
python -m labs.lab05_linear_regression.salary
python -m labs.lab05_linear_regression.canada_income
```

## Layout

```
src/mlkit/    shared utilities
labs/         one package per lab
tests/        pytest suite (synthetic data, no downloads needed)
docs/adr/     architecture decision records
```

## Engineering practices

- Conventional Commits and one pull request per task
- `main` is protected: changes need a PR and passing CI (ruff lint/format + pytest on Python 3.11 and 3.12)
- Preprocessing lives inside scikit-learn pipelines and is fit on training data only, so no test-set leakage