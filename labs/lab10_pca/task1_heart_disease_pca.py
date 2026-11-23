"""Compare heart-disease classifiers with and without principal components."""

import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data" / "raw" / "heart.csv"
CONTINUOUS_COLUMNS = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak",
]


def load_heart_data() -> pd.DataFrame:
    """Load the CSV and show an initial inspection of its structure."""
    # Step 1: loading creates a DataFrame for inspecting rows and feature types.
    heart = pd.read_csv(DATA_PATH)
    print("\n1. Heart data loaded")
    print(f"Shape: {heart.shape}")
    print("\nHead:")
    print(heart.head())
    print("\nDtypes:")
    print(heart.dtypes)
    return heart


def remove_continuous_outliers(heart: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with an absolute Z-score above 3 on continuous features."""
    # Step 2: Z-scores identify unusually distant values in comparable scales.
    row_count_before = len(heart)
    z_scores = heart[CONTINUOUS_COLUMNS].apply(
        lambda column: (column - column.mean()) / column.std(ddof=0)
    )
    outlier_mask = z_scores.abs().gt(3).fillna(False).any(axis=1)
    filtered = heart.loc[~outlier_mask].copy()

    # Printing both counts shows how many rows extreme values removed.
    print("\n2. Continuous-feature outlier removal")
    print(f"Rows before: {row_count_before}")
    print(f"Rows after: {len(filtered)}")
    print(f"Rows removed: {row_count_before - len(filtered)}")
    return filtered


def encode_features(heart: pd.DataFrame) -> pd.DataFrame:
    """Encode binary text columns and one-hot encode the multiclass columns."""
    # Step 3: numeric features are required because the selected ML models use arrays.
    encoded = heart.copy()
    for column in ("Sex", "ExerciseAngina"):
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column])

    # Dropping one level per category avoids redundant linear predictors.
    encoded = pd.get_dummies(
        encoded,
        columns=["ChestPainType", "RestingECG", "ST_Slope"],
        drop_first=True,
        dtype=int,
    )
    print("\n3. Encoded feature columns")
    print(f"Final shape: {encoded.shape}")
    return encoded


def split_data(
    encoded: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separate X and y and create a stratified 80/20 split."""
    # Step 4: keeping the class proportions stable makes both splits comparable.
    X = encoded.drop(columns=["HeartDisease"]).to_numpy()
    y = encoded["HeartDisease"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=30,
        stratify=y,
    )
    print("\n4. Train/test split")
    print(f"Training shape: {X_train.shape}; test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaling on training data and transform both partitions."""
    # Step 5: fitting only on training data prevents test-data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\n5. Scaling complete")
    print("StandardScaler fit on training data only")
    return X_train_scaled, X_test_scaled, scaler


def make_models() -> dict[str, object]:
    """Create one fresh instance of each classifier."""
    # Separate model instances ensure PCA and non-PCA runs do not affect each other.
    return {
        "SVC": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=30),
    }


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    print_times: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    """Train classifiers, return accuracies and wall-clock training times."""
    accuracies: dict[str, float] = {}
    training_times: dict[str, float] = {}
    for model_name, model in make_models().items():
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start_time
        predictions = model.predict(X_test)
        accuracies[model_name] = float(accuracy_score(y_test, predictions))
        training_times[model_name] = elapsed
        if print_times:
            print(
                f"{model_name}: test accuracy={accuracies[model_name]:.4f}, "
                f"training time={elapsed:.4f}s"
            )
        else:
            print(f"{model_name}: test accuracy={accuracies[model_name]:.4f}")
    return accuracies, training_times


def apply_pca(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    """Fit PCA on the training partition and transform both partitions."""
    # Step 7: the PCA fit learns directions only from training data to avoid leakage.
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print("\n7. PCA applied")
    print(f"Original feature count: {X_train_scaled.shape[1]}")
    print(f"Number of components kept: {pca.n_components_}")
    print("Explained variance ratio:")
    print(np.array2string(pca.explained_variance_ratio_, precision=4))
    return pca, X_train_pca, X_test_pca


def print_comparison_table(
    baseline_accuracies: dict[str, float],
    pca_accuracies: dict[str, float],
    baseline_times: dict[str, float],
    pca_times: dict[str, float],
) -> None:
    """Print a final side-by-side accuracy and training-time comparison."""
    # Step 9: the table makes the accuracy-versus-computation trade-off visible.
    print("\n9. Final comparison table")
    header = (
        f"{'Model':<22}{'Without PCA':>14}{'With PCA':>12}"
        f"{'Difference':>12}{'Time no PCA':>16}{'Time PCA':>14}"
    )
    print(header)
    print("-" * len(header))
    for model_name in baseline_accuracies:
        without_pca = baseline_accuracies[model_name]
        with_pca = pca_accuracies[model_name]
        difference = with_pca - without_pca
        print(
            f"{model_name:<22}{without_pca:>14.4f}{with_pca:>12.4f}"
            f"{difference:>12.4f}{baseline_times[model_name]:>14.4f}s"
            f"{pca_times[model_name]:>12.4f}s"
        )


def save_accuracy_chart(
    baseline_accuracies: dict[str, float],
    pca_accuracies: dict[str, float],
) -> None:
    """Save a grouped bar chart comparing test accuracies."""
    # Step 10a: side-by-side bars expose whether PCA changed each model differently.
    model_names = list(baseline_accuracies)
    x_positions = np.arange(len(model_names))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        x_positions - width / 2,
        [baseline_accuracies[name] for name in model_names],
        width,
        label="Without PCA",
        color="tab:blue",
    )
    axis.bar(
        x_positions + width / 2,
        [pca_accuracies[name] for name in model_names],
        width,
        label="With PCA",
        color="tab:orange",
    )
    axis.set_xticks(x_positions, model_names)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Test Accuracy")
    axis.set_title("Heart-Disease Model Accuracy: With and Without PCA")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "heart_pca_accuracy_comparison.png", dpi=150)
    plt.close(fig)


def save_cumulative_variance_chart(
    X_train_scaled: np.ndarray,
    pca_95: PCA,
) -> None:
    """Save cumulative PCA variance and mark the 95% threshold."""
    # Step 10b: the curve shows how many components cover the retained variance.
    full_pca = PCA()
    full_pca.fit(X_train_scaled)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    component_numbers = np.arange(1, len(cumulative) + 1)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        component_numbers,
        cumulative,
        marker=".",
        linewidth=1.5,
        label="Cumulative explained variance",
    )
    axis.axhline(
        y=0.95,
        color="tab:red",
        linestyle="--",
        label="95% variance threshold",
    )
    axis.scatter(
        pca_95.n_components_,
        cumulative[pca_95.n_components_ - 1],
        color="tab:green",
        zorder=5,
        label=f"Chosen components: {pca_95.n_components_}",
    )
    axis.set_xlabel("Number of Principal Components")
    axis.set_ylabel("Cumulative Explained Variance")
    axis.set_title("Cumulative Explained Variance for Heart Features")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "heart_pca_cumulative_variance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the complete heart-disease PCA classification task."""
    heart = load_heart_data()
    heart = remove_continuous_outliers(heart)
    encoded = encode_features(heart)
    X_train, X_test, y_train, y_test = split_data(encoded)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # Step 6: baseline models use all scaled features and provide the comparison.
    print("\n6. Baseline models (without PCA)")
    baseline_accuracies, baseline_times = train_and_evaluate(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        print_times=False,
    )
    best_baseline = max(baseline_accuracies, key=baseline_accuracies.get)
    print(f"Best baseline model: {best_baseline}")

    pca_95, X_train_pca, X_test_pca = apply_pca(
        X_train_scaled,
        X_test_scaled,
    )

    # Step 8: retraining fresh models shows the effect of PCA's fewer features.
    print("\n8. PCA models")
    pca_accuracies, pca_times = train_and_evaluate(
        X_train_pca,
        X_test_pca,
        y_train,
        y_test,
        print_times=True,
    )

    print_comparison_table(
        baseline_accuracies,
        pca_accuracies,
        baseline_times,
        pca_times,
    )
    save_accuracy_chart(baseline_accuracies, pca_accuracies)
    save_cumulative_variance_chart(X_train_scaled, pca_95)
    print("\n10. Saved PNG plots:")
    print(f"- {SCRIPT_DIR / 'heart_pca_accuracy_comparison.png'}")
    print(f"- {SCRIPT_DIR / 'heart_pca_cumulative_variance.png'}")

    # Step 11: summarize the central educational trade-off for students.
    print("\nNote: PCA may lower accuracy slightly but reduces dimensions and computation.")


if __name__ == "__main__":
    main()
