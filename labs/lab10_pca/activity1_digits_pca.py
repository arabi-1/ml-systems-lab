"""PCA analysis of the sklearn handwritten digits dataset."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

matplotlib.use("Agg")

OUTPUT_DIR = Path(__file__).resolve().parent


def load_dataset() -> tuple[Bunch, np.ndarray, np.ndarray]:
    """Load the digits dataset and print its basic dimensions."""
    # Loading the data lets students inspect both the feature matrix and labels.
    digits = load_digits()

    # The shape reveals how many samples and pixel features are available.
    print(f"dataset.data.shape: {digits.data.shape}")

    # Reshaping a row to 8x8 shows why each feature represents one image pixel.
    print("First digit image (8x8):")
    print(digits.images[0])

    return digits, digits.data, digits.target


def save_digit_examples(digits: Bunch) -> None:
    """Save side-by-side images for samples 0 and 9."""
    # Comparing two examples makes the 64-dimensional pixel representation visible.
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    for axis, sample_index in zip(axes, (0, 9), strict=True):
        axis.imshow(digits.images[sample_index], cmap="gray")
        axis.set_title(f"Digit: {digits.target[sample_index]}")
        axis.axis("off")

    fig.suptitle("Digits Dataset Examples")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "digits_examples.png", dpi=150)
    plt.close(fig)


def build_feature_dataframe(digits: Bunch, X: np.ndarray) -> pd.DataFrame:
    """Create a labeled pandas table and summarize all pixel features."""
    # Feature names make each column understandable instead of an anonymous number.
    frame = pd.DataFrame(X, columns=digits.feature_names)
    print("\nDataFrame head():")
    print(frame.head())
    print("\nDataFrame describe():")
    print(frame.describe())
    return frame


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Split data, fit logistic regression, and return test accuracy."""
    # A fixed split gives comparable results across all PCA representations.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=30,
    )

    # Logistic regression learns a linear boundary between the digit classes.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))


def evaluate_baseline(X: np.ndarray, y: np.ndarray) -> float:
    """Scale all features, then train the 64-feature baseline model."""
    # Scaling gives each pixel feature similar numerical influence.
    X_scaled = StandardScaler().fit_transform(X)

    # The scaled feature matrix preserves all 64 original dimensions.
    accuracy = train_and_evaluate(X_scaled, y)
    print(f"\nBaseline test accuracy (64 scaled features): {accuracy:.4f}")
    return accuracy


def evaluate_pca_95(X: np.ndarray, y: np.ndarray) -> tuple[PCA, np.ndarray, float]:
    """Fit PCA for 95% variance and evaluate a logistic regression model."""
    # PCA combines correlated pixel features into orthogonal components.
    pca_95 = PCA(n_components=0.95)
    X_pca_95 = pca_95.fit_transform(X)

    # This printout shows the dimensionality reduction and retained component detail.
    print(f"\nPCA 95% component count: {pca_95.n_components_} out of {X.shape[1]}")
    print("PCA 95% explained_variance_ratio_:")
    print(pca_95.explained_variance_ratio_)

    # Transformed values replace the original pixels before classification.
    accuracy = train_and_evaluate(X_pca_95, y)
    print(f"PCA 95% test accuracy: {accuracy:.4f}")
    return pca_95, X_pca_95, accuracy


def evaluate_pca_2(X: np.ndarray, y: np.ndarray) -> tuple[PCA, np.ndarray, float]:
    """Fit two PCA components and evaluate their classification usefulness."""
    # Two components allow the dataset to be displayed in a 2D learning graph.
    pca_2 = PCA(n_components=2)
    X_pca_2 = pca_2.fit_transform(X)

    # The two ratios quantify how much total variation each plotted axis captures.
    print("\nPCA 2 explained_variance_ratio_:")
    print(pca_2.explained_variance_ratio_)
    total_variance = float(pca_2.explained_variance_ratio_.sum())
    print(f"PCA 2 total retained variance: {total_variance:.4f}")

    # Classification reveals whether two components retain useful class structure.
    accuracy = train_and_evaluate(X_pca_2, y)
    print(f"PCA 2 test accuracy: {accuracy:.4f}")
    return pca_2, X_pca_2, accuracy


def save_pca_scatter(X_pca_2: np.ndarray, y: np.ndarray) -> None:
    """Save a 2D PCA scatter plot with one color per digit."""
    # Points close together contain similar handwritten patterns in this projection.
    fig, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        X_pca_2[:, 0],
        X_pca_2[:, 1],
        c=y,
        cmap="tab10",
        alpha=0.75,
        edgecolor="none",
    )
    axis.set_xlabel("Principal Component 1")
    axis.set_ylabel("Principal Component 2")
    axis.set_title("Digits Dataset in the First Two PCA Components")
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Digit label")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "digits_pca_2d_scatter.png", dpi=150)
    plt.close(fig)


def save_cumulative_variance(X: np.ndarray, pca_95: PCA) -> None:
    """Plot cumulative variance and mark the 95% component threshold."""
    # A full PCA supplies enough components to explain all dataset variance.
    full_pca = PCA()
    full_pca.fit(X)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)

    # The curve helps students choose dimensions from a variance requirement.
    component_numbers = np.arange(1, len(cumulative_variance) + 1)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        component_numbers,
        cumulative_variance,
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
        cumulative_variance[pca_95.n_components_ - 1],
        color="tab:green",
        zorder=5,
        label=f"Chosen components: {pca_95.n_components_}",
    )
    axis.set_xlabel("Number of Principal Components")
    axis.set_ylabel("Cumulative Explained Variance")
    axis.set_title("PCA Cumulative Explained Variance")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "digits_pca_cumulative_variance.png", dpi=150)
    plt.close(fig)


def print_summary(
    baseline_accuracy: float,
    pca95_accuracy: float,
    pca2_accuracy: float,
    pca95_components: int,
) -> None:
    """Print a concise model-comparison table."""
    # Comparing accuracy and feature count makes the tradeoff from PCA explicit.
    print("\nFinal summary:")
    print(f"{'Model':<20}{'Features':>10}{'Test Accuracy':>16}")
    print("-" * 46)
    print(f"{'Baseline':<20}{64:>10}{baseline_accuracy:>16.4f}")
    print(f"{'PCA 95%':<20}{pca95_components:>10}{pca95_accuracy:>16.4f}")
    print(f"{'PCA 2':<20}{2:>10}{pca2_accuracy:>16.4f}")


def main() -> None:
    """Run the complete Lab 10 PCA activity in the requested order."""
    # Step 1: load and inspect the raw digits representation.
    digits, X, y = load_dataset()

    # Step 2: save representative digit images to show PCA's input format.
    save_digit_examples(digits)

    # Step 3: label and statistically summarize the 64 feature columns.
    build_feature_dataframe(digits, X)

    # Step 4: establish a baseline before applying dimensionality reduction.
    baseline_accuracy = evaluate_baseline(X, y)

    # Step 5: retain 95% of variance and measure the compact representation.
    pca_95, _, pca95_accuracy = evaluate_pca_95(X, y)

    # Step 6: reduce to two components and test how much structure remains.
    _, X_pca_2, pca2_accuracy = evaluate_pca_2(X, y)

    # Step 7: visualize digit groups in the first two principal components.
    save_pca_scatter(X_pca_2, y)

    # Step 8: relate component count to cumulative retained variance.
    save_cumulative_variance(X, pca_95)

    # Step 9: summarize the dimensionality and accuracy tradeoffs.
    print_summary(
        baseline_accuracy,
        pca95_accuracy,
        pca2_accuracy,
        pca_95.n_components_,
    )


if __name__ == "__main__":
    main()
