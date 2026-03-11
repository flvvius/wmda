"""Exercise 3 (10 minutes): Decision Tree Regression."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def create_data(n_samples=30):
    """Create synthetic dataset with multiple features and non-linear relationships."""
    np.random.seed(42)
    X = np.random.rand(n_samples, 3) * 10

    # True relationship: 2*Feature1 + 0.5*Feature2^2 - 3*Feature3 + noise
    true_y = 2 * X[:, 0] + 0.5 * (X[:, 1]**2) - 3 * X[:, 2]
    noise = np.random.normal(0, 5, size=n_samples)
    y = true_y + noise

    df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3"])
    df["Target"] = y
    return df


def train_tree(X_train, X_test, y_train, y_test, max_depth):
    """Train DecisionTreeRegressor with given max_depth."""
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=2,  # Prevent single-leaf nodes
        random_state=42
    )
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Also compute train performance to detect overfitting
    y_train_pred = tree.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)

    return tree, (r2, mse, mae), (r2_train, mse_train), y_pred


def main():
    # 1. Create dataset
    df = create_data(n_samples=30)
    X = df[["Feature1", "Feature2", "Feature3"]]
    y = df["Target"]

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("=" * 70)
    print("DECISION TREE REGRESSOR - IMPACT OF MAX_DEPTH")
    print("=" * 70)
    print(f"Dataset: n={len(X)}, train={len(X_train)}, test={len(X_test)}")
    print()

    # 3. Train models with different max_depth values
    depths = [1, 2, 3, 5, 10]
    results = {}

    for depth in depths:
        print("─" * 70)
        print(f"MAX_DEPTH = {depth}")
        print("─" * 70)

        tree, (r2_test, mse_test, mae_test), (r2_train, mse_train), y_pred = train_tree(
            X_train, X_test, y_train, y_test, max_depth=depth
        )
        results[depth] = (tree, r2_test, mse_test, mae_test, r2_train, mse_train)

        print(f"TRAIN performance:")
        print(f"  R²:  {r2_train:.4f}")
        print(f"  MSE: {mse_train:.4f}")
        print(f"TEST performance:")
        print(f"  R²:  {r2_test:.4f}")
        print(f"  MSE: {mse_test:.4f}")
        print(f"  MAE: {mae_test:.4f}")

        # Overfitting indicator
        gap = r2_train - r2_test
        if gap > 0.1:
            print(f"⚠️  OVERFITTING DETECTED: R² gap (train-test) = {gap:.4f}")
        else:
            print(f"✓ Generalization OK: R² gap = {gap:.4f}")

        print(f"Feature importances:")
        for i, importance in enumerate(tree.feature_importances_, 1):
            print(f"  Feature{i}: {importance:.4f}")
        print()

    # 4. Summary comparison
    print("=" * 70)
    print("SUMMARY: OVERFITTING ANALYSIS")
    print("=" * 70)
    print(f"{'Depth':<8} {'Train R²':<12} {'Test R²':<12} {'Overfitting Gap':<15}")
    print("─" * 70)
    for depth in depths:
        tree, r2_test, mse_test, mae_test, r2_train, mse_train = results[depth]
        gap = r2_train - r2_test
        print(f"{depth:<8} {r2_train:<12.4f} {r2_test:<12.4f} {gap:<15.4f}")

    # 5. Visualize the best tree (e.g., max_depth=3)
    print("─" * 70)
    print("Visualizing tree with max_depth=3...")
    print("─" * 70)
    best_depth = 3
    best_tree = results[best_depth][0]

    try:
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            best_tree,
            feature_names=["Feature1", "Feature2", "Feature3"],
            filled=True,
            rounded=True,
            ax=ax
        )
        plt.tight_layout()
        plt.savefig("decision_tree.png", dpi=100, bbox_inches="tight")
        print("✓ Saved: decision_tree.png")
    except Exception as e:
        print(f"Note: Could not save tree visualization: {e}")

    print()


if __name__ == "__main__":
    main()
