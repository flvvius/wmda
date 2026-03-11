"""Exercise 6 (10 minutes): k-NN for Regression.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def create_data(n_samples=40):
    """Create synthetic dataset with two continuous features."""
    np.random.seed(42)
    X = np.random.rand(n_samples, 2) * 10
    # True relationship: y = 3*Feature1 + 2*Feature2 + noise
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 5, size=n_samples)

    df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
    df["Target"] = y
    return df


def train_knn(X_train_scaled, X_test_scaled, y_train, y_test, k):
    """Train kNN regressor with given k and return metrics."""
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # Predictions on both train and test
    y_train_pred = knn.predict(X_train_scaled)
    y_test_pred = knn.predict(X_test_scaled)

    # Metrics on train set
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)

    # Metrics on test set
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    return knn, (r2_train, mse_train), (r2_test, mse_test, mae_test), y_test_pred


def main():
    # 1. Create dataset
    df = create_data(n_samples=40)
    X = df[["Feature1", "Feature2"]]
    y = df["Target"]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("=" * 80)
    print("k-NEAREST NEIGHBORS (kNN) FOR REGRESSION")
    print("=" * 80)
    print(f"Dataset: n={len(X)}, train={len(X_train)}, test={len(X_test)}")
    print(f"True relationship: y = 3*Feature1 + 2*Feature2 + noise")
    print()

    # 3. Feature scaling (CRITICAL for distance-based methods)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("─" * 80)
    print("FEATURE SCALING (StandardScaler)")
    print("─" * 80)
    print("Why scale for kNN?")
    print("  • kNN relies on distance metrics (Euclidean, Manhattan, etc.)")
    print("  • Features with large ranges dominate distance calculations")
    print("  • Scaling ensures all features contribute equally")
    print()
    print("Before scaling:")
    print(f"  Feature1: mean={X_train['Feature1'].mean():.2f}, std={X_train['Feature1'].std():.2f}")
    print(f"  Feature2: mean={X_train['Feature2'].mean():.2f}, std={X_train['Feature2'].std():.2f}")
    print()
    print("After scaling:")
    print(f"  Feature1: mean={X_train_scaled[:, 0].mean():.2f}, std={X_train_scaled[:, 0].std():.2f}")
    print(f"  Feature2: mean={X_train_scaled[:, 1].mean():.2f}, std={X_train_scaled[:, 1].std():.2f}")
    print()

    # 4. Explore different k values
    k_values = [1, 3, 5, 7, 10]
    results = {}

    print("=" * 80)
    print("PERFORMANCE ACROSS DIFFERENT k VALUES")
    print("=" * 80)
    print()

    for k in k_values:
        print("─" * 80)
        print(f"k = {k}")
        print("─" * 80)

        knn, (r2_train, mse_train), (r2_test, mse_test, mae_test), y_pred = train_knn(
            X_train_scaled, X_test_scaled, y_train, y_test, k
        )
        results[k] = (r2_train, mse_train, r2_test, mse_test, mae_test)

        print(f"TRAIN performance:")
        print(f"  R²:  {r2_train:.4f}")
        print(f"  MSE: {mse_train:.4f}")
        print()
        print(f"TEST performance:")
        print(f"  R²:  {r2_test:.4f}")
        print(f"  MSE: {mse_test:.4f}")
        print(f"  MAE: {mae_test:.4f}")
        print()

        # Overfitting/underfitting analysis
        gap = r2_train - r2_test
        if gap > 0.15:
            print(f"⚠️  OVERFITTING: k={k} is too small (R² gap = {gap:.4f})")
            print("    • Model memorizes close neighbors")
            print("    • High train performance, poor generalization")
        elif gap < 0.05:
            print(f"✓ Good balance: k={k} generalizes well (R² gap = {gap:.4f})")
        else:
            print(f"⚠️  Slight underfitting: k={k} may be too large (R² gap = {gap:.4f})")
            print("    • Averaging too many neighbors")
            print("    • Predictions become overly smoothed")
        print()

    # 5. Summary comparison
    print("=" * 80)
    print("SUMMARY: IMPACT OF k ON PERFORMANCE")
    print("=" * 80)
    print(f"{'k':<5} {'Train R²':<12} {'Test R²':<12} {'Test MSE':<12} {'Overfitting Gap':<20}")
    print("─" * 80)
    for k in k_values:
        r2_train, mse_train, r2_test, mse_test, mae_test = results[k]
        gap = r2_train - r2_test
        print(f"{k:<5} {r2_train:<12.4f} {r2_test:<12.4f} {mse_test:<12.4f} {gap:<20.4f}")
    print()

    # 6. Best k
    best_k = min(results.keys(), key=lambda k: results[k][3])  # Lowest test MSE
    print("=" * 80)
    print(f"BEST k VALUE: {best_k}")
    print("=" * 80)
    r2_train_best, mse_train_best, r2_test_best, mse_test_best, mae_test_best = results[best_k]
    print(f"Train R²: {r2_train_best:.4f}, Test R²: {r2_test_best:.4f}")
    print(f"Test MSE: {mse_test_best:.4f}, Test MAE: {mae_test_best:.4f}")
    print()

    # 7. Key insights
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("✓ How kNN Regression Works:")
    print("  • For each test point, find k nearest training points")
    print("  • Predict as the MEAN (average) of their target values")
    print("  • Simple, intuitive, but can be affected by local outliers")
    print()
    print("✓ Impact of k:")
    print("  • k=1: Extremely sensitive to individual points → OVERFITTING")
    print("  • k=3-5: Good balance (typical sweet spot)")
    print("  • k=7+: Predictions become overly smoothed → UNDERFITTING")
    print()
    print("✓ Feature Scaling Importance:")
    print("  • Distance-based: Unscaled features with large ranges dominate")
    print("  • Scaling ensures equal feature importance")
    print("  • Always apply StandardScaler or MinMaxScaler before kNN")
    print()
    print("✓ Computational Complexity:")
    print("  • kNN is 'lazy': no training phase, stores all data")
    print("  • Prediction is O(n*d) for each test point (expensive with large datasets)")
    print()


if __name__ == "__main__":
    main()
