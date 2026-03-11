"""Exercise 4 (10 minutes): Ridge vs. Lasso Regularization.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def create_data(n_samples=30):
    """Create synthetic dataset with correlated and irrelevant features."""
    np.random.seed(42)

    # Features:
    # X1 is the main driver
    # X2 is highly correlated with X1 (multicollinearity)
    # X3 is mostly irrelevant
    X1 = np.random.rand(n_samples) * 10
    X2 = X1 + np.random.rand(n_samples) * 2  # Correlated with X1
    X3 = np.random.rand(n_samples) * 10      # Mostly irrelevant

    # True relationship: y = 3*X1 + 1.5*X2 + noise
    # X3 should ideally be ignored
    y = 3 * X1 + 1.5 * X2 + np.random.normal(0, 5, size=n_samples)

    df = pd.DataFrame({
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "Target": y
    })
    return df


def train_model(X_train, X_test, y_train, y_test, model_type, alpha=1.0):
    """Train a regression model and evaluate on test set."""
    if model_type == "LinearRegression":
        model = LinearRegression()
    elif model_type == "Ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "Lasso":
        model = Lasso(alpha=alpha, max_iter=10000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, (r2, mse, mae), y_pred


def main():
    # 1. Create dataset
    df = create_data(n_samples=30)
    X = df[["X1", "X2", "X3"]]
    y = df["Target"]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("=" * 75)
    print("RIDGE VS. LASSO: REGULARIZATION & FEATURE SELECTION")
    print("=" * 75)
    print(f"Dataset: n={len(X)}, train={len(X_train)}, test={len(X_test)}")
    print(f"True relationship: y = 3*X1 + 1.5*X2 + noise (X3 is irrelevant)")
    print()

    # 3. Train baseline LinearRegression (no regularization)
    print("─" * 75)
    print("BASELINE: LinearRegression (no regularization)")
    print("─" * 75)
    lr, (r2_lr, mse_lr, mae_lr), _ = train_model(
        X_train, X_test, y_train, y_test, "LinearRegression"
    )
    print("Coefficients:")
    for i, (name, coef) in enumerate(zip(["X1", "X2", "X3"], lr.coef_), 1):
        print(f"  {name}: {coef:8.4f}")
    print(f"Intercept: {lr.intercept_:.4f}")
    print(f"Test R²: {r2_lr:.4f}, MSE: {mse_lr:.4f}, MAE: {mae_lr:.4f}")
    print()

    # 4. Train Ridge (L2 regularization)
    print("─" * 75)
    print("RIDGE REGRESSION (L2 Regularization, alpha=1.0)")
    print("─" * 75)
    ridge, (r2_ridge, mse_ridge, mae_ridge), _ = train_model(
        X_train, X_test, y_train, y_test, "Ridge", alpha=1.0
    )
    print("Coefficients:")
    for i, (name, coef) in enumerate(zip(["X1", "X2", "X3"], ridge.coef_), 1):
        shrinkage = abs(lr.coef_[i-1] - coef)
        print(f"  {name}: {coef:8.4f} (shrinkage: {shrinkage:.4f})")
    print(f"Intercept: {ridge.intercept_:.4f}")
    print(f"Test R²: {r2_ridge:.4f}, MSE: {mse_ridge:.4f}, MAE: {mae_ridge:.4f}")
    print()

    # 5. Train Lasso (L1 regularization)
    print("─" * 75)
    print("LASSO REGRESSION (L1 Regularization, alpha=1.0)")
    print("─" * 75)
    lasso, (r2_lasso, mse_lasso, mae_lasso), _ = train_model(
        X_train, X_test, y_train, y_test, "Lasso", alpha=1.0
    )
    print("Coefficients:")
    for i, (name, coef) in enumerate(zip(["X1", "X2", "X3"], lasso.coef_), 1):
        status = "→ ZERO (feature eliminated!)" if coef == 0 else ""
        print(f"  {name}: {coef:8.4f}  {status}")
    print(f"Intercept: {lasso.intercept_:.4f}")
    print(f"Test R²: {r2_lasso:.4f}, MSE: {mse_lasso:.4f}, MAE: {mae_lasso:.4f}")
    print()

    # 6. Performance comparison
    print("=" * 75)
    print("PERFORMANCE COMPARISON ON TEST SET")
    print("=" * 75)
    models_data = [
        ("LinearRegression", r2_lr, mse_lr, mae_lr),
        ("Ridge (α=1.0)", r2_ridge, mse_ridge, mae_ridge),
        ("Lasso (α=1.0)", r2_lasso, mse_lasso, mae_lasso),
    ]
    print(f"{'Model':<20} {'R²':<10} {'MSE':<12} {'MAE':<10}")
    print("─" * 75)
    for name, r2, mse, mae in models_data:
        print(f"{name:<20} {r2:<10.4f} {mse:<12.4f} {mae:<10.4f}")
    print()

    # 7. Effect of different alpha values on Lasso
    print("─" * 75)
    print("LASSO: EFFECT OF ALPHA ON FEATURE SELECTION")
    print("─" * 75)
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    print(f"{'Alpha':<8} {'X1':<10} {'X2':<10} {'X3':<10} {'Zeros':<8} {'R²':<10}")
    print("─" * 75)
    for alpha in alphas:
        lasso_temp, (r2_temp, _, _), _ = train_model(
            X_train, X_test, y_train, y_test, "Lasso", alpha=alpha
        )
        n_zeros = np.sum(lasso_temp.coef_ == 0)
        coef_str = f"{lasso_temp.coef_[0]:.3f}|{lasso_temp.coef_[1]:.3f}|{lasso_temp.coef_[2]:.3f}"
        print(f"{alpha:<8.1f} {lasso_temp.coef_[0]:<10.4f} {lasso_temp.coef_[1]:<10.4f} {lasso_temp.coef_[2]:<10.4f} {n_zeros:<8} {r2_temp:<10.4f}")


if __name__ == "__main__":
    main()
