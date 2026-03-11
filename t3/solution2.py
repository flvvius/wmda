"""Exercise 2 (10 minutes): Polynomial Regression."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def create_nonlinear_data(n_samples=30):
    """Create synthetic non-linear dataset: y = 2*x² - 3*x + noise."""
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    # True relationship: quadratic
    y_true = (2 * (X**2) - 3 * X).flatten()
    noise = np.random.normal(0, 3, size=n_samples)
    y = y_true + noise
    return X, y


def train_polynomial_model(X_train, X_test, y_train, y_test, degree):
    """Train a polynomial regression model of given degree."""
    # Transform features to polynomial
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train linear model on polynomial features
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predict
    y_pred = model.predict(X_test_poly)

    # Evaluate
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, poly, (r2, mse, mae), y_pred


def train_linear_baseline(X_train, X_test, y_train, y_test):
    """Train a baseline linear (degree=1) model for comparison."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, (r2, mse, mae), y_pred


def main():
    # 1. Create synthetic non-linear data
    X, y = create_nonlinear_data(n_samples=30)

    # 2. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=" * 60)
    print("POLYNOMIAL REGRESSION COMPARISON")
    print("=" * 60)
    print(f"Dataset: n={len(X)}, train={len(X_train)}, test={len(X_test)}")
    print()

    # 3. Baseline: Linear model (degree=1)
    print("─" * 60)
    print("BASELINE: Linear Regression (degree=1)")
    print("─" * 60)
    linear_model, linear_metrics, linear_pred = train_linear_baseline(
        X_train, X_test, y_train, y_test
    )
    r2_lin, mse_lin, mae_lin = linear_metrics
    print(f"R²:   {r2_lin:.4f}")
    print(f"MSE:  {mse_lin:.4f}")
    print(f"MAE:  {mae_lin:.4f}")
    print()

    # 4. Polynomial degree=2
    print("─" * 60)
    print("POLYNOMIAL REGRESSION (degree=2)")
    print("─" * 60)
    model_p2, poly_p2, metrics_p2, pred_p2 = train_polynomial_model(
        X_train, X_test, y_train, y_test, degree=2
    )
    r2_p2, mse_p2, mae_p2 = metrics_p2
    print(f"R²:   {r2_p2:.4f}")
    print(f"MSE:  {mse_p2:.4f}")
    print(f"MAE:  {mae_p2:.4f}")
    print(f"Improvement vs. linear: R² +{r2_p2 - r2_lin:.4f}, MSE {mse_p2 - mse_lin:.4f}")
    print()

    # 5. Polynomial degree=3
    print("─" * 60)
    print("POLYNOMIAL REGRESSION (degree=3)")
    print("─" * 60)
    model_p3, poly_p3, metrics_p3, pred_p3 = train_polynomial_model(
        X_train, X_test, y_train, y_test, degree=3
    )
    r2_p3, mse_p3, mae_p3 = metrics_p3
    print(f"R²:   {r2_p3:.4f}")
    print(f"MSE:  {mse_p3:.4f}")
    print(f"MAE:  {mae_p3:.4f}")
    print(f"Improvement vs. linear: R² +{r2_p3 - r2_lin:.4f}, MSE {mse_p3 - mse_lin:.4f}")
    print()

    # 6. Summary and discussion
    print("=" * 60)
    print("SUMMARY & DISCUSSION")
    print("=" * 60)
    print(f"Linear (degree=1):     R²={r2_lin:.4f}")
    print(f"Polynomial (degree=2): R²={r2_p2:.4f}  → {'✓ Better' if r2_p2 > r2_lin else '✗ Worse'}")
    print(f"Polynomial (degree=3): R²={r2_p3:.4f}  → {'✓ Better' if r2_p3 > r2_lin else '✗ Worse'}")


if __name__ == "__main__":
    main()
