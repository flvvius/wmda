"""Exercise 5 (10 minutes): Hyperparameter Tuning with GridSearchCV.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def create_data(n_samples=50):
    """Create synthetic dataset for regression."""
    np.random.seed(42)
    X = np.random.rand(n_samples, 2) * 10
    # True relationship: y = 3*x0 + 2*x1 + noise
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 5, size=n_samples)

    df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
    df["Target"] = y
    return df


def gridsearch_decision_tree(X_train, X_test, y_train, y_test):
    """Perform GridSearchCV for DecisionTreeRegressor."""
    print("=" * 75)
    print("GRIDSEARCH: DECISION TREE REGRESSOR")
    print("=" * 75)
    print("Hyperparameter Space:")
    print("  max_depth: [1, 2, 3, 5, 10]")
    print("  min_samples_leaf: [1, 2, 5]")
    print()

    param_grid = {
        'max_depth': [1, 2, 3, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    model = DecisionTreeRegressor(random_state=42)

    # GridSearchCV: exhaustive search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',  # Negative MSE for maximization convention
        n_jobs=-1,  # Use all cores
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    # 7. Evaluate best model on test set
    best_model = grid_search.best_estimator_
    y_pred_test = best_model.predict(X_test)

    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Also compute train MSE to check for overfitting
    y_pred_train = best_model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)

    return grid_search, best_model, (r2_test, mse_test, mae_test), (mse_train, mse_test)


def gridsearch_ridge(X_train, X_test, y_train, y_test):
    """Perform GridSearchCV for Ridge regression."""
    print("=" * 75)
    print("GRIDSEARCH: RIDGE REGRESSION")
    print("=" * 75)
    print("Hyperparameter Space:")
    print("  alpha: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]")
    print()

    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }

    model = Ridge()

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_test = best_model.predict(X_test)

    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    y_pred_train = best_model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)

    return grid_search, best_model, (r2_test, mse_test, mae_test), (mse_train, mse_test)


def main():
    # 1. Create data
    df = create_data(n_samples=50)
    X = df[["Feature1", "Feature2"]]
    y = df["Target"]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("Dataset: n={}, train={}, test={}".format(len(X), len(X_train), len(X_test)))
    print("True relationship: y = 3*Feature1 + 2*Feature2 + noise")
    print()

    # GridSearch for DecisionTreeRegressor
    print()
    gs_tree, best_tree, (r2_tree, mse_tree, mae_tree), (mse_train_tree, mse_test_tree) = gridsearch_decision_tree(
        X_train, X_test, y_train, y_test
    )

    print("─" * 75)
    print("BEST PARAMETERS (Decision Tree):")
    print(f"  max_depth: {gs_tree.best_params_['max_depth']}")
    print(f"  min_samples_leaf: {gs_tree.best_params_['min_samples_leaf']}")
    print(f"  Best CV Score (neg MSE): {gs_tree.best_score_:.4f} → MSE: {-gs_tree.best_score_:.4f}")
    print()
    print("TEST SET PERFORMANCE (Decision Tree):")
    print(f"  R²: {r2_tree:.4f}")
    print(f"  MSE: {mse_tree:.4f}")
    print(f"  MAE: {mae_tree:.4f}")
    print(f"  Train MSE: {mse_train_tree:.4f}, Test MSE: {mse_test_tree:.4f}")
    print(f"  Overfitting Gap: {mse_test_tree - mse_train_tree:.4f}")
    print()

    # GridSearch for Ridge
    print()
    gs_ridge, best_ridge, (r2_ridge, mse_ridge, mae_ridge), (mse_train_ridge, mse_test_ridge) = gridsearch_ridge(
        X_train, X_test, y_train, y_test
    )

    print("─" * 75)
    print("BEST PARAMETERS (Ridge):")
    print(f"  alpha: {gs_ridge.best_params_['alpha']}")
    print(f"  Best CV Score (neg MSE): {gs_ridge.best_score_:.4f} → MSE: {-gs_ridge.best_score_:.4f}")
    print()
    print("TEST SET PERFORMANCE (Ridge):")
    print(f"  R²: {r2_ridge:.4f}")
    print(f"  MSE: {mse_ridge:.4f}")
    print(f"  MAE: {mae_ridge:.4f}")
    print(f"  Train MSE: {mse_train_ridge:.4f}, Test MSE: {mse_test_ridge:.4f}")
    print(f"  Overfitting Gap: {mse_test_ridge - mse_train_ridge:.4f}")
    print()

    # 3. Comparison of all CV results (top 10 for each model)
    print("=" * 75)
    print("TOP 10 CONFIGURATIONS - DECISION TREE")
    print("=" * 75)
    results_df_tree = pd.DataFrame(gs_tree.cv_results_)
    results_df_tree = results_df_tree.sort_values('rank_test_score').head(10)
    print(results_df_tree[['param_max_depth', 'param_min_samples_leaf', 'mean_test_score', 'std_test_score']].to_string(index=False))
    print()

    print("=" * 75)
    print("TOP 10 CONFIGURATIONS - RIDGE")
    print("=" * 75)
    results_df_ridge = pd.DataFrame(gs_ridge.cv_results_)
    results_df_ridge = results_df_ridge.sort_values('rank_test_score').head(10)
    print(results_df_ridge[['param_alpha', 'mean_test_score', 'std_test_score']].to_string(index=False))
    print()

    # 4. Final summary
    print("=" * 75)
    print("FINAL COMPARISON: BEST MODELS")
    print("=" * 75)
    print(f"{'Model':<30} {'Best Params':<40} {'Test R²':<10} {'Test MSE':<10}")
    print("─" * 75)
    print(f"{'DecisionTree':<30} {str(gs_tree.best_params_):<40} {r2_tree:<10.4f} {mse_tree:<10.4f}")
    print(f"{'Ridge':<30} {str(gs_ridge.best_params_):<40} {r2_ridge:<10.4f} {mse_ridge:<10.4f}")
    print()

    # 5. Key insights
    print("=" * 75)
    print("KEY INSIGHTS")
    print("=" * 75)
    print("✓ GridSearchCV Benefits:")
    print("  - Systematically explores the hyperparameter space")
    print("  - Uses cross-validation to reduce overfitting risk")
    print("  - Automatically finds the best configuration")
    print("  - Prevents manual (and often inefficient) trial-and-error")
    print()
    print("✓ Impact of Hyperparameter Tuning:")
    print(f"  - Decision Tree: Optimal max_depth = {gs_tree.best_params_['max_depth']}")
    print(f"  - Ridge: Optimal alpha = {gs_ridge.best_params_['alpha']}")
    print(f"  - Both reduce overfitting and improve generalization")
    print()
    print("✓ Cross-Validation:")
    print("  - 5-fold CV averages performance across multiple splits")
    print("  - More reliable estimate than single train-test")
    print("  - Reduces variance in hyperparameter selection")
    print()


if __name__ == "__main__":
    main()
