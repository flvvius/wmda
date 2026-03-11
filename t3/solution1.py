import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_or_create_data():
	"""Try to load a CSV named 'housing.csv' in the cwd; otherwise create a synthetic dataset."""
	csv_path = "housing.csv"
	if os.path.exists(csv_path):
		df = pd.read_csv(csv_path)
		print(f"Loaded data from {csv_path} — shape: {df.shape}")
		return df

	# Create a small synthetic dataset with a categorical feature
	np.random.seed(42)
	n = 200
	sqft = np.random.normal(1500, 300, size=n).clip(300)
	bedrooms = np.random.choice([1, 2, 3, 4], size=n, p=[0.1, 0.4, 0.35, 0.15])
	# Categorical location with different baseline effects
	locations = np.random.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])

	# price = 150 * sqft + 10000 * bedrooms + location_effect + noise
	loc_effect = {"A": 20000, "B": 0, "C": -10000}
	price = 150 * sqft + 10000 * bedrooms + np.vectorize(loc_effect.get)(locations)
	price += np.random.normal(0, 20000, size=n)  # add noise

	df = pd.DataFrame({
		"sqft": sqft,
		"bedrooms": bedrooms,
		"location": locations,
		"price": price,
	})
	print(f"Created synthetic dataset — shape: {df.shape}")
	return df


def prepare_features(df):
	"""Prepare X and y. Encode categorical features using get_dummies."""
	df2 = df.copy()
	# Target
	y = df2["price"].values

	# Features: numeric + categorical
	X = df2[["sqft", "bedrooms", "location"]]
	# One-hot encode 'location' and drop one column to avoid collinearity
	X = pd.get_dummies(X, columns=["location"], drop_first=True)
	return X, y


def train_and_evaluate(X, y, test_size=0.2, random_state=42):
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state
	)

	model = LinearRegression()
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	r2 = r2_score(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)

	return model, X.columns.tolist(), (r2, mse, mae), (y_test, y_pred)


def main():
	df = load_or_create_data()
	X, y = prepare_features(df)

	model, feature_names, metrics_vals, (y_test, y_pred) = train_and_evaluate(X, y)
	r2, mse, mae = metrics_vals

	print("\nModel summary:")
	print("Intercept:", round(model.intercept_, 3))
	print("Coefficients:")
	for name, coef in zip(feature_names, model.coef_):
		print(f" - {name}: {coef:.3f}")

	print(f"\nEvaluation on test set (n={len(y_test)}):")
	print(f" R²: {r2:.3f}")
	print(f" MSE: {mse:.3f}")
	print(f" MAE: {mae:.3f}")


if __name__ == "__main__":
	main()
