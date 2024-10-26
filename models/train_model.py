import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic memory usage data
memory_data = np.random.normal(50, 10, 1000).reshape(-1, 1)  # Adjust as per actual data

# Isolation Forest for anomaly detection
isolation_forest = IsolationForest(contamination=0.1)  # Tune contamination based on expected anomaly rate
isolation_forest.fit(memory_data)

# Polynomial Regression for prediction
poly = PolynomialFeatures(degree=2)
memory_poly = poly.fit_transform(memory_data)
polynomial_regression = LinearRegression()
polynomial_regression.fit(memory_poly, memory_data)

# Save models
joblib.dump(isolation_forest, "models/isolation_forest.pkl")
joblib.dump(polynomial_regression, "models/polynomial_regression.pkl")
