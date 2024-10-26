from flask import Flask, render_template, jsonify
import joblib
import numpy as np
import psutil
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import json
import threading
import time

app = Flask(__name__)

# Load pre-trained models
isolation_forest = joblib.load("models/isolation_forest.pkl")
polynomial_regression = joblib.load("models/polynomial_regression.pkl")

# Function to simulate memory usage data
def get_memory_usage():
    return psutil.virtual_memory().percent

# Background task to monitor memory and detect anomalies
def monitor_memory():
    while True:
        memory_data = get_memory_usage()
        memory_data = np.array(memory_data).reshape(-1, 1)

        # Anomaly detection with Isolation Forest
        anomaly_score = isolation_forest.decision_function(memory_data)
        is_anomaly = isolation_forest.predict(memory_data) == -1

        # Prediction with Polynomial Regression
        poly = PolynomialFeatures(degree=2)
        memory_poly = poly.fit_transform(memory_data)
        prediction = polynomial_regression.predict(memory_poly)

        # If an anomaly is detected, log and send alert (can be integrated with email/SMS)
        if is_anomaly[0]:
            print(f"Anomaly detected at {time.ctime()}: Memory usage {memory_data[0][0]}%")

        time.sleep(10)  # Adjust frequency as needed

# Endpoint to provide real-time memory data to the frontend
@app.route("/memory_data")
def memory_data():
    memory_usage = get_memory_usage()
    return jsonify({"memory_usage": memory_usage})

# Main dashboard
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    # Start memory monitoring in the background
    threading.Thread(target=monitor_memory, daemon=True).start()
    app.run(debug=True, host="0.0.0.0")
