from flask import Flask, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import psutil

app = Flask(__name__)

# Load models
isolation_forest = joblib.load("models/isolation_forest.pkl")
polynomial_regression = joblib.load("models/polynomial_regression.pkl")

# Function to get real-time system memory data
def get_memory_data():
    memory_info = psutil.virtual_memory()
    memory_data = np.array([[memory_info.percent]])  # Memory usage in percentage for model input
    return memory_data, memory_info.used / (1024 ** 2)  # Return percentage and actual used memory in MB

# Endpoint for dynamic visualization data
@app.route("/api/memory_data", methods=["GET"])
def memory_data():
    memory_data, memory_used_mb = get_memory_data()
    poly = PolynomialFeatures(degree=2)
    memory_poly = poly.fit_transform(memory_data)
    predictions = polynomial_regression.predict(memory_poly)
    anomalies = isolation_forest.predict(memory_data)

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "memory_usage_percent": memory_data.flatten().tolist()[0],
        "memory_usage_mb": memory_used_mb,
        "predictions": predictions.flatten().tolist()[0],
        "anomalies": anomalies.tolist()[0],
    }
    return jsonify(data)

# Main route to serve the HTML front-end
@app.route("/")
def index():
    return render_template("index.html")

# Scheduler setup for additional tasks if needed
scheduler = BackgroundScheduler()
scheduler.start()

if __name__ == "__main__":
    app.run(debug=True)
