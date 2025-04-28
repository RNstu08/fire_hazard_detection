from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
import time
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)

# Prometheus metrics
metrics = PrometheusMetrics(app)

# Load trained model and scaler
model = joblib.load("./models/best_model_xgboost.joblib")
scaler = joblib.load("./models/scaler.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()

        # Read incoming JSON
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty request"}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # ✨ Scale input features
        input_scaled = scaler.transform(input_df)

        # Predict probabilities
        fire_prob = model.predict_proba(input_scaled)[0][1]
        fire_label = int(fire_prob >= 0.5)

        elapsed_time = round(time.time() - start_time, 3)

        return jsonify({
            "fire_label": fire_label,
            "fire_probability": round(float(fire_prob), 4),
            "inference_time_sec": elapsed_time
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🔥 Fire Hazard Detection API is running!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
