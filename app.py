from pathlib import Path
import joblib
import pandas as pd

from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

@app.get("/")
@app.get("/health")
def health():

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.post("/predict")
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    try:

        sample = pd.DataFrame([{
            feature: float(data[feature])
            for feature in FEATURES
        }])

        if ((sample < 0) | (sample > 100)).any().any():
            return jsonify({
                "error": "Values must be between 0 and 100"
            }), 400

        if not MODEL_PATH.exists():
            return jsonify({
                "error": "Model file not found"
            }), 500

        model = joblib.load(MODEL_PATH)

        prediction_code = int(model.predict(sample)[0])

        prediction = (
            "PASS" if prediction_code == 1 else "FAIL"
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "Feature values must be numeric"
        }), 400


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
