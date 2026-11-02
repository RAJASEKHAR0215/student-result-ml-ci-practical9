import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

def main():

    np.random.seed(42)

    rows = 2000

    data = pd.DataFrame({
        "attendance": np.random.randint(30, 101, rows),
        "internal_marks": np.random.randint(10, 101, rows),
        "assignment_marks": np.random.randint(10, 101, rows),
        "previous_score": np.random.randint(10, 101, rows)
    })

    score = (
        0.30 * data["attendance"]
        + 0.30 * data["internal_marks"]
        + 0.20 * data["assignment_marks"]
        + 0.20 * data["previous_score"]
    )

    data["result"] = (score >= 60).astype(int)

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(data[FEATURES], data["result"])

    joblib.dump(model, "student_result_model.pkl")

    data.to_csv("student_results.csv", index=False)

    print("Training completed.")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
