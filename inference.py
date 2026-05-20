"""
Example inference code for the saved Iris classification model.

Run:
    python inference.py
"""

import joblib
import pandas as pd

MODEL_PATH = "models/best_iris_model.joblib"

model = joblib.load(MODEL_PATH)

# Feature order must match training:
# sepal_length, sepal_width, petal_length, petal_width
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

prediction = model.predict(sample)[0]
print("Input sample:")
print(sample)
print("\nPredicted iris species:", prediction)
