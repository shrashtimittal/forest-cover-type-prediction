import joblib
import pandas as pd
import numpy as np

# Load the saved model
model = joblib.load("artifacts/xgb_best_model.joblib")

# Example: use one row from the training set
sample = pd.read_parquet("artifacts/X_test.parquet").iloc[[0]]  # first row of test set
prediction = model.predict(sample)[0] + 1  # add +1 to shift back to 1–7
print("Predicted Cover Type:", prediction)
