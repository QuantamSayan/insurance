# the loaded model + prediction logic
import joblib
import pandas as pd
import sys

original_sys_path = sys.path.copy()
sys.path.append("..")
model = joblib.load("../models/linear_model.pkl")
MODEL_VERSION = "1.0.0"
sys.path=original_sys_path

def predict_charges(input_data: dict) -> float:
    # column names must match exactly what the pipeline was fit on
    df = pd.DataFrame([input_data])
    output = float(model.predict(df)[0])
    return output