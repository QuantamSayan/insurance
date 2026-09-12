# the loaded model + prediction logic
import joblib
import pandas as pd
from pathlib import Path
from transformers import (
    smokers_fun,
    smokers_feature_name,
    non_smokers_fun,
    non_smokers_feature_name,
)

model_path = Path("models/linear_model.pkl")
model = joblib.load(model_path)
MODEL_VERSION = "1.0.0"

def predict_charges(input_data: dict) -> float:
    # column names must match exactly what the pipeline was fit on
    df = pd.DataFrame([input_data])
    output = float(model.predict(df)[0])
    return output