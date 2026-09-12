# Accepts the user input
# Predict the insurance cost

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models.predict import model, MODEL_VERSION

app = FastAPI()
@app.get("/")
def home():
    return {"message":"welcome to insurance predictor"}
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": MODEL_VERSION,
        "model_loaded" : model is not None
    }

from models.predict import predict_charges
from schema.io import UserInput, InsuranceOutput

@app.post("/predict", response_model=InsuranceOutput)
def predict(user_input: UserInput):
    user_input_dict = {
        "Age": user_input.age,
        "Sex": user_input.sex,
        "BMI": user_input.bmi,
        "Children": user_input.children,
        "Smoker": user_input.smoker,
        "Region": user_input.region
    }
    try:
        prediction = predict_charges(user_input_dict)
        return JSONResponse(status_code=200,content={"insurance amount": prediction})
    except Exception as e:
        return JSONResponse(status_code=500,content={"error": str(e)})


