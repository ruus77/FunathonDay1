from fastapi import FastAPI
import mlflow
import pandas as pd

from config import MLFLOW_TRACKING_URI

app = FastAPI()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# load production model from MLflow registry
model = mlflow.sklearn.load_model(
    "models:/housing_model/Production"
)


@app.post("/predict")
def predict(payload: dict):

    X = pd.DataFrame([payload])

    pred = model.predict(X)[0]

    return {"prediction": float(pred)}

