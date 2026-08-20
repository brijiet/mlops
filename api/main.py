from fastapi import FastAPI
from pydantic import BaseModel,Field

from src.registry import load_champion_model


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)


model = load_champion_model()


class PredictionRequest(BaseModel):

    features: list[
        float
    ] = Field(
        ...,
        min_length=30,
        max_length=30
    )


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "CustomerChurnModel",
        "alias": "champion"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    prediction = model.predict(
        [request.features]
    )

    probability = model.predict_proba(
        [request.features]
    )

    return {
        "prediction": int(
            prediction[0]
        ),
        "probability": float(
            probability[0].max()
        )
    }