from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.model_service import load_model


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)


try:

    model = load_model()

    model_status = "loaded"

except Exception as error:

    model = None

    model_status = "failed"

    print(
        f"Model loading failed: {error}"
    )


class PredictionRequest(BaseModel):

    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30
    )
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model: str
    alias: str


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "CustomerChurnModel",
        "alias": "champion",
        "model_status": model_status
    }


@app.post("/predict",response_model=PredictionResponse)
def predict(
    request: PredictionRequest
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="ML model is not available"
        )
    def get_prediction_label(prediction):
        if prediction == 0:
            return "malignant"

        return "benign"

    try:

        prediction = model.predict(
            [request.features]
        )

        probability = model.predict_proba(
            [request.features]
        )

        return {
            "prediction": get_prediction_label(int(prediction[0])),
            "probability": float(
                probability[0].max()
            ),
            "model": "CustomerChurnModel",
            "alias": "champion"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )