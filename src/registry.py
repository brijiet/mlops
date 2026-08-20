import mlflow
import os


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "CustomerChurnModel"
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion"
)


def load_champion_model():

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    return mlflow.sklearn.load_model(
        model_uri
    )