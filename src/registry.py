import mlflow


TRACKING_URI = "http://127.0.0.1:5000"

MODEL_NAME = "CustomerChurnModel"


def load_champion_model():

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    model_uri = (
        f"models:/{MODEL_NAME}@champion"
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    return model