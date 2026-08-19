import joblib
from pathlib import Path


MODEL_DIR = Path("models")


def save_model(model, scaler):

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        model,
        MODEL_DIR / "model.pkl"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )


def load_model():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    return model, scaler