import pandas as pd

from src.registry import load_champion_model


def predict(input_data):

    model = load_champion_model()

    input_df = pd.DataFrame(
        [input_data]
    )

    prediction = model.predict(
        input_df
    )

    probability = model.predict_proba(
        input_df
    )

    return {
        "prediction": int(
            prediction[0]
        ),
        "probability": float(
            probability[0].max()
        )
    }