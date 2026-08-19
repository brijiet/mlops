import numpy as np

from src.model_utils import load_model


def predict(input_data):

    model, scaler = load_model()

    input_array = np.array(input_data)

    input_array = input_array.reshape(1, -1)

    input_scaled = scaler.transform(
        input_array
    )

    prediction = model.predict(
        input_scaled
    )

    probability = model.predict_proba(
        input_scaled
    )

    return {
        "prediction": int(prediction[0]),
        "probability": float(
            probability[0].max()
        )
    }


if __name__ == "__main__":

    sample = [
        14.0, 20.0, 90.0, 600.0, 0.1,
        0.1, 0.1, 0.05, 0.2, 0.06,
        1.0, 1.0, 2.0, 10.0, 0.005,
        0.02, 0.02, 0.01, 0.03, 0.01,
        16.0, 25.0, 100.0, 700.0, 0.12,
        0.15, 0.2, 0.1, 0.25, 0.08
    ]

    result = predict(sample)

    print(result)