import mlflow

from mlflow import MlflowClient
from src.promote_model import (
    promote_to_champion
)


TRACKING_URI = "http://127.0.0.1:5000"

MODEL_NAME = "CustomerChurnModel"


def get_candidate_version():

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    client = MlflowClient()

    model_version = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            "candidate"
        )
    )

    return model_version


def validate_candidate():

    candidate = get_candidate_version()

    print(
        "Candidate version:",
        candidate.version
    )

    return True


if __name__ == "__main__":

    candidate = get_candidate_version()

    print(
        "Candidate version:",
        candidate.version
    )

    MIN_F1 = 0.80
    run = mlflow.get_run(
    candidate.run_id
    )
    f1 = run.data.metrics.get(
    "f1"
    )
    if f1 is None:

        raise ValueError(
            "Candidate has no F1 metric"
        )
    if f1 >= MIN_F1:

        promote_to_champion(
            candidate.version
        )

        print(
            f"Promotion successful. "
            f"F1={f1:.4f}"
        )

else:

        print(
            f"Promotion rejected. "
            f"F1={f1:.4f}"
        )

