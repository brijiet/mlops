import mlflow

from mlflow import MlflowClient


TRACKING_URI = "http://127.0.0.1:5000"

MODEL_NAME = "CustomerChurnModel"


def promote_to_candidate(version):

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="candidate",
        version=str(version)
    )

    print(
        f"Version {version} "
        f"promoted to candidate"
    )


def promote_to_champion(version):

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=str(version)
    )

    print(
        f"Version {version} "
        f"promoted to champion"
    )


if __name__ == "__main__":

    version = 4

    promote_to_candidate(version)

    print(
        "Candidate model is ready "
        "for validation."
    )