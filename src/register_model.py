import mlflow
from mlflow import MlflowClient


MODEL_NAME = "CustomerChurnModel"

client = MlflowClient(
    tracking_uri="http://127.0.0.1:5000"
)


def set_champion(version):

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=str(version)
    )

    print(
        f"Version {version} is now the champion"
    )


def get_latest_version():

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        raise RuntimeError(
            f"No versions found for {MODEL_NAME}"
        )

    latest = max(
        versions,
        key=lambda v: int(v.version)
    )

    return latest.version


if __name__ == "__main__":

    latest_version = get_latest_version()

    print(
        f"Latest model version: {latest_version}"
    )

    set_champion(latest_version)