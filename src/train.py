from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from contextlib import nullcontext
from src.data_processing import load_data
from src.feature_engineering import create_features
from src.evaluate import evaluate_model
from src.config import load_config
from src.logger import get_logger
from src.model_utils import save_model
from src.model import create_model


Path("models").mkdir(parents=True, exist_ok=True)

logger = get_logger(__name__)


def main():

    logger.info("Starting ML training pipeline")

    # --------------------------------
    # MLflow configuration
    # --------------------------------

    mlflow_uri = None

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("customer-churn")

    # --------------------------------
    # Load configuration
    # --------------------------------

    config = load_config()

    # --------------------------------
    # Load data
    # --------------------------------

    logger.info("Loading dataset")

    X, y = load_data()

    logger.info(
        f"Dataset shape: {X.shape}"
    )

    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    # --------------------------------
    # Train / Test split
    # --------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # --------------------------------
    # Feature engineering
    # --------------------------------

    logger.info("Creating features")

    X_train, X_test, scaler = create_features(
        X_train,
        X_test
    )

    # --------------------------------
    # Create model
    # --------------------------------

    n_estimators = config["model"]["n_estimators"]

    logger.info(
        f"Training Random Forest with {n_estimators} trees"
    )

    model = create_model(
        n_estimators=n_estimators,
        random_state=random_state
    )

    # --------------------------------
    # MLflow run
    # --------------------------------

    with mlflow.start_run() if mlflow_uri else nullcontext():

        if mlflow_uri:

            mlflow.log_param(
                "n_estimators",
                n_estimators
            )

            mlflow.log_param(
                "random_state",
                random_state
            )

            mlflow.log_param(
                "test_size",
                test_size
            )

        # --------------------------------
        # Train
        # --------------------------------

        model.fit(
            X_train,
            y_train
        )

        logger.info(
            "Model training completed"
        )

        # --------------------------------
        # Evaluate
        # --------------------------------

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        logger.info(
            "Model evaluation completed"
        )

        for name, value in metrics.items():

            logger.info(
                f"{name}: {value:.4f}"
            )

            if mlflow_uri:
                mlflow.log_metric(
                    name,
                    value
                )

        # --------------------------------
        # Save model
        # --------------------------------

        save_model(
            model,
            scaler
        )

        logger.info(
            "Model saved successfully"
        )

        # --------------------------------
        # MLflow model
        # --------------------------------

        if mlflow_uri:

            mlflow.sklearn.log_model(
                model,
                "model"
            )


if __name__ == "__main__":
    main()