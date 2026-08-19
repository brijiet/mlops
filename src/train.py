from pathlib import Path
import os
import time

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from src.data_processing import load_data
from src.feature_engineering import create_features
from src.evaluate import evaluate_model
from src.config import load_config
from src.logger import get_logger
from src.model_utils import save_model
from src.model import create_model


# --------------------------------
# Create models directory
# --------------------------------

Path("models").mkdir(parents=True, exist_ok=True)

logger = get_logger(__name__)


def main():

    logger.info("Starting ML training pipeline")

    # --------------------------------
    # MLflow configuration
    # --------------------------------

    mlflow_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )

    mlflow.set_tracking_uri(mlflow_uri)

    mlflow.set_experiment(
        "customer-churn"
    )

    # --------------------------------
    # Start MLflow run
    # --------------------------------

    with mlflow.start_run():

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

        # Log dataset information

        mlflow.log_param(
            "dataset_rows",
            X.shape[0]
        )

        mlflow.log_param(
            "dataset_features",
            X.shape[1]
        )

        # --------------------------------
        # Configuration parameters
        # --------------------------------

        test_size = config["data"]["test_size"]

        random_state = config["data"]["random_state"]

        mlflow.log_param(
            "test_size",
            test_size
        )

        mlflow.log_param(
            "random_state",
            random_state
        )

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

        logger.info(
            "Creating features"
        )

        X_train, X_test, scaler = create_features(
            X_train,
            X_test
        )

        # --------------------------------
        # Create model
        # --------------------------------

        n_estimators = config["model"]["n_estimators"]

        logger.info(
            f"Training Random Forest with "
            f"{n_estimators} trees"
        )

        model = create_model(
            n_estimators=n_estimators,
            random_state=random_state
        )

        mlflow.log_param(
            "n_estimators",
            n_estimators
        )

        # --------------------------------
        # Training
        # --------------------------------

        start_time = time.time()

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

        training_time = time.time() - start_time

        logger.info(
            "Model training completed"
        )

        # Log training time

        mlflow.log_metric(
            "training_time_seconds",
            training_time
        )

        # --------------------------------
        # Evaluation
        # --------------------------------

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        logger.info(
            "Model evaluation completed"
        )

        # --------------------------------
        # Log metrics
        # --------------------------------

        for name, value in metrics.items():

            logger.info(
                f"{name}: {value:.4f}"
            )

            mlflow.log_metric(
                name,
                value
            )

        # --------------------------------
        # Save model locally
        # --------------------------------

        save_model(
            model,
            scaler
        )

        logger.info(
            "Model saved successfully"
        )

        # --------------------------------
        # Log model to MLflow
        # --------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="CustomerChurnModel"
        )

        logger.info(
            "Model logged to MLflow"
        )


if __name__ == "__main__":
    main()