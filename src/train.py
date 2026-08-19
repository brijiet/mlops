from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import load_data
from src.feature_engineering import create_features
from src.evaluate import evaluate_model
from src.config import load_config
from src.logger import get_logger
from src.model_utils import save_model
from src.model import create_model


logger = get_logger(__name__)


def main():

    logger.info("Starting ML training pipeline")

    config = load_config()

    logger.info("Loading dataset")

    X, y = load_data()

    logger.info(
        f"Dataset shape: {X.shape}"
    )

    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info("Creating features")

    X_train, X_test, scaler = create_features(
        X_train,
        X_test
    )

    n_estimators = config["model"]["n_estimators"]

    logger.info(
        f"Training Random Forest with "
        f"{n_estimators} trees"
    )

    model = create_model(
        n_estimators=n_estimators,
        random_state=random_state
)

    model.fit(X_train, y_train)

    logger.info("Model training completed")

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    logger.info("Model evaluation completed")

    for name, value in metrics.items():

        logger.info(
            f"{name}: {value:.4f}"
        )

    logger.info("Saving model")

    save_model(
        model,
        scaler
    )

    logger.info("Model saved successfully")


if __name__ == "__main__":
    main()