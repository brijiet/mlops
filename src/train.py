from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from .data_processing import load_data
from .feature_engineering import create_features
from .evaluate import evaluate_model


def main():

    print("Loading data...")

    X, y = load_data()

    print("Dataset shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Creating features...")

    X_train, X_test, scaler = create_features(
        X_train,
        X_test
    )

    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nModel Metrics:")

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )


if __name__ == "__main__":
    main()