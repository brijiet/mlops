from sklearn.model_selection import train_test_split

from src.data_processing import load_data
from src.feature_engineering import create_features


def test_feature_engineering():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = create_features(
        X_train,
        X_test
    )

    assert X_train_scaled.shape[1] == 30
    assert X_test_scaled.shape[1] == 30
    assert scaler is not None