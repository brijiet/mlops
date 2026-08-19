from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import load_data
from src.feature_engineering import create_features
from sklearn.metrics import (
    accuracy_score,
)
from src.model_utils import load_model
from src.model import create_model


def test_model_accuracy():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_test, scaler = create_features(
        X_train,
        X_test
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    assert accuracy >= 0.80

def test_saved_model():

    model, scaler = load_model()

    assert model is not None
    assert scaler is not None

def test_create_model():

    model = create_model(
        n_estimators=10,
        random_state=42
    )

    assert model is not None
    assert model.n_estimators == 10
    assert model.random_state == 42