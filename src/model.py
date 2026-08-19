from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def create_model(
    n_estimators=100,
    random_state=42
):

    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            )
        )
    ])

    return pipeline