from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

def test_ready():

    response = client.get(
        "/ready"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"

def test_invalid_prediction():

    response = client.post(
        "/predict",
        json={
            "features": [1, 2, 3]
        }
    )

    assert response.status_code == 422