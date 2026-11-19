from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

def test_invalid_input():

    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422