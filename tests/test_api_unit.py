import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    # Do not enter TestClient as a context manager:
    # this avoids running the GPU-loading lifespan.
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is False


def test_predict_without_model(client):
    payload = {
        "node_features": [[0.0, 0.0]],
        "mesh_edge_index": [[0], [0]],
        "mesh_edge_features": [[0.0]],
        "world_pos": [[0.0, 0.0, 0.0]],
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"


def test_invalid_radius(client):
    payload = {
        "node_features": [[0.0, 0.0]],
        "mesh_edge_index": [[0], [0]],
        "mesh_edge_features": [[0.0]],
        "world_pos": [[0.0, 0.0, 0.0]],
        "world_edge_radius": -0.01,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422