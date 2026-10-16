import pytest
from pydantic import ValidationError

from app.schemas import PredictRequest


@pytest.fixture
def valid_payload():
    return {
        "node_features": [[0.0, 0.0]],
        "mesh_edge_index": [[0], [0]],
        "mesh_edge_features": [[0.0]],
        "world_pos": [[0.0, 0.0, 0.0]],
    }


def test_default_radius(valid_payload):
    request = PredictRequest(**valid_payload)

    assert request.world_edge_radius == 0.03


def test_negative_radius(valid_payload):
    valid_payload["world_edge_radius"] = -0.01

    with pytest.raises(ValidationError):
        PredictRequest(**valid_payload)