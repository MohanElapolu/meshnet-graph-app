import numpy as np
import requests
import torch


def test_predict():
    graph = torch.load(
        "tests/sample_graph.pt",
        weights_only=False,
    )

    payload = {
        "node_features": graph.x.tolist(),
        "mesh_edge_index": graph.edge_index.tolist(),
        "mesh_edge_features": graph.edge_attr.tolist(),
        "world_pos": graph.world_pos.tolist(),
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload,
        timeout=60,
    )

    assert response.status_code == 200

    result = response.json()
    velocity = np.asarray(result["velocity"])

    assert velocity.shape == (graph.x.shape[0], 3)
    assert np.isfinite(velocity).all()