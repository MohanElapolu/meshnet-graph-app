import requests
import torch

graph = torch.load(
    "tests/sample_graph.pt",
    weights_only=False,
    map_location="cpu",
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
    timeout=120,
)

print("Status:", response.status_code)
print("Response:", response.text[:500])