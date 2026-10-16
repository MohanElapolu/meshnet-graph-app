import json
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from torch_geometric.data import Data

from app.schemas import PredictRequest
from helpers import add_world_edges

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = HybridMeshGraphNet.from_checkpoint(
        "checkpoints/HybridMeshGraphNet.0.5.mdlus"
    ).to("cuda")

    model.eval()
    print("Model loaded on GPU")

    yield

    model = None
    torch.cuda.empty_cache()


app = FastAPI(
    title="MeshNet Graph API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": "cuda",
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    device = next(model.parameters()).device

    graph = Data(
        x=torch.tensor(request.node_features, dtype=torch.float32, device=device),
        edge_index=torch.tensor(request.mesh_edge_index, dtype=torch.long, device=device),
        edge_attr=torch.tensor(request.mesh_edge_features, dtype=torch.float32, device=device),
        world_pos=torch.tensor(request.world_pos, dtype=torch.float32, device=device),
    )

    graph, mesh_edges, world_edges = add_world_edges(
        graph,
        world_edge_radius=request.world_edge_radius,
        edge_stats_path="stats/edge_stats.json",
    )

    with torch.inference_mode():
        output = model(graph.x, mesh_edges, world_edges, graph)

    with open("stats/node_stats.json", encoding="utf-8") as f:
        stats = json.load(f)

    velocity_mean = torch.as_tensor(stats["velocity_mean"], device=device)
    velocity_std = torch.as_tensor(stats["velocity_std"], device=device)

    velocity = output[:, :3] * velocity_std + velocity_mean
    next_world_pos = graph.world_pos[:, :3] + velocity

    return {
        "velocity": velocity.cpu().tolist(),
        "next_world_pos": next_world_pos.cpu().tolist(),
        "stress_normalized": output[:, 3].cpu().tolist(),
    }