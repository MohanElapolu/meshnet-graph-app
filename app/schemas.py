from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    node_features: list[list[float]]
    mesh_edge_index: list[list[int]]
    mesh_edge_features: list[list[float]]
    world_pos: list[list[float]]
    world_edge_radius: float = Field(default=0.03, gt=0)