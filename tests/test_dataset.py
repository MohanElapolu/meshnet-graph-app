import torch

from deforming_plate_dataset import DeformingPlateDataset

dataset = DeformingPlateDataset(
    data_dir="data",
    split="test",
    num_samples=1,
    num_steps=2,
    noise_std=0.0,
)

graph, *_ = dataset[0]

print("Node features:", graph.x.shape)
print("Mesh edges:", graph.edge_index.shape)
print("Edge features:", graph.edge_attr.shape)
print("World positions:", graph.world_pos.shape)

torch.save(graph, "tests/sample_graph.pt")

print("Sample graph saved successfully!")