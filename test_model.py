from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

checkpoint_path = "checkpoints/HybridMeshGraphNet.0.5.mdlus"

model = HybridMeshGraphNet.from_checkpoint(checkpoint_path)
model = model.to("cuda")
model.eval()

print("Checkpoint loaded successfully!")
print("Device:", next(model.parameters()).device)