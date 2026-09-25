import torch
from app.models.encoder import CNNEncoder


model = CNNEncoder(embed_size=256)

dummy_image = torch.randn(1, 3, 224, 224)

output = model(dummy_image)

print("Input shape:", dummy_image.shape)
print("Output shape:", output.shape)