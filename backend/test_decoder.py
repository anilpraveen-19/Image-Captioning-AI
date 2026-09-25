import torch

from app.models.decoder import LSTMDecoder


# Model settings
feature_size = 256
embed_size = 256
hidden_size = 512
vocab_size = 28


# Create decoder
decoder = LSTMDecoder(
    feature_size=feature_size,
    embed_size=embed_size,
    hidden_size=hidden_size,
    vocab_size=vocab_size
)


# Simulated CNN features
features = torch.randn(
    2,
    256
)


# Two captions
captions = torch.tensor([
    [1, 4, 22, 7, 23, 24, 10, 25, 26, 27, 2],
    [1, 4, 12, 7, 8, 9, 10, 13, 2, 0, 0]
])


# Forward pass
outputs = decoder(
    features,
    captions
)


print("Features shape:", features.shape)
print("Captions shape:", captions.shape)
print("Output shape:", outputs.shape)