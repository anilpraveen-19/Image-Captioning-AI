import torch
import torch.nn as nn
from torchvision import models


class CNNEncoder(nn.Module):

    def __init__(self, embed_size=256):
        super().__init__()

        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove the final classification layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Freeze ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Convert ResNet's 2048 features
        # into the size required by our decoder
        self.fc = nn.Linear(2048, embed_size)

        self.relu = nn.ReLU()

    def forward(self, images):

        # Extract image features
        features = self.resnet(images)

        # Flatten:
        # [batch, 2048, 1, 1]
        #      ↓
        # [batch, 2048]
        features = features.view(features.size(0), -1)

        # Project features
        features = self.fc(features)

        features = self.relu(features)

        return features