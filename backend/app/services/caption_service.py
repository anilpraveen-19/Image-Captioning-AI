from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from app.models.decoder import LSTMDecoder


class CaptionService:

    def __init__(self):
        self.device = torch.device("cpu")

        checkpoint_path = (
            Path(__file__).resolve().parents[2]
            / "checkpoints"
            / "image_captioning_model.pth"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        self.word_to_idx = checkpoint["vocab_word_to_idx"]
        self.idx_to_word = checkpoint["vocab_idx_to_word"]

        # JSON/checkpoints can store integer keys as strings
        self.idx_to_word = {
            int(k): v for k, v in self.idx_to_word.items()
        }

        self.decoder = LSTMDecoder(
            feature_size=2048,
            embed_size=256,
            hidden_size=512,
            vocab_size=len(self.word_to_idx)
        )

        self.decoder.load_state_dict(
            checkpoint["decoder_state_dict"]
        )

        self.decoder.to(self.device)
        self.decoder.eval()

        weights = models.ResNet50_Weights.DEFAULT

        self.encoder = models.resnet50(
            weights=weights
        )

        self.encoder.fc = nn.Identity()
        self.encoder.to(self.device)
        self.encoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def generate_caption(self, image_path):

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            features = self.encoder(image)

            h = self.decoder.init_h(features)
            c = self.decoder.init_c(features)

            h = h.view(
                1,
                self.decoder.num_layers,
                self.decoder.hidden_size
            )

            h = h.permute(1, 0, 2).contiguous()

            c = c.view(
                1,
                self.decoder.num_layers,
                self.decoder.hidden_size
            )

            c = c.permute(1, 0, 2).contiguous()

        current_token = self.word_to_idx["<START>"]
        end_token = self.word_to_idx["<END>"]

        words = []

        for _ in range(20):

            token = torch.tensor(
                [[current_token]],
                dtype=torch.long,
                device=self.device
            )

            with torch.no_grad():

                embedding = self.decoder.embedding(token)

                output, (h, c) = self.decoder.lstm(
                    embedding,
                    (h, c)
                )

                scores = self.decoder.linear(output)

                next_token = scores.argmax(
                    dim=-1
                ).item()

            if next_token == end_token:
                break

            word = self.idx_to_word.get(next_token)

            if word and word not in [
                "<PAD>",
                "<START>",
                "<END>"
            ]:
                words.append(word)

            current_token = next_token

        return " ".join(words)