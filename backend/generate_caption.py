import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from app.models.decoder import LSTMDecoder
from app.utils.vocabulary import Vocabulary


# -----------------------------
# Settings
# -----------------------------

device = torch.device("cpu")

image_path = "data/mini_flickr/images/dog.jpg"
checkpoint_path = "checkpoints/image_captioning_model.pth"


# -----------------------------
# Load checkpoint
# -----------------------------

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

word_to_idx = checkpoint["vocab_word_to_idx"]
idx_to_word = checkpoint["vocab_idx_to_word"]

vocab_size = len(word_to_idx)


# -----------------------------
# Load decoder
# -----------------------------

decoder = LSTMDecoder(
    feature_size=2048,
    embed_size=256,
    hidden_size=512,
    vocab_size=vocab_size
)

decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)

decoder.to(device)
decoder.eval()


# -----------------------------
# Load ResNet50
# -----------------------------

weights = models.ResNet50_Weights.DEFAULT

encoder = models.resnet50(
    weights=weights
)

encoder.fc = nn.Identity()

encoder.to(device)
encoder.eval()


# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


image = Image.open(
    image_path
).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)


# -----------------------------
# Extract image features
# -----------------------------

with torch.no_grad():

    features = encoder(image)


# -----------------------------
# Generate caption
# -----------------------------

start_token = word_to_idx["<START>"]
end_token = word_to_idx["<END>"]

current_token = start_token

caption_words = []


# Initial hidden/cell states

with torch.no_grad():

    h = decoder.init_h(features)

    c = decoder.init_c(features)

    h = h.view(
        1,
        decoder.num_layers,
        decoder.hidden_size
    )

    h = h.permute(1, 0, 2).contiguous()

    c = c.view(
        1,
        decoder.num_layers,
        decoder.hidden_size
    )

    c = c.permute(1, 0, 2).contiguous()


# Generate one word at a time

for _ in range(20):

    token = torch.tensor(
        [[current_token]],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        embedding = decoder.embedding(token)

        output, (h, c) = decoder.lstm(
            embedding,
            (h, c)
        )

        scores = decoder.linear(output)

        next_token = scores.argmax(
            dim=-1
        ).item()


    if next_token == end_token:
        break

    if next_token != start_token:

        word = idx_to_word[str(next_token)] \
            if str(next_token) in idx_to_word \
            else idx_to_word[next_token]

        if word not in ["<PAD>", "<START>", "<END>"]:
            caption_words.append(word)


    current_token = next_token


# -----------------------------
# Print result
# -----------------------------

caption = " ".join(caption_words)

print()
print("Image:", image_path)
print("Generated Caption:", caption)