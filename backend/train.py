import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from app.models.decoder import LSTMDecoder
from app.services.dataset import ImageCaptionDataset, collate_fn
from app.utils.vocabulary import Vocabulary


# ============================================================
# 1. Device
# ============================================================

device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# 2. Paths
# ============================================================

image_dir = "data/mini_flickr/images"
captions_file = "data/mini_flickr/captions.txt"


# ============================================================
# 3. Build Vocabulary
# ============================================================

captions_list = []

with open(
    captions_file,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        image_name, caption = line.split("|", 1)

        captions_list.append(caption)


vocab = Vocabulary(min_freq=1)

vocab.build(captions_list)

print("Vocabulary size:", len(vocab))


# ============================================================
# 4. Dataset
# ============================================================

dataset = ImageCaptionDataset(
    image_dir=image_dir,
    captions_file=captions_file,
    vocabulary=vocab
)


print("Dataset size:", len(dataset))


# ============================================================
# 5. DataLoader
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)


# ============================================================
# 6. ResNet50 Encoder
# ============================================================

print("Loading ResNet50...")


weights = models.ResNet50_Weights.DEFAULT

resnet = models.resnet50(
    weights=weights
)


# Remove final classification layer
resnet.fc = nn.Identity()


# Freeze CNN
for parameter in resnet.parameters():
    parameter.requires_grad = False


resnet = resnet.to(device)

resnet.eval()


print("ResNet50 ready.")


# ============================================================
# 7. LSTM Decoder
# ============================================================

decoder = LSTMDecoder(
    feature_size=2048,
    embed_size=256,
    hidden_size=512,
    vocab_size=len(vocab)
)


decoder = decoder.to(device)


# ============================================================
# 8. Loss Function
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=vocab.word_to_idx["<PAD>"]
)


# ============================================================
# 9. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    decoder.parameters(),
    lr=0.001
)


# ============================================================
# 10. Training
# ============================================================

num_epochs = 5

print("\nStarting training...\n")


for epoch in range(num_epochs):

    decoder.train()

    total_loss = 0

    for images, captions in loader:

        images = images.to(device)
        captions = captions.to(device)


        # ----------------------------------------------------
        # CNN feature extraction
        # ----------------------------------------------------

        with torch.no_grad():

            features = resnet(images)


        # ----------------------------------------------------
        # Teacher forcing
        #
        # Input:
        # <START> a dog is ...
        #
        # Target:
        # a dog is ... <END>
        # ----------------------------------------------------

        inputs = captions[:, :-1]

        targets = captions[:, 1:]


        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        outputs = decoder(
            features,
            inputs
        )


        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            outputs.reshape(-1, len(vocab)),
            targets.reshape(-1)
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


    average_loss = total_loss / len(loader)


    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {average_loss:.4f}"
    )


# ============================================================
# 11. Save Model
# ============================================================

checkpoint = {
    "decoder_state_dict": decoder.state_dict(),
    "vocab_word_to_idx": vocab.word_to_idx,
    "vocab_idx_to_word": vocab.idx_to_word
}


torch.save(
    checkpoint,
    "checkpoints/image_captioning_model.pth"
)


print("\nTraining complete!")
print("Model saved to:")
print("checkpoints/image_captioning_model.pth")