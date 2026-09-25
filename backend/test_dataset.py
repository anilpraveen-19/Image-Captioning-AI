from torch.utils.data import DataLoader

from app.services.dataset import (
    ImageCaptionDataset,
    collate_fn
)

from app.utils.vocabulary import Vocabulary


# --------------------------------------------------
# Paths
# --------------------------------------------------

image_dir = "data/mini_flickr/images"

captions_file = "data/mini_flickr/captions.txt"


# --------------------------------------------------
# Read captions
# --------------------------------------------------

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


# --------------------------------------------------
# Build vocabulary
# --------------------------------------------------

vocab = Vocabulary(min_freq=1)

vocab.build(captions_list)


print("Vocabulary size:", len(vocab))


# --------------------------------------------------
# Create Dataset
# --------------------------------------------------

dataset = ImageCaptionDataset(
    image_dir=image_dir,
    captions_file=captions_file,
    vocabulary=vocab
)


print("Dataset size:", len(dataset))


# --------------------------------------------------
# Create DataLoader
# --------------------------------------------------

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)


# --------------------------------------------------
# Get one batch
# --------------------------------------------------

images, captions = next(iter(loader))


# --------------------------------------------------
# Display information
# --------------------------------------------------

print(
    "Image batch shape:",
    images.shape
)

print(
    "Caption batch shape:",
    captions.shape
)

print(
    "Caption tokens:"
)

print(captions)