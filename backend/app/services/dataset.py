import os

from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms


class ImageCaptionDataset(Dataset):

    def __init__(self, image_dir, captions_file, vocabulary):

        self.image_dir = image_dir
        self.vocabulary = vocabulary

        self.data = []

        # Read captions.txt
        with open(captions_file, "r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                image_name, caption = line.split("|", 1)

                self.data.append(
                    (image_name, caption)
                )

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_name, caption = self.data[index]

        # Build image path
        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Apply preprocessing
        image = self.transform(image)

        # Convert caption words to token IDs
        tokens = self.vocabulary.numericalize(caption)

        caption_tensor = torch.tensor(
            tokens,
            dtype=torch.long
        )

        return image, caption_tensor


def collate_fn(batch):

    images = []
    captions = []

    # Separate images and captions
    for image, caption in batch:

        images.append(image)
        captions.append(caption)

    # Combine images into one batch
    images = torch.stack(images)

    # Pad captions to equal length
    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0
    )

    return images, captions