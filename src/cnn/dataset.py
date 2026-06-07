import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

TARGET_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Pneumothorax"
]

class ChestXrayDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None
    ):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = (
            f"{self.image_dir}/"
            f"{row['Image Index']}"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        labels = torch.tensor(
            row[TARGET_DISEASES].values.astype("float32")
        )

        if self.transform:
            image = self.transform(image)

        return image, labels