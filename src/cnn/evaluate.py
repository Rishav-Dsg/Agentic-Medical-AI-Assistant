import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

from src.cnn.dataset import ChestXrayDataset
from src.cnn.model import build_model

TARGET_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Pneumothorax"
]

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

dataset = ChestXrayDataset(
    csv_file="data/val.csv",
    image_dir="data/images",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False
)

model = build_model()

model.load_state_dict(
    torch.load(
        "best_model_finetuned.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        all_labels.append(
            labels.numpy()
        )

        all_preds.append(
            preds.cpu().numpy()
        )

        all_probs.append(
            probs.cpu().numpy()
        )

all_labels = np.vstack(all_labels)
all_preds = np.vstack(all_preds)
all_probs = np.vstack(all_probs)

for i, disease in enumerate(TARGET_DISEASES):

    print("\n")
    print("=" * 60)
    print(disease)

    print(
        classification_report(
            all_labels[:, i],
            all_preds[:, i],
            zero_division=0
        )
    )

    auc = roc_auc_score(
        all_labels[:, i],
        all_probs[:, i]
    )

    print(
        f"AUC: {auc:.4f}"
    )