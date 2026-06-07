import torch
from PIL import Image
from torchvision import transforms

from typing import cast 
from src.cnn.model import build_model
from torch import Tensor

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

model = build_model()

model.load_state_dict(
    torch.load(
        "models/best_model_finetuned.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def predict_xray(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = cast(Tensor, transform(image))

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        outputs = model(tensor)

        probs = torch.sigmoid(outputs)

    probs = probs.squeeze()

    results = {}

    for i, disease in enumerate(
        TARGET_DISEASES
    ):
        results[disease] = round(
            probs[i].item(),
            4
        )

    return results


if __name__ == "__main__":

    results = predict_xray(
        "data/images/00000001_000.png"
    )

    print(results)

def get_top_diseases(results, threshold=0.5):

    diseases = []

    for disease, prob in results.items():

        if prob >= threshold:
            diseases.append(
                (disease, prob)
            )

    diseases.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return diseases