import torch 
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from torch import Tensor 
from torch.nn import Module 
from src.cnn.model import build_model
from typing import cast 

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

gradients: torch.Tensor | None = None
activations: torch.Tensor | None = None


def forward_hook(module, inputs, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


target_layer = cast(Module,model.features.denseblock4)

target_layer.register_forward_hook(forward_hook)

target_layer.register_full_backward_hook(backward_hook)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_path = "data/images/00000001_000.png"

image = Image.open(
    image_path
).convert("RGB")

input_tensor = cast(Tensor, transform(image))
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)

output = model(input_tensor)

probabilities = torch.sigmoid(output)

print("\nPredictions:\n")

for i, disease in enumerate(TARGET_DISEASES):
    print(
        f"{disease}: "
        f"{probabilities[0, i].item():.4f}"
    )

target_index = int(
    torch.argmax(probabilities)
)

print(
    f"\nGenerating GradCAM for: "
    f"{TARGET_DISEASES[target_index]}"
)

score = output[0, target_index]

model.zero_grad()

score.backward()

if gradients is None:
    raise RuntimeError(
        "Gradients were not captured."
    )

if activations is None:
    raise RuntimeError(
        "Activations were not captured."
    )

pooled_gradients = torch.mean(
    gradients,
    dim=(0, 2, 3)
)

activation_map = activations[0].clone()

for i in range(
    pooled_gradients.shape[0]
):
    activation_map[i] *= pooled_gradients[i]

heatmap = torch.mean(
    activation_map,
    dim=0
)

heatmap = heatmap.detach().cpu().numpy()

heatmap = np.maximum(
    heatmap,
    0
)

if np.max(heatmap) > 0:
    heatmap /= np.max(heatmap)

image_cv = cv2.imread(
    image_path
)

if image_cv is None:
    raise FileNotFoundError(
        image_path
    )

heatmap = cv2.resize(
    heatmap,
    (
        image_cv.shape[1],
        image_cv.shape[0]
    )
)

heatmap = np.uint8(
    255 * heatmap
)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

superimposed = cv2.addWeighted(
    image_cv,
    0.6,
    heatmap,
    0.4,
    0
)

output_path = (
    "outputs/gradcam/"
    "gradcam_output.png"
)

cv2.imwrite(
    output_path,
    superimposed
)

print(
    f"\nSaved: {output_path}"
)