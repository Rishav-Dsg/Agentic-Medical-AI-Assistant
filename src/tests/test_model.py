# src/test_model.py

from src.cnn.model import build_model
import torch

model = build_model()

x = torch.randn(
    4,
    3,
    224,
    224
)

y = model(x)

print(y.shape)