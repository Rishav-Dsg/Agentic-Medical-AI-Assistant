import pandas as pd
import torch

TARGET_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Pneumothorax"
]

df = pd.read_csv("data/train.csv")

weights = []

for disease in TARGET_DISEASES:

    positives = df[disease].sum()

    negatives = len(df) - positives

    weight = negatives / positives

    weights.append(weight)

    print(
        disease,
        round(weight, 2)
    )

weights = torch.tensor(
    weights,
    dtype=torch.float32
)

print("\nTensor:")
print(weights)