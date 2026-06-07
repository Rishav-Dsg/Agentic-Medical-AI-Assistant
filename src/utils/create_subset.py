import pandas as pd
import os

TARGET_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Pneumothorax"
]

df = pd.read_csv("data/Data_Entry_2017.csv")

available_images = set(
    os.listdir("data/images")
)

df = df[
    df["Image Index"].isin(
        available_images
    )
]

def keep_row(labels):
    return any(
        disease in labels
        for disease in TARGET_DISEASES
    )

subset = df[
    df["Finding Labels"].apply(
        keep_row
    )
]

print("Subset size:", len(subset))

subset.to_csv(
    "data/subset.csv",
    index=False
)

print("Saved subset.csv")