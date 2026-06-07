import pandas as pd

TARGET_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Pneumothorax"
]

df = pd.read_csv("data/subset.csv")

for disease in TARGET_DISEASES:
    df[disease] = df["Finding Labels"].apply(
        lambda x: int(disease in x)
    )

columns = [
    "Image Index",
    *TARGET_DISEASES
]

df = df[columns]

print(df.head())

df.to_csv(
    "data/encoded_labels.csv",
    index=False
)

print("Saved encoded_labels.csv")