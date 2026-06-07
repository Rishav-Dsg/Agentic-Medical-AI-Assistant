import pandas as pd

df = pd.read_csv("data/subset.csv")

for disease in [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Pneumothorax"
]:
    count = (
        df["Finding Labels"]
        .str.contains(disease)
        .sum()
    )
    print(disease, count)