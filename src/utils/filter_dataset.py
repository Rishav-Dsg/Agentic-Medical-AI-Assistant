import pandas as pd
import os 

df = pd.read_csv("data/Data_Entry_2017.csv")

available_images = set(os.listdir("data/images"))

df = df[df["Image Index"].isin(available_images)]

target_diseases = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Pneumothorax"
]

for disease in target_diseases:
    count = df["Finding Labels"].str.contains(disease).sum()
    print(f"{disease}: {count}")