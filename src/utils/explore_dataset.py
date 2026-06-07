import pandas as pd

df = pd.read_csv("./data/Data_Entry_2017.csv")

print("Total records:", len(df))

print("\nTop labels:\n")
print(df["Finding Labels"].value_counts().head(20))