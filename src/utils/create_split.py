import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "data/encoded_labels.csv"
)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_df.to_csv(
    "data/train.csv",
    index=False
)

val_df.to_csv(
    "data/val.csv",
    index=False
)

print("Train:", len(train_df))
print("Val:", len(val_df))