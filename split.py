import pandas as pd, os
from sklearn.model_selection import train_test_split

df = pd.read_csv("merged.csv", encoding="utf-8-sig")
print("Rows (pehle):", len(df))


df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(int)
df = df[df["text"].str.len() > 2]

n = df.groupby("text")["label"].nunique()
df = df[~df["text"].isin(n[n > 1].index)]   
df = df.drop_duplicates(subset="text")
print("Rows (baad me):", len(df))
print(df["label"].value_counts())

# 80/20 split
train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

os.makedirs("data", exist_ok=True)
train.to_csv("data/train.csv", index=False, encoding="utf-8")
test.to_csv("data/test.csv", index=False, encoding="utf-8")
print("Train:", len(train), "| Test:", len(test))