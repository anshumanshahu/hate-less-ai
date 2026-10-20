import pandas as pd, joblib
from sklearn.metrics import precision_score, recall_score

model = joblib.load("text_model.pkl")
test = pd.read_csv("data/test.csv", encoding="utf-8-sig")
p = model.predict_proba(test["text"])[:, 1]

for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = (p >= th).astype(int)
    print(th, "precision:", round(precision_score(test["label"], pred), 3),
          "recall:", round(recall_score(test["label"], pred), 3))