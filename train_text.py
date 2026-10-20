import pandas as pd, joblib, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

train = pd.read_csv("data/train.csv", encoding="utf-8-sig")
test = pd.read_csv("data/test.csv", encoding="utf-8-sig")
print("Train:", len(train), "| Test:", len(test))

model = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                    max_features=300000, min_df=3,
                    sublinear_tf=True, dtype="float32"),
    LogisticRegression(max_iter=300, class_weight="balanced",
                       solver="saga", C=2.0),
)

print("Training Started...")
model.fit(train["text"], train["label"])
joblib.dump(model, "text_model.pkl")
print("Model saved as: text_model.pkl")

# Overall score
test["pred"] = model.predict(test["text"])
print("\n=== OVERALL ===")
print(classification_report(test["label"], test["pred"], digits=3))


is_hi = test["text"].str.contains(r"[\u0900-\u097F]", regex=True)
for name, part in [("HINDI (Devanagari)", test[is_hi]), ("ROMAN/ENGLISH", test[~is_hi])]:
    if len(part):
        print(f"\n=== {name} ({len(part)} rows) ===")
        print(classification_report(part["label"], part["pred"], digits=3))

# Source-wise accuracy (If source column exists)
if "source" in test.columns:
    print("\n=== SOURCE-WISE ACCURACY ===")
    print(test.groupby("source").apply(lambda g: (g["pred"] == g["label"]).mean()))