import joblib, re, unicodedata

model = joblib.load("text_model.pkl")

REJECT_TH = 0.8
REVIEW_TH = 0.5

def norm(t):
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"http\S+|@\w+", " ", t)
    t = t.replace("#", " ")
    return re.sub(r"\s+", " ", t).strip()

def check_text(text):
    p = model.predict_proba([norm(text)])[0][1]
    if p >= REJECT_TH:
        return "REJECT", p
    if p >= REVIEW_TH:
        return "REVIEW", p
    return "CLEAN", p

print("Text daalo (band karne ke liye 'q' likho)\n")
while True:
    text = input("> ").strip()
    if text.lower() == "q":
        break
    if not text:
        continue
    status, p = check_text(text)
    print(f"   {status}  (vulgar probability: {p:.2f})\n")