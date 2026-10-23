import re, unicodedata, joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REJECT_TH, REVIEW_TH = 0.8, 0.5
model = joblib.load("text_model.pkl")

app = FastAPI(title="Text Moderation Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class TextIn(BaseModel):
    text: str

def norm(t):
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"http\S+|@\w+", " ", t)
    t = t.replace("#", " ")
    return re.sub(r"\s+", " ", t).strip()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/check-text")
def check_text(body: TextIn):
    clean = norm(body.text)
    if len(clean) < 2:
        return {"status": "clean", "probability": 0.0}
    p = float(model.predict_proba([clean])[0][1])
    status = "reject" if p >= REJECT_TH else "review" if p >= REVIEW_TH else "clean"
    return {"status": status, "probability": round(p, 4)}