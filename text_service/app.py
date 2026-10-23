import re
import unicodedata
import joblib
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REJECT_TH, REVIEW_TH = 0.8, 0.5
model = joblib.load("text_model.pkl")

app = FastAPI(title="Text Moderation Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextIn(BaseModel):
    text: str


class ManyIn(BaseModel):
    texts: List[str]


def norm(t):
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"http\S+|@\w+", " ", t)
    t = t.replace("#", " ")
    return re.sub(r"\s+", " ", t).strip()


def to_status(p):
    if p >= REJECT_TH:
        return "reject"
    if p >= REVIEW_TH:
        return "review"
    return "clean"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check-text")
def check_text(body: TextIn):
    clean = norm(body.text)
    if len(clean) < 2:
        return {"status": "clean", "probability": 0.0}
    p = float(model.predict_proba([clean])[0][1])
    return {"status": to_status(p), "probability": round(p, 4)}


@app.post("/check-text-many")
def check_text_many(body: ManyIn):
    """Score several pieces of text and report the worst one."""
    cleaned = [c for c in (norm(t) for t in body.texts) if len(c) >= 2]
    if not cleaned:
        return {"status": "clean", "probability": 0.0, "worst_text": ""}
    probs = model.predict_proba(cleaned)[:, 1]
    i = int(probs.argmax())
    p = float(probs[i])
    return {"status": to_status(p), "probability": round(p, 4), "worst_text": cleaned[i]}