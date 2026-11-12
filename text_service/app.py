import re
import unicodedata
import joblib
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pii import find_pii

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


def personal_info_result(types, worst_text=None):
    res = {"status": "reject", "probability": 1.0, "reason": "personal_info", "types": types}
    if worst_text is not None:
        res["worst_text"] = worst_text
    return res


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check-text")
def check_text(body: TextIn):
    # 1. personal info pehle, ML se pehle
    pii = find_pii(body.text)
    if pii:
        return personal_info_result(pii)

    # 2. ML model
    clean = norm(body.text)
    if len(clean) < 2:
        return {"status": "clean", "probability": 0.0}
    p = float(model.predict_proba([clean])[0][1])
    return {"status": to_status(p), "probability": round(p, 4)}


@app.post("/check-text-many")
def check_text_many(body: ManyIn):
    # koi bhi piece me personal info ho to poori request reject
    for t in body.texts:
        pii = find_pii(t)
        if pii:
            return personal_info_result(pii, worst_text="")

    cleaned = [c for c in (norm(t) for t in body.texts) if len(c) >= 2]
    if not cleaned:
        return {"status": "clean", "probability": 0.0, "worst_text": ""}
    probs = model.predict_proba(cleaned)[:, 1]
    i = int(probs.argmax())
    p = float(probs[i])
    return {"status": to_status(p), "probability": round(p, 4), "worst_text": cleaned[i]}