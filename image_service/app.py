import os
import unicodedata
import joblib
import torch
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPProcessor

from ocr import extract_text, available_langs

ImageFile.LOAD_TRUNCATED_IMAGES = True

TEXT_MANY_URL = os.getenv("TEXT_SERVICE_MANY_URL", "http://127.0.0.1:8001/check-text-many")
REJECT_TH, REVIEW_TH = 0.80, 0.50
MIN_OCR_WORDS = 3          # ignore OCR output with fewer words (usually noise)
CHUNK_WORDS, CHUNK_STEP = 8, 4

MODEL = "openai/clip-vit-base-patch32"
clip = CLIPModel.from_pretrained(MODEL).eval()
proc = CLIPProcessor.from_pretrained(MODEL)
bundle = joblib.load("image_model.pkl")
clf, classes = bundle["clf"], bundle["classes"]

app = FastAPI(title="Image Moderation Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RANK = {"clean": 0, "review": 1, "reject": 2}


def classify(img):
    with torch.no_grad():
        f = clip.get_image_features(**proc(images=[img], return_tensors="pt"))
    if not torch.is_tensor(f):
        f = f.pooler_output
    f = (f / f.norm(dim=-1, keepdim=True)).numpy()
    return dict(zip(classes, clf.predict_proba(f)[0]))


def word_chars(token):
    # letters, digits and combining marks (needed for Devanagari matras)
    return sum(ch.isalnum() or unicodedata.category(ch).startswith("M") for ch in token)


def clean_ocr(text):
    """Drop stray symbols and single characters that OCR picks up from pictures."""
    return " ".join(tok for tok in text.split() if word_chars(tok) >= 2)


def make_chunks(text):
    """Whole text plus short overlapping windows, so one abusive line is not
    diluted by a long harmless text around it."""
    words = text.split()
    pieces = [text]
    if len(words) > CHUNK_WORDS:
        for i in range(0, len(words), CHUNK_STEP):
            window = words[i:i + CHUNK_WORDS]
            if len(window) >= MIN_OCR_WORDS:
                pieces.append(" ".join(window))
    return pieces


@app.get("/health")
def health():
    return {"status": "ok", "ocr_languages": sorted(available_langs())}


@app.post("/check-image")
def check_image(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 1. image classification
    probs = classify(img)
    unsafe = float(1 - probs["safe"])
    top = max((c for c in probs if c != "safe"), key=lambda c: probs[c])
    if unsafe >= REJECT_TH:
        img_status = "reject"
    elif unsafe >= REVIEW_TH:
        img_status = "review"
    else:
        img_status = "clean"

    # 2. OCR, cleanup, then text service
    raw_text, text_res = "", None
    try:
        raw_text = extract_text(img)
    except Exception:
        raw_text = ""
    text = clean_ocr(raw_text)
    if len(text.split()) >= MIN_OCR_WORDS:
        try:
            r = httpx.post(TEXT_MANY_URL, json={"texts": make_chunks(text)}, timeout=15)
            text_res = r.json()
        except Exception:
            text_res = {"status": "unavailable"}

    # 3. final decision = the stricter of the two
    final = img_status
    if text_res and text_res.get("status") in RANK:
        if RANK[text_res["status"]] > RANK[final]:
            final = text_res["status"]

    return {
        "status": final,
        "image": {
            "status": img_status,
            "label": top if img_status != "clean" else "safe",
            "unsafe_score": round(unsafe, 4),
            "scores": {c: round(float(p), 4) for c, p in probs.items()},
        },
        "ocr": {"text": text, "result": text_res},
    }