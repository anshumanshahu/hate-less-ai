import os, joblib, torch, httpx, pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFile
from transformers import CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True
TEXT_URL = os.getenv("TEXT_SERVICE_URL", "http://127.0.0.1:8001/check-text")
REJECT_TH, REVIEW_TH = 0.80, 0.50

MODEL = "openai/clip-vit-base-patch32"
clip = CLIPModel.from_pretrained(MODEL).eval()
proc = CLIPProcessor.from_pretrained(MODEL)
bundle = joblib.load("image_model.pkl")
clf, classes = bundle["clf"], bundle["classes"]

app = FastAPI(title="Image Moderation Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

RANK = {"clean": 0, "review": 1, "reject": 2}

def classify(img):
    with torch.no_grad():
        f = clip.get_image_features(**proc(images=[img], return_tensors="pt"))
    if not torch.is_tensor(f):
        f = f.pooler_output
    f = (f / f.norm(dim=-1, keepdim=True)).numpy()
    return dict(zip(classes, clf.predict_proba(f)[0]))

def ocr_text(img):
    g = ImageOps.grayscale(img)
    g.thumbnail((2000, 2000))
    if min(g.size) < 800:
        g = g.resize((g.width * 2, g.height * 2))
    g = ImageOps.autocontrast(g)
    t = pytesseract.image_to_string(g, lang="eng+hin", config="--psm 6")
    return " ".join(t.split())

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/check-image")
def check_image(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    probs = classify(img)
    unsafe = float(1 - probs["safe"])
    top = max((c for c in probs if c != "safe"), key=lambda c: probs[c])
    img_status = "reject" if unsafe >= REJECT_TH else "review" if unsafe >= REVIEW_TH else "clean"

    # OCR + text service
    text, text_res = "", None
    try:
        text = ocr_text(img)
    except Exception:
        pass
    if len(text.split()) >= 3:          # 1-2 random akshar (noise) ko ignore karo
        try:
            text_res = httpx.post(TEXT_URL, json={"text": text}, timeout=10).json()
        except Exception:
            text_res = {"status": "unavailable"}

    final = img_status
    if text_res and text_res.get("status") in RANK and RANK[text_res["status"]] > RANK[final]:
        final = text_res["status"]

    return {
        "status": final,
        "image": {"status": img_status, "label": top if img_status != "clean" else "safe",
                  "unsafe_score": round(unsafe, 4),
                  "scores": {c: round(float(p), 4) for c, p in probs.items()}},
        "ocr": {"text": text, "result": text_res},
    }