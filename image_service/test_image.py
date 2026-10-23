import os, sys, joblib, torch
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL = "openai/clip-vit-base-patch32"
clip = CLIPModel.from_pretrained(MODEL).eval()
proc = CLIPProcessor.from_pretrained(MODEL)
bundle = joblib.load("image_model.pkl")
clf, classes = bundle["clf"], bundle["classes"]

REJECT_TH = 0.80
REVIEW_TH = 0.50
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

def predict(path):
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        f = clip.get_image_features(**proc(images=[img], return_tensors="pt"))
    if not torch.is_tensor(f):
        f = f.pooler_output
    f = (f / f.norm(dim=-1, keepdim=True)).numpy()
    return dict(zip(classes, clf.predict_proba(f)[0]))

def decide(probs):
    unsafe = 1 - probs["safe"]
    top = max((c for c in probs if c != "safe"), key=lambda c: probs[c])
    if unsafe >= REJECT_TH:
        return "REJECT", top, unsafe
    if unsafe >= REVIEW_TH:
        return "REVIEW", top, unsafe
    return "CLEAN", "safe", unsafe

def show(path):
    try:
        probs = predict(path)
    except Exception as e:
        print("   error:", e)
        return
    status, label, unsafe = decide(probs)
    print(f"   {status} ({label})  unsafe score: {unsafe:.2f}")
    for c, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"      {c:18s} {p:.2f}")

def run(path):
    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(EXTS):
                print(f)
                show(os.path.join(path, f))
    else:
        show(path)

if len(sys.argv) > 1:
    run(sys.argv[1])
else:
    print("Image ka path ya folder daalo (band karne ke liye 'q')\n")
    while True:
        p = input("> ").strip().strip("'\"")
        if p.lower() == "q":
            break
        if p:
            run(p)
            print()