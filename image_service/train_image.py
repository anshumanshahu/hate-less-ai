import os, numpy as np, torch, joblib
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_num_threads(os.cpu_count())

MODEL = "openai/clip-vit-base-patch32"
clip = CLIPModel.from_pretrained(MODEL).eval()
proc = CLIPProcessor.from_pretrained(MODEL)

def list_files(part):
    classes = sorted(os.listdir(f"images/{part}"))
    paths, labels = [], []
    for i, c in enumerate(classes):
        for f in os.listdir(f"images/{part}/{c}"):
            paths.append(f"images/{part}/{c}/{f}")
            labels.append(i)
    return paths, labels, classes

def extract(paths, labels, bs=32):
    X, y = [], []
    for i in range(0, len(paths), bs):
        imgs, lbl = [], []
        for p, l in zip(paths[i:i + bs], labels[i:i + bs]):
            try:
                imgs.append(Image.open(p).convert("RGB"))
                lbl.append(l)
            except Exception:
                print("skip:", p)
        if not imgs:
            continue
        with torch.no_grad():
            f = clip.get_image_features(**proc(images=imgs, return_tensors="pt"))
        if not torch.is_tensor(f):
            f = f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        X.append(f.numpy())
        y += lbl
        print(f"{min(i + bs, len(paths))}/{len(paths)}", end="\r")
    return np.vstack(X), np.array(y)

if not os.path.exists("img_feats.npz"):
    ptr, ltr, classes = list_files("train")
    pte, lte, _ = list_files("test")
    Xtr, ytr = extract(ptr, ltr)
    Xte, yte = extract(pte, lte)
    np.savez("img_feats.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, classes=classes)

d = np.load("img_feats.npz")
classes = [str(c) for c in d["classes"]]

clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)
clf.fit(d["Xtr"], d["ytr"])
pred = clf.predict(d["Xte"])

print(classification_report(d["yte"], pred, target_names=classes, digits=3))
print(confusion_matrix(d["yte"], pred))

joblib.dump({"clf": clf, "classes": classes}, "image_model.pkl")
print("Saved: image_model.pkl")