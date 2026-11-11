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


for part in ["train", "test"]:
    fn = f"feats_{part}.npz"
    if not os.path.exists(fn):
        paths, labels, classes = list_files(part)
        X, y = extract(paths, labels)
        np.savez(fn, X=X, y=y, classes=classes)
        print(f"\nSaved {fn}")

tr, te = np.load("feats_train.npz"), np.load("feats_test.npz")
classes = [str(c) for c in tr["classes"]]

clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)
clf.fit(tr["X"], tr["y"])
pred = clf.predict(te["X"])

print(classification_report(te["y"], pred, target_names=classes, digits=3))
print(confusion_matrix(te["y"], pred))

joblib.dump({"clf": clf, "classes": classes}, "image_model.pkl")
print("Saved: image_model.pkl")