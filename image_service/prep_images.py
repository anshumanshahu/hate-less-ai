import os, hashlib, random, shutil
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SRC, DST = "raw_images", "images"
MAX_SAFE = None       
random.seed(42)

seen, items = set(), {}
for cls in sorted(os.listdir(SRC)):
    folder = f"{SRC}/{cls}"
    if not os.path.isdir(folder):
        continue
    for f in os.listdir(folder):
        p = f"{folder}/{f}"
        try:
            with open(p, "rb") as fh:
                h = hashlib.md5(fh.read()).hexdigest()
            if h in seen:
                continue
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if min(im.size) < 64:
                    continue
        except Exception:
            continue
        seen.add(h)
        items.setdefault(cls, []).append(p)

if MAX_SAFE and "safe" in items:
    random.shuffle(items["safe"])
    items["safe"] = items["safe"][:MAX_SAFE]

for cls, lst in items.items():
    random.shuffle(lst)
    k = int(len(lst) * 0.8)
    for part, chunk in [("train", lst[:k]), ("test", lst[k:])]:
        os.makedirs(f"{DST}/{part}/{cls}", exist_ok=True)
        for p in chunk:
            shutil.copy(p, f"{DST}/{part}/{cls}/{os.path.basename(p)}")
    print(cls, len(lst))