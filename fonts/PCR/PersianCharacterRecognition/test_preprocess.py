import csv
import random
import cv2
import numpy as np
from pathlib import Path
from src.preprocess import load_image, preprocess_image

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(ROOT / "dataset/printed/manifest.csv", encoding="utf-8")))
random.seed(1)
picked = random.sample(rows, 12)


def degrade(img):
    h, w = img.shape[:2]
    shadow = np.linspace(1.0, 0.45, w, dtype=np.float32)[None, :, None]
    out = img.astype(np.float32) * shadow
    out = cv2.GaussianBlur(out, (7, 7), 0)
    out += np.random.normal(0, 8, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


cols = []
for r in picked:
    img = load_image(ROOT / r["path"])
    bad = degrade(img)
    show = lambda a: cv2.resize(a, (64, 64), interpolation=cv2.INTER_AREA)
    gray = lambda a: cv2.cvtColor(show(a), cv2.COLOR_BGR2GRAY)
    def out(a):
        p = preprocess_image(a)
        return np.zeros((64, 64), np.uint8) if p is None else (p * 255).astype(np.uint8)
    cols.append(np.vstack([gray(img), gray(bad), out(img), out(bad)]))

cv2.imwrite(str(ROOT / "preprocess_check.png"), np.hstack(cols))
print("Saved preprocess_check.png (rows: original, degraded, processed original, processed degraded)")