import csv
import cv2
import numpy as np
from pathlib import Path
from src.preprocess import load_image, preprocess_image

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(ROOT / "dataset/printed/manifest.csv", encoding="utf-8")))
CLASSES = [4, 6, 7, 25, 26]       
fams = sorted({r["family"] for r in rows if r["split"] == "test"})

lines = []
for fam in fams:
    for c in CLASSES:
        picks = [r for r in rows if r["family"] == fam and int(r["class_index"]) == c][:4]
        cells = []
        for r in picks:
            img = load_image(ROOT / r["path"])
            orig = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 64),
                              interpolation=cv2.INTER_AREA)
            p = preprocess_image(img)
            proc = np.zeros((64, 64), np.uint8) if p is None else (255 - (p * 255)).astype(np.uint8)
            cells += [orig, proc]
        row = np.hstack(cells) if cells else np.full((64, 512), 255, np.uint8)
        label = np.full((64, 150), 255, np.uint8)
        cv2.putText(label, f"{fam} c{c}", (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        lines.append(np.hstack([label, row]))

cv2.imwrite(str(ROOT / "dots_check.png"), np.vstack(lines))
print("Saved dots_check.png")