import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocess import load_image, preprocess_image

rows = list(csv.DictReader(open(ROOT / "dataset/printed/manifest.csv", encoding="utf-8")))
data = {s: ([], []) for s in ("train", "val", "test")}
rejected = 0

for r in rows:
    img = preprocess_image(load_image(ROOT / r["path"]))
    if img is None:
        rejected += 1
        continue
    X, y = data[r["split"]]
    X.append((img * 255).astype(np.uint8))
    y.append(int(r["class_index"]))

out = {}
for s, (X, y) in data.items():
    out[f"X_{s}"] = np.array(X, dtype=np.uint8)
    out[f"y_{s}"] = np.array(y, dtype=np.int64)
    print(f"{s}: {len(y)} images")

np.savez_compressed(ROOT / "dataset.npz", **out)
print(f"Rejected by preprocessing: {rejected}")
print("Saved dataset.npz")