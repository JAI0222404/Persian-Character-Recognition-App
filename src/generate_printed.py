import math
import cv2
import numpy as np
import csv
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.labels import LETTERS

FONT_DIR = ROOT / "fonts"
OUT_DIR = ROOT / "dataset" / "printed"
CANVAS = 128          
VARIANTS = 24          
FONT_SIZE = 80
TEST_FAMILIES = {"Almarai"}
VAL_FAMILIES = {"MarkaziText"}
random.seed(42)


def family_of(path: Path) -> str:
    return re.split(r"[-\[]", path.stem)[0]


def split_of(family: str) -> str:
    if family in TEST_FAMILIES:
        return "test"
    if family in VAL_FAMILIES:
        return "val"
    return "train"


def font_has_all_letters(path: Path) -> bool:
    try:
        cmap = TTFont(str(path), lazy=True).getBestCmap()
    except Exception as err:
        print(f"  SKIP {path.name}: cannot read ({err})")
        return False
    missing = [c for c, _, _ in LETTERS if ord(c) not in cmap]
    if missing:
        print(f"  SKIP {path.name}: missing {len(missing)} letters")
        return False
    return True


def render(font_path: Path, ch: str, stroke: int) -> Image.Image:
    font = ImageFont.truetype(str(font_path), FONT_SIZE)
    img = Image.new("L", (CANVAS, CANVAS), 255)
    ImageDraw.Draw(img).text(
        (CANVAS / 2, CANVAS / 2), ch, font=font, fill=0, anchor="mm",
        stroke_width=stroke, stroke_fill=0)
    return img


def transform(img: Image.Image, angle: float, zoom: float) -> Image.Image:
    arr = np.array(img)
    if angle:
        m = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), angle, 1.0)
        arr = cv2.warpAffine(arr, m, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    if zoom > 1:
        c = int(CANVAS / zoom)
        o = (CANVAS - c) // 2
        arr = cv2.resize(arr[o:o + c, o:o + c], (CANVAS, CANVAS), interpolation=cv2.INTER_CUBIC)
    elif zoom < 1:
        n = max(4, int(CANVAS * zoom))
        small = cv2.resize(arr, (n, n), interpolation=cv2.INTER_AREA)
        arr = np.full((CANVAS, CANVAS), 255, np.uint8)
        o = (CANVAS - n) // 2
        arr[o:o + n, o:o + n] = small
    return Image.fromarray(arr)


def random_rotation() -> float:
    if random.random() < 0.8:
        return random.choice([90, 180, 270]) + random.uniform(-10, 10)
    return random.uniform(0, 360)


def sample_geometry():
    tilt = random.uniform(-5, 5)
    r = random.random()
    if r < 0.35:
        return tilt, 1.0
    if r < 0.60:
        return tilt, math.exp(random.uniform(math.log(0.25), math.log(1.35)))
    if r < 0.85:
        return random_rotation(), 1.0
    return random_rotation(), math.exp(random.uniform(math.log(0.3), math.log(1.2)))


def is_good(img: Image.Image) -> bool:
    """Reject blank images or letters clipped by the canvas edge."""
    box = ImageOps.invert(img).getbbox()
    if box is None:
        return False
    left, top, right, bottom = box
    return left > 0 and top > 0 and right < CANVAS and bottom < CANVAS


def main() -> None:
    font_paths = sorted(FONT_DIR.glob("*.ttf")) + sorted(FONT_DIR.glob("*.otf"))
    if not font_paths:
        sys.exit(f"No fonts found in {FONT_DIR}")

    print("Checking fonts...")
    usable = [p for p in font_paths if font_has_all_letters(p)]
    print(f"Usable fonts: {len(usable)} of {len(font_paths)}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    rows, skipped = [], 0
    for font_path in usable:
        family = family_of(font_path)
        split = split_of(family)
        for index, (ch, name, _) in enumerate(LETTERS):
            class_dir = OUT_DIR / split / f"{index:02d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            for n in range(VARIANTS):
                img = None
                for _ in range(8):
                    stroke = random.choice([0, 0, 1, 2])
                    angle, zoom = sample_geometry()
                    candidate = transform(render(font_path, ch, stroke), angle, zoom)
                    if is_good(candidate):
                        img = candidate
                        break
                if img is None:
                    skipped += 1
                    continue
                file = class_dir / f"{font_path.stem}_{n}.png"
                img.save(file)
                rows.append([file.relative_to(ROOT).as_posix(),
                             index, family, split])

    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class_index", "family", "split"])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} images, skipped {skipped} bad renders")
    for split in ("train", "val", "test"):
        n = sum(1 for r in rows if r[3] == split)
        fams = sorted({r[2] for r in rows if r[3] == split})
        print(f"  {split}: {n} images, families: {', '.join(fams)}")


if __name__ == "__main__":
    main()