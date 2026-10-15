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
VARIANTS = 20          
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


def render(font_path: Path, ch: str, stroke: int, angle: float) -> Image.Image:
    font = ImageFont.truetype(str(font_path), FONT_SIZE)
    img = Image.new("L", (CANVAS, CANVAS), 255)    
    ImageDraw.Draw(img).text(
        (CANVAS / 2, CANVAS / 2), ch, font=font, fill=0, anchor="mm",
        stroke_width=stroke, stroke_fill=0)               
    if angle:
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
    return img


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
                stroke = random.choice([0, 0, 1, 2])
                angle = random.uniform(-5, 5)
                img = render(font_path, ch, stroke, angle)
                if not is_good(img):
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