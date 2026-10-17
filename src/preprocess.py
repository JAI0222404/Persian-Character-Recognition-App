import cv2
import numpy as np

IMG_SIZE = 64
MARGIN = 0.15
MIN_INK = 0.002
MAX_INK = 0.60
WORK_SIZE = 128
BG_SIDE = 64
BG_KERNEL = 0.5


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray.astype(np.uint8)


def ensure_min_size(image, min_side=WORK_SIZE):
    h, w = image.shape[:2]
    if min(h, w) >= min_side:
        return image
    s = min_side / min(h, w)
    return cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def flatten(gray):
    """Remove uneven lighting. Closing erases dark ink; a blur would average it in."""
    h, w = gray.shape
    s = min(1.0, BG_SIDE / min(h, w))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    k = max(3, int(BG_KERNEL * min(small.shape))) | 1
    paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    paper = cv2.GaussianBlur(paper, (k, k), 0)
    paper = cv2.resize(paper, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray, np.maximum(paper, 1), scale=255)


def preprocess_image(image):
    if image is None or image.size == 0:
        return None
    image = ensure_min_size(image)
    gray = to_gray(image)
    h, w = gray.shape

    flat = cv2.GaussianBlur(flatten(gray), (3, 3), 0)
    if int(flat.max()) - int(flat.min()) < 40:
        return None

    _, binary = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count < 2:
        return None
    areas = stats[:, cv2.CC_STAT_AREA]
    min_area = max(2, min(int(0.0015 * h * w), int(0.02 * areas[1:].max())))
    keep = areas >= min_area
    keep[0] = False
    clean = (keep[labels].astype(np.uint8)) * 255

    ink_fraction = np.count_nonzero(clean) / clean.size
    if ink_fraction < MIN_INK or ink_fraction > MAX_INK:
        return None

    x, y, bw, bh = cv2.boundingRect(cv2.findNonZero(clean))
    letter = clean[y:y + bh, x:x + bw]
    side = max(bw, bh)
    pad = int(MARGIN * side)
    canvas = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.uint8)
    top = pad + (side - bh) // 2
    left = pad + (side - bw) // 2
    canvas[top:top + bh, left:left + bw] = letter
    small = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0