import cv2
import numpy as np

IMG_SIZE = 64
MARGIN = 0.15          
MIN_INK = 0.01        
MAX_INK = 0.60         


def load_image(path):
    """Read an image; np.fromfile avoids problems with non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def preprocess_image(image):
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.uint8)
    h, w = gray.shape

    k = (min(h, w) // 2) | 1                   
    background = cv2.GaussianBlur(gray, (k, k), 0)
    flat = cv2.divide(gray, background, scale=255)
    flat = cv2.GaussianBlur(flat, (3, 3), 0)

    # 2) Need real contrast between ink and paper
    if int(flat.max()) - int(flat.min()) < 40:
        return None

    # 3) Otsu threshold: dark ink -> white (255), paper -> black (0)
    _, binary = cv2.threshold(flat, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(4, int(0.0015 * h * w))
    clean = np.zeros_like(binary)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

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