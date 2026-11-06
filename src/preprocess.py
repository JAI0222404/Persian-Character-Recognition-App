import cv2
import numpy as np

IMG_SIZE = 64
MARGIN = 0.15
MIN_INK = 0.002        # fraction of pixels that must be ink (letters can be small in the frame)
MAX_INK = 0.60
WORK_SIZE = 128        # inputs smaller than this are upscaled so tiny letters keep their dots
FIT_SIDE = 96          # the lighting model is fitted on a copy this small (fast, size-independent)
STROKE_TARGET = 0.075  # preferred pen width as a fraction of the letter size (printed fonts ~0.08-0.12)


def load_image(path):
    """Read an image; np.fromfile avoids problems with non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray.astype(np.uint8)


def ensure_min_size(image, min_side=WORK_SIZE):
    """Upscale small images so a zoomed-out (tiny) letter still has usable detail."""
    h, w = image.shape[:2]
    if min(h, w) >= min_side:
        return image
    s = min_side / min(h, w)
    return cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def _basis(rows, cols):
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float32)
    x = x / max(cols - 1, 1) * 2 - 1
    y = y / max(rows - 1, 1) * 2 - 1
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)


def flatten(gray):
    """Divide out uneven lighting (shadows, vignetting) so paper becomes ~255 everywhere.

    The paper brightness is modelled as a smooth quadratic surface fitted only to the paper
    pixels: dark outliers (the letter, grid lines) and bright outliers (glare) are rejected
    and the fit is repeated. Unlike a blur, a letter that fills the frame cannot be absorbed
    into the background, and unlike Otsu on the raw photo, darker page edges are not
    mistaken for ink when the real letter is tiny.
    """
    h, w = gray.shape
    s = min(1.0, FIT_SIDE / min(h, w))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    small = small.astype(np.float32)
    A = _basis(*small.shape).reshape(-1, 6)
    b = small.reshape(-1)

    keep = b >= np.percentile(b, 35)
    coef = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
    for _ in range(5):
        fit = A @ coef
        new_keep = (b > fit * 0.94) & (b < fit * 1.2)
        if new_keep.sum() < 0.2 * b.size:
            break
        keep = new_keep
        coef = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]

    paper = np.tensordot(_basis(h, w), coef, axes=([2], [0]))
    paper = np.maximum(paper, 1.0)
    return np.clip(gray.astype(np.float32) / paper * 255.0, 0, 255).astype(np.uint8)


def _drop_edge_strips(mask):
    """Remove thin strips that run along (almost) the whole crop: the edge of the page."""
    h, w = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, count):
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if (bh >= 0.9 * h and bw <= 0.08 * w) or (bw >= 0.9 * w and bh <= 0.08 * h):
            mask[labels == i] = 0
    return mask


SOLID_EDGE = 0.09      # a blob touching the border thicker than this (fraction of the image) is background
SOLID_ANY = 0.16       # ...and any blob thicker than this


def drop_solid_blobs(mask):
    """Remove big solid dark areas (a shadow, the table behind the page).

    Letters and pen strokes are thin: even a zoomed-in printed letter stays under ~8% of the
    image across its thickest stroke, while a dark patch of background is many times thicker.
    """
    h, w = mask.shape
    side = min(h, w)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < 200:
            continue
        region = labels[y:y + bh, x:x + bw] == i
        radius = dist[y:y + bh, x:x + bw][region].max()
        edge = x <= 0 or y <= 0 or x + bw >= w or y + bh >= h
        if radius > (SOLID_EDGE if edge else SOLID_ANY) * side:
            mask[labels == i] = 0
    return mask


def _thicken(mask):
    """Pen strokes are much thinner than printed glyphs; grow them to a printed-like width."""
    x, y, bw, bh = cv2.boundingRect(cv2.findNonZero(mask))
    side = max(bw, bh)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    width = 2.2 * np.percentile(dist[dist > 0], 90)
    grow = int(round(STROKE_TARGET * side - width))
    if grow >= 2:
        k = grow | 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def preprocess_image(image):
    if image is None or image.size == 0:
        return None

    image = ensure_min_size(image)
    gray = to_gray(image)
    h, w = gray.shape

    flat = cv2.GaussianBlur(flatten(gray), (3, 3), 0)

    # Need real contrast between ink and paper
    if int(flat.max()) - int(flat.min()) < 40:
        return None

    # Otsu threshold (capped so paper noise is never taken for ink): dark ink -> 255
    thr, _ = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = ((flat < min(thr, 200)) * 255).astype(np.uint8)
    binary = drop_solid_blobs(_drop_edge_strips(binary))

    count, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count < 2:
        return None

    # Drop specks. The limit is relative to the image for big letters, but never more than
    # 2% of the largest blob, so the dots of a small letter are not mistaken for noise.
    areas = stats[:, cv2.CC_STAT_AREA]
    min_area = max(2, min(int(0.0015 * h * w), int(0.02 * areas[1:].max())))
    keep = areas >= min_area
    keep[0] = False

    # Drop leftovers of the background (a corner of a shadow) that touch the border of the crop
    # and sit in its outer ring. The letter itself is the heavy blob closest to the centre.
    def touches(i):
        x, y, bw, bh = stats[i, :4]
        return x <= 0 or y <= 0 or x + bw >= w or y + bh >= h

    def score(i):
        cx, cy = cents[i]
        centrality = (1.0 - np.hypot(cx / w - 0.5, cy / h - 0.5)) ** 2
        return areas[i] * centrality * (0.4 if touches(i) else 1.0)

    main = max((i for i in range(1, count) if keep[i]), key=score)
    for i in range(1, count):
        # only fragments about as big as the letter itself: dots are always much smaller
        if i != main and keep[i] and touches(i) and areas[i] > 0.3 * areas[main]:
            cx, cy = cents[i]
            if min(cx, cy, w - cx, h - cy) < 0.15 * min(h, w):
                keep[i] = False
    clean = (keep[labels].astype(np.uint8)) * 255

    ink_fraction = np.count_nonzero(clean) / clean.size
    if ink_fraction < MIN_INK or ink_fraction > MAX_INK:
        return None

    clean = _thicken(clean)

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