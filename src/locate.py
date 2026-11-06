import cv2
import numpy as np

from src.preprocess import drop_solid_blobs, ensure_min_size, flatten, to_gray

MIN_BOX = 0.0005       
MIN_MASS = 30          
EDGE_PENALTY = 0.4     
INK_RATIO = 0.72       


def _best_box(ink, sh, sw):
    grow = max(3, int(0.05 * min(sh, sw))) | 1
    grouped = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    mass = np.bincount(labels[ink > 0], minlength=count)

    best, best_score = None, 0.0
    for i in range(1, count):
        x, y, bw, bh, _ = stats[i]
        if bw * bh < MIN_BOX * sh * sw or mass[i] < MIN_MASS:
            continue
        frame = bw > 0.8 * sw and bh > 0.8 * sh and mass[i] < 0.3 * bw * bh
        strip = (bh > 0.8 * sh and bw < 0.1 * sw) or (bw > 0.8 * sw and bh < 0.1 * sh)
        if frame or strip:
            continue
        dist = np.hypot((x + bw / 2) / sw - 0.5, (y + bh / 2) / sh - 0.5)
        score = mass[i] * (1.0 - dist) ** 2
        if x <= 0 or y <= 0 or x + bw >= sw or y + bh >= sh:
            score *= EDGE_PENALTY
        if score > best_score:
            best, best_score = (x, y, bw, bh), score
    return best


def find_character(image, pad_ratio=0.35):
    if image is None or image.size == 0:
        return image
    image = ensure_min_size(image)
    h, w = image.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    small = (cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
             if scale < 1 else image)
    gray = to_gray(small)
    sh, sw = gray.shape

    flat = cv2.GaussianBlur(flatten(gray), (5, 5), 0)
    _, otsu = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fixed = ((flat < INK_RATIO * 255) * 255).astype(np.uint8)

    best = None
    for ink in (fixed, otsu):        
        best = _best_box(drop_solid_blobs(ink.copy()), sh, sw)
        if best is not None:
            break
    if best is None:
        return image

    x, y, bw, bh = [int(v / scale) for v in best]
    half = max(bw, bh) // 2 + int(pad_ratio * max(bw, bh))
    cx, cy = x + bw // 2, y + bh // 2
    return image[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]