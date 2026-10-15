import cv2
import numpy as np


def find_character(image, pad_ratio=0.35):
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    small = (cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
             if scale < 1 else image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    sh, sw = gray.shape

    k = (min(sh, sw) // 2) | 1
    flat = cv2.divide(gray, cv2.GaussianBlur(gray, (k, k), 0), scale=255)
    flat = cv2.GaussianBlur(flat, (5, 5), 0)
    _, ink = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    grow = max(3, int(0.05 * min(sh, sw))) | 1
    grouped = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)

    best, best_score = None, 0.0
    for i in range(1, count):
        x, y, bw, bh, _ = stats[i]
        if x <= 0 or y <= 0 or x + bw >= sw or y + bh >= sh:
            continue            
        if bw * bh < 0.003 * sh * sw:
            continue
        dist = np.hypot((x + bw / 2) / sw - 0.5, (y + bh / 2) / sh - 0.5)
        score = bw * bh * (1.0 - dist)   
        if score > best_score:
            best, best_score = (x, y, bw, bh), score
    if best is None:
        return image

    x, y, bw, bh = [int(v / scale) for v in best]
    half = max(bw, bh) // 2 + int(pad_ratio * max(bw, bh))
    cx, cy = x + bw // 2, y + bh // 2
    return image[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]