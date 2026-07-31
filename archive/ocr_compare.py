"""
OCR engine comparison for spine title reading.
Test image: PXL_20260607_195849222.jpg (Jewish/holiday cooking nonfiction shelf, ~48 books)

Engines tested:
  1. RapidOCR v3 (rapidocr-onnxruntime 1.2.3  — PP-OCRv3, current server baseline)
  2. RapidOCR v6 (rapidocr 3.9.0              — PP-OCRv6, newest PaddleOCR)
  3. EasyOCR 1.7.2
  4. ocr.space (Engine 2, free 'helloworld' key)

Strategy for all engines:
  - Rotate image 90° CW so vertical spine text becomes horizontal (same as app.py Pass 2)
  - Collect all text regions, filter out white-background areas (= sticker labels)
  - Report what titles were found and score against ground truth
"""

import sys
import time
import base64
import re
import cv2
import numpy as np
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGE_PATH = r"C:\Users\krish\shelfcheck\test_items\PXL_20260607_195849222.jpg"
MAX_SIDE   = 1600
MIN_SCORE  = 0.30
WHITE_THRESH = 175
WHITE_RATIO  = 0.55

GROUND_TRUTH = [
    "SECOND GENERATION", "MODERN JEWISH BAKER", "Pomegranates & Artichokes",
    "THE INSTANT POT KOSHER COOKBOOK", "THE COOK and THE RABBI", "NOSH",
    "Eat Jewish", "MICHAEL W. TWITTY KOSHER", "THE ARTISAN JEWISH DELI",
    "FEAST", "Rooza", "ALL-TIME BEST HOLIDAY ENTERTAINING", "TRICKY TREATS",
    "Food Gift Love", "CELEBRATE with BABS", "The Unofficial Disney Parks HOLIDAYS COOKBOOK",
    "Weeknight WONDERS", "VEGAN HOLIDAY COOKBOOK", "The Ultimate Minnesota Cookie Book",
    "A YEAR OF HOLIDAYS", "Ghost Food", "SPOOKY FOOD", "BEST HOLIDAY SWEETS & TREATS",
    "SHARE", "SET FOR THE HOLIDAYS", "Entertaining in the Country", "FOOD&WINE Potluck",
    "EVERYDAY CELEBRATIONS", "halloween treats", "HOW TO CELEBRATE EVERYTHING",
    "ARTY PARTIES", "SAM SIFTON THANKSGIVING", "FRIENDSGIVING",
    "2019 HOLIDAY & CELEBRATIONS COOKBOOK", "2020 HOLIDAY & CELEBRATIONS COOKBOOK",
    "2021 HOLIDAY & CELEBRATIONS COOKBOOK", "FAMILY STYLE", "Celebrations",
    "COOKING for FRIENDS", "The Best of THANKSGIVING", "THE EASY CHRISTMAS COOKIE COOKBOOK",
    "THE COZY CHRISTMAS MOVIE",
]

# ── Image helpers ─────────────────────────────────────────────────────────────

def load_image(path, max_side=MAX_SIDE):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def pts_bbox(box):
    a = np.array(box, dtype=np.float32)
    x1, y1 = a[:, 0].min(), a[:, 1].min()
    return float(x1), float(y1), float(a[:, 0].max() - x1), float(a[:, 1].max() - y1)

def is_white_bg(img_bgr, box):
    x, y, w, h = [int(v) for v in pts_bbox(box)]
    ih, iw = img_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    if x2 <= x1 or y2 <= y1:
        return False
    gray = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float((gray >= WHITE_THRESH).mean()) >= WHITE_RATIO

def is_horizontal_box(box):
    _, _, w, h = pts_bbox(box)
    return w >= h * 0.75

def safe(s):
    return s.encode("ascii", errors="replace").decode("ascii")

# ── Scoring ───────────────────────────────────────────────────────────────────

def word_set(s):
    return set(re.sub(r"[^a-z0-9]", " ", s.lower()).split())

def score_against_gt(found_texts):
    """
    For each ground truth title, check if ANY found text fragment contains
    enough of its key words to be recognisable (>= 60% word overlap, min 2 words matched).
    Returns list of (gt_title, matched_fragment_or_None).
    """
    results = []
    for gt in GROUND_TRUTH:
        gt_words = word_set(gt)
        gt_words -= {"the", "a", "an", "and", "of", "in", "with", "for", "&"}
        if not gt_words:
            gt_words = word_set(gt)
        best_match, best_ratio = None, 0.0
        for text in found_texts:
            tw = word_set(text)
            overlap = len(gt_words & tw)
            if overlap == 0:
                continue
            ratio = overlap / len(gt_words)
            if overlap >= 2 and ratio >= best_ratio:
                best_ratio = ratio
                best_match = text
            elif overlap == 1 and len(gt_words) == 1 and ratio >= best_ratio:
                best_ratio = ratio
                best_match = text
        results.append((gt, best_match, best_ratio))
    return results

# ── Engine 1: RapidOCR v3 (rapidocr-onnxruntime) ─────────────────────────────

def run_rapidocr_v3(img):
    from rapidocr_onnxruntime import RapidOCR as RapidOCRv3
    engine = RapidOCRv3()
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    result, _ = engine(rotated, use_cls=False)
    texts = []
    for box, text, score in (result or []):
        if not text or float(score) < MIN_SCORE:
            continue
        if not is_horizontal_box(box):
            continue
        if is_white_bg(rotated, box):
            continue
        texts.append(text.strip())
    return texts

# ── Engine 2: RapidOCR v6 (rapidocr package, PP-OCRv6) ───────────────────────

def run_rapidocr_v6(img):
    import logging
    logging.disable(logging.CRITICAL)  # suppress info spam
    from rapidocr import RapidOCR as RapidOCRv6
    engine = RapidOCRv6()
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    result = engine(rotated, use_det=True, use_cls=False, use_rec=True)
    texts = []
    if result.boxes is None or result.txts is None:
        return texts
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        if not text or float(score) < MIN_SCORE:
            continue
        if not is_horizontal_box(box):
            continue
        if is_white_bg(rotated, box):
            continue
        texts.append(text.strip())
    return texts

# ── Engine 3: EasyOCR ─────────────────────────────────────────────────────────

def run_easyocr(img):
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    result = reader.readtext(rotated, detail=1, paragraph=False)
    texts = []
    for bbox, text, conf in result:
        if not text or float(conf) < MIN_SCORE:
            continue
        if not is_horizontal_box(bbox):
            continue
        if is_white_bg(rotated, bbox):
            continue
        texts.append(text.strip())
    return texts

# ── Engine 4: ocr.space ───────────────────────────────────────────────────────

def run_ocrspace(img):
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    ok, buf = cv2.imencode(".jpg", rotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ["[encoding failed]"]
    b64 = base64.b64encode(buf.tobytes()).decode()
    try:
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            data={
                "base64Image": f"data:image/jpeg;base64,{b64}",
                "apikey": "helloworld",
                "language": "eng",
                "scale": True,
                "isOverlayRequired": True,
                "detectOrientation": False,
                "OCREngine": 2,
            },
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [f"[API error: {e}]"]

    if data.get("IsErroredOnProcessing"):
        msgs = data.get("ErrorMessage", ["unknown"])
        return [f"[API error: {msgs}]"]

    # Extract line-level text from the overlay
    texts = []
    for parsed in data.get("ParsedResults", []):
        lines = parsed.get("TextOverlay", {}).get("Lines", [])
        for line in lines:
            words = [w.get("WordText", "") for w in line.get("Words", [])]
            text = " ".join(w for w in words if w).strip()
            if text:
                texts.append(text)
        # Fallback: plain text split by newline
        if not texts:
            raw = parsed.get("ParsedText", "")
            texts = [l.strip() for l in raw.split("\r\n") if l.strip()]
    return texts

# ── Main ──────────────────────────────────────────────────────────────────────

ENGINES = [
    ("RapidOCR v3 (PP-OCRv3, current server)", run_rapidocr_v3),
    ("RapidOCR v6 (PP-OCRv6, newest)",          run_rapidocr_v6),
    ("EasyOCR 1.7.2",                            run_easyocr),
    ("ocr.space Engine 2",                       run_ocrspace),
]

def print_header(s, ch="="):
    print(ch * 62)
    print(s)
    print(ch * 62)

if __name__ == "__main__":
    print_header("Loading & preprocessing image")
    img = load_image(IMAGE_PATH)
    print(f"Loaded: {img.shape[1]}x{img.shape[0]}")
    print(f"Ground truth: {len(GROUND_TRUTH)} titles")
    print()

    for engine_name, fn in ENGINES:
        print_header(f"ENGINE: {engine_name}")
        t0 = time.time()
        try:
            found = fn(img)
        except Exception as e:
            print(f"CRASHED: {e}")
            print()
            continue
        elapsed = time.time() - t0

        print(f"Time: {elapsed:.1f}s  |  {len(found)} text regions after title filtering")
        print()

        print("--- All title-region text found ---")
        for t in found:
            print(f"  {safe(t)}")
        print()

        scored = score_against_gt(found)
        hits   = [(gt, frag, r) for gt, frag, r in scored if frag is not None]
        misses = [gt for gt, frag, r in scored if frag is None]

        print(f"--- Ground truth matched: {len(hits)}/{len(GROUND_TRUTH)} ---")
        for gt, frag, ratio in sorted(hits, key=lambda x: -x[2]):
            print(f"  [{ratio:.0%}] GT: {safe(gt)!r:45s}  OCR: {safe(frag)!r}")
        print()
        if misses:
            print(f"--- Missed ({len(misses)}) ---")
            for gt in misses:
                print(f"  {safe(gt)}")
        print()
