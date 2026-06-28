"""
ShelfCheck OCR v2 — FastAPI + RapidOCR with label-sticker filtering.

POST /ocr → {labels:[{text,score,bbox:[x,y,w,h],quad,shelfRow}], ms, imageSize, n}

Filtering pipeline (fixes the grouping failure of v1):
  1. Resize to MAX_SIDE — keeps sticker detail without slow inference on huge images
  2. CLAHE contrast boost — evens out shelf lighting before OCR
  3. RapidOCR (det + rec, no angle classifier — stickers are always horizontal)
  4. Drop results below MIN_SCORE confidence
  5. Drop non-horizontal regions (spine titles printed vertically)
  6. Drop regions whose background isn't mostly white (library stickers are white)
  7. Sort top→bottom, left→right; assign shelfRow from y-centroid clustering

Env vars (all optional):
  SHELFCHECK_API_KEY   if set, requests must send X-Api-Key header
  ALLOWED_ORIGINS      comma-separated CORS origins (default *)
  MAX_SIDE             downscale longest edge to this before OCR (default 1600)
  MIN_SCORE            minimum OCR confidence to keep (default 0.4)
  WHITE_THRESH         pixel brightness floor for "white" (0-255, default 175)
  WHITE_RATIO          fraction of pixels that must be white (default 0.55)
"""
import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from rapidocr_onnxruntime import RapidOCR

API_KEY         = os.environ.get("SHELFCHECK_API_KEY")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
MAX_SIDE        = int(os.environ.get("MAX_SIDE",     "1600"))
MIN_SCORE       = float(os.environ.get("MIN_SCORE",  "0.4"))
WHITE_THRESH    = int(os.environ.get("WHITE_THRESH", "175"))
WHITE_RATIO     = float(os.environ.get("WHITE_RATIO","0.55"))

app = FastAPI(title="ShelfCheck OCR v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RapidOCR()
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ── geometry helpers ────────────────────────────────────────────────────────

def _pts_bbox(box):
    a = np.array(box, dtype=np.float32)
    x1, y1 = a[:, 0].min(), a[:, 1].min()
    return float(x1), float(y1), float(a[:, 0].max() - x1), float(a[:, 1].max() - y1)


def _is_horizontal(box):
    _, _, w, h = _pts_bbox(box)
    return w >= h * 0.75


def _has_white_background(img_bgr, box):
    x, y, w, h = _pts_bbox(box)
    ih, iw = img_bgr.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return False
    gray = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float((gray >= WHITE_THRESH).mean()) >= WHITE_RATIO


# ── endpoints ───────────────────────────────────────────────────────────────

@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "service": "shelfcheck-ocr-v2"}


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    raw = await file.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image too large (max 25 MB)")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode image")

    orig_h, orig_w = img.shape[:2]

    # Resize so the longest edge = MAX_SIDE (keeps OCR fast without losing sticker detail)
    scale = 1.0
    if max(orig_h, orig_w) > MAX_SIDE:
        scale = MAX_SIDE / max(orig_h, orig_w)
        img = cv2.resize(
            img,
            (round(orig_w * scale), round(orig_h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    # CLAHE contrast equalisation — helps with uneven shelf lighting
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = _clahe.apply(l_ch)
    img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    t0 = time.time()
    # use_cls=False: stickers are horizontal — angle classifier adds latency with no gain
    result, _ = engine(img, use_cls=False)
    ms = int((time.time() - t0) * 1000)

    inv = 1.0 / scale
    candidates = []

    for box, text, score in result or []:
        text = (text or "").strip()
        if not text or float(score) < MIN_SCORE:
            continue
        if not _is_horizontal(box):
            continue
        if not _has_white_background(img, box):
            continue

        x, y, w, h = _pts_bbox(box)
        candidates.append({
            "text":  text,
            "score": round(float(score), 3),
            # Return coords in original-image pixel space (undo the resize)
            "bbox":  [round(x * inv), round(y * inv), round(w * inv), round(h * inv)],
            "quad":  [[round(float(p[0]) * inv), round(float(p[1]) * inv)] for p in box],
        })

    # Assign shelf rows by y-centroid: bucket = 10% of original image height
    row_h = max(1, round(orig_h * 0.10))
    for c in candidates:
        bx, by, bw, bh = c["bbox"]
        c["_row"] = (by + bh // 2) // row_h

    # Sort: top shelf first, then left-to-right within each shelf
    candidates.sort(key=lambda c: (c["_row"], c["bbox"][0]))

    # Normalise row buckets → 0, 1, 2, … (skipped buckets due to gaps collapse away)
    bucket_map: dict[int, int] = {}
    for c in candidates:
        b = c["_row"]
        if b not in bucket_map:
            bucket_map[b] = len(bucket_map)

    labels = [
        {
            "text":     c["text"],
            "score":    c["score"],
            "bbox":     c["bbox"],
            "quad":     c["quad"],
            "shelfRow": bucket_map[c["_row"]],
        }
        for c in candidates
    ]

    return {
        "labels":    labels,
        "ms":        ms,
        "imageSize": [orig_w, orig_h],
        "n":         len(labels),
    }
