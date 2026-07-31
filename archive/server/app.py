"""
ShelfCheck OCR v2 — FastAPI + RapidOCR with label-sticker filtering.

POST /ocr → {labels:[{text,score,bbox:[x,y,w,h],quad,shelfRow,title,title_score}], ms, imageSize, n}

Filtering pipeline:
  Pass 1 — call-number stickers
    1. Resize to MAX_SIDE, CLAHE contrast boost
    2. RapidOCR (no angle classifier — stickers are always horizontal)
    3. Keep: score ≥ MIN_SCORE, horizontal region, white background
    4. Sort top→bottom, left→right; assign shelfRow by y-centroid

  Pass 2 — spine titles
    5. Rotate image 90° CW (bottom-to-top spine text becomes left-to-right)
    6. RapidOCR on rotated image
    7. Keep: horizontal region in rotated space, NOT white background
    8. Map coordinates back to original image space
    9. Attach nearest title to each sticker label (matched by x-center)

Env vars (all optional):
  SHELFCHECK_API_KEY   if set, requests must send X-Api-Key header
  ALLOWED_ORIGINS      comma-separated CORS origins (default *)
  MAX_SIDE             downscale longest edge before OCR (default 1600)
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


def _attach_titles(candidates, title_candidates):
    """Assign spine-title fragments to call-number stickers.

    Each title fragment belongs to exactly ONE sticker — the one whose x-center
    is nearest — so a single loud title can never be stamped onto every neighbour.
    A sticker then concatenates all the fragments it owns, ordered top-to-bottom
    (the spine's natural reading order), into one title string.

    Fragments further than half the typical column spacing from any sticker are
    dropped, so stray text far from a spine doesn't get glued onto a title.
    """
    if not candidates:
        return

    centers = sorted(c["bbox"][0] + c["bbox"][2] / 2 for c in candidates)
    if len(centers) >= 2:
        gaps = [b - a for a, b in zip(centers, centers[1:]) if b - a > 1]
        spacing = float(np.median(gaps)) if gaps else float("inf")
    else:
        spacing = float("inf")
    # A fragment must sit within ~0.7 of a column of its sticker's x-center.
    # (Assignment is exclusive nearest-sticker, so a looser bound only recovers
    #  coverage — it can never stamp one title onto multiple spines.)
    max_dist = spacing * 0.7 if spacing != float("inf") else float("inf")

    owned: dict[int, list] = {}
    for tc in title_candidates:
        tx, ty, tw, th = tc["bbox"]
        t_cx = tx + tw / 2
        best_i, best_dist = None, float("inf")
        for i, c in enumerate(candidates):
            lx, ly, lw, lh = c["bbox"]
            dist = abs((lx + lw / 2) - t_cx)
            if dist < best_dist:
                best_dist, best_i = dist, i
        if best_i is None or best_dist > max_dist:
            continue
        owned.setdefault(best_i, []).append(tc)

    for i, frags in owned.items():
        frags.sort(key=lambda f: f["bbox"][1])  # top of spine first
        candidates[i]["title"] = " ".join(f["text"] for f in frags)
        candidates[i]["title_score"] = round(
            min(f["score"] for f in frags), 3
        )


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

    # Resize so the longest edge = MAX_SIDE
    scale = 1.0
    if max(orig_h, orig_w) > MAX_SIDE:
        scale = MAX_SIDE / max(orig_h, orig_w)
        img = cv2.resize(
            img,
            (round(orig_w * scale), round(orig_h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    # CLAHE contrast equalisation in LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = _clahe.apply(l_ch)
    img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    scaled_h = img.shape[0]  # height of scaled image, needed for title coord mapping
    inv = 1.0 / scale

    t0 = time.time()

    # ── Pass 1: call-number sticker labels (white + horizontal) ─────────────
    result1, _ = engine(img, use_cls=False)

    candidates = []
    for box, text, score in result1 or []:
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
            "bbox":  [round(x * inv), round(y * inv), round(w * inv), round(h * inv)],
            "quad":  [[round(float(p[0]) * inv), round(float(p[1]) * inv)] for p in box],
            "title":       None,
            "title_score": None,
        })

    # ── Pass 2: spine titles (rotate 90° CW, non-white + horizontal) ────────
    #
    # cv2.ROTATE_90_CLOCKWISE maps original pixel (x, y) → rotated (scaled_h-1-y, x).
    # Inverse: rotated (rx, ry) → original (ry, scaled_h-1-rx).
    # For a bbox (rx, ry, rw, rh) in rotated space the original bbox is:
    #   orig_x = ry,  orig_y = scaled_h-1-rx-rw,  orig_w = rh,  orig_h = rw
    #
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    result2, _ = engine(rotated, use_cls=False)

    title_candidates = []
    for box, text, score in result2 or []:
        text = (text or "").strip()
        if not text or float(score) < MIN_SCORE:
            continue
        if not _is_horizontal(box):          # horizontal in rotated space = was vertical
            continue
        if _has_white_background(rotated, box):  # skip regions that are sticker labels
            continue
        rx, ry, rw, rh = _pts_bbox(box)
        orig_x = ry
        orig_y = max(0.0, scaled_h - 1 - rx - rw)
        orig_w = rh
        orig_h = rw
        title_candidates.append({
            "text":  text,
            "score": round(float(score), 3),
            "bbox":  [round(orig_x * inv), round(orig_y * inv),
                      round(orig_w * inv), round(orig_h * inv)],
        })

    ms = int((time.time() - t0) * 1000)

    # ── Match titles to sticker labels (one fragment → one nearest sticker) ──
    _attach_titles(candidates, title_candidates)

    # ── Sort + assign shelfRow ───────────────────────────────────────────────
    row_h = max(1, round(orig_h * 0.10))
    for c in candidates:
        bx, by, bw, bh = c["bbox"]
        c["_row"] = (by + bh // 2) // row_h

    candidates.sort(key=lambda c: (c["_row"], c["bbox"][0]))

    bucket_map: dict[int, int] = {}
    for c in candidates:
        b = c["_row"]
        if b not in bucket_map:
            bucket_map[b] = len(bucket_map)

    labels = [
        {
            "text":        c["text"],
            "score":       c["score"],
            "bbox":        c["bbox"],
            "quad":        c["quad"],
            "shelfRow":    bucket_map[c["_row"]],
            "title":       c["title"],
            "title_score": c["title_score"],
        }
        for c in candidates
    ]

    return {
        "labels":    labels,
        "ms":        ms,
        "imageSize": [orig_w, orig_h],
        "n":         len(labels),
    }
