"""
Validate the full Path B pipeline: two-call approach.

Call 1: original image  -> label stickers (horizontal + white-bg in original)
Call 2: rotated image   -> spine titles (horizontal + non-white in rotated, map back)

Then: attach_titles, score {label->title} pairs against ground_truth.json.
"""

import json, time, base64, re, cv2, numpy as np, requests

IMAGE_PATH   = r"C:\Users\krish\shelfcheck\test_items\PXL_20260607_195849222.jpg"
GT_PATH      = r"C:\Users\krish\shelfcheck\test_items\ground_truth.json"
MAX_SIDE     = 1600
WHITE_THRESH = 175
WHITE_RATIO  = 0.55
MIN_BOX_W    = 8
OCRSPACE_KEY = "helloworld"

# Ground truth: {spine_label_upper -> title}
with open(GT_PATH) as f:
    gt_raw = json.load(f)
GT_BOOKS = gt_raw["PXL_20260607_195849222.jpg"]["books"]
GT_BY_LABEL = {b["spine_label"].strip().upper(): b["title"] for b in GT_BOOKS}
print(f"Ground truth: {len(GT_BOOKS)} books, {len(GT_BY_LABEL)} unique labels")
print()

# ---------------------------------------------------------------------------
def load_preprocess(path, max_side=MAX_SIDE):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b2 = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b2]), cv2.COLOR_LAB2BGR)

def encode_jpeg(img, q=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok; return buf.tobytes()

def has_white_bg(img_bgr, x, y, w, h):
    ih, iw = img_bgr.shape[:2]
    x1,y1 = max(0,int(x)), max(0,int(y))
    x2,y2 = min(iw,int(x+w)), min(ih,int(y+h))
    if x2<=x1 or y2<=y1: return False
    gray = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float((gray>=WHITE_THRESH).mean()) >= WHITE_RATIO

def is_horiz(x, y, w, h): return w >= h * 0.75

def call_ocrspace(img_bytes):
    b64 = base64.b64encode(img_bytes).decode()
    t0 = time.time()
    resp = requests.post(
        "https://api.ocr.space/parse/image",
        data={
            "base64Image": f"data:image/jpeg;base64,{b64}",
            "apikey": OCRSPACE_KEY, "language": "eng",
            "OCREngine": "2", "isOverlayRequired": "true",
            "detectOrientation": "false", "scale": "true",
        }, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    if data.get("IsErroredOnProcessing"):
        raise RuntimeError(f"ocr.space error: {data.get('ErrorMessage')}")
    return data, time.time()-t0

def parse_overlay(data):
    lines = []
    for parsed in data.get("ParsedResults", []):
        for line in parsed.get("TextOverlay", {}).get("Lines", []):
            words = line.get("Words", [])
            if not words: continue
            text = " ".join(w.get("WordText","") for w in words).strip()
            if not text: continue
            ls=[w["Left"] for w in words]; ts=[w["Top"] for w in words]
            rs=[w["Left"]+w["Width"] for w in words]; bs=[w["Top"]+w["Height"] for w in words]
            x=min(ls);y=min(ts);w2=max(rs)-x;h2=max(bs)-y
            lines.append({"text": text, "bbox": [x, y, w2, h2], "score": 0.9})
    return lines

def map_back_rotated(rx, ry, rw, rh, orig_h):
    """bbox in rotated (90-CW) image -> bbox in original image"""
    return [ry, max(0, orig_h - rx - rw), rh, rw]

def attach_titles(labels, title_cands):
    if not labels: return
    centers = sorted(l["bbox"][0]+l["bbox"][2]/2 for l in labels)
    gaps = [b-a for a,b in zip(centers,centers[1:]) if b-a>1]
    spacing = float(np.median(gaps)) if gaps else float("inf")
    max_dist = spacing*0.7 if spacing!=float("inf") else float("inf")
    owned = {}
    for tc in title_cands:
        tx,ty,tw,th = tc["bbox"]
        t_cx = tx+tw/2
        best_i,best_dist = None, float("inf")
        for i,lbl in enumerate(labels):
            lx,ly,lw,lh = lbl["bbox"]
            dist = abs((lx+lw/2)-t_cx)
            if dist < best_dist: best_dist,best_i = dist,i
        if best_i is None or best_dist>max_dist: continue
        owned.setdefault(best_i,[]).append(tc)
    for i,frags in owned.items():
        frags.sort(key=lambda f:f["bbox"][1])
        labels[i]["title"]       = " ".join(f["text"] for f in frags)
        labels[i]["title_score"] = min(f["score"] for f in frags)

def score_pairs(labels):
    def words(s): return set(re.sub(r"[^a-z0-9]"," ",s.lower()).split()) - {"the","a","an","and","of","in","with","for"}
    hits, misses, wrong, no_gt = [], [], [], []
    for lbl in labels:
        norm = re.sub(r"\s+"," ", lbl["text"].strip().upper())
        gt_title = GT_BY_LABEL.get(norm)
        if gt_title is None:
            for gk,gv in GT_BY_LABEL.items():
                if norm in gk or gk in norm: gt_title=gv; break
        if gt_title is None: no_gt.append(lbl["text"]); continue
        pred = lbl.get("title")
        if pred is None: misses.append({"label":norm,"gt":gt_title}); continue
        gt_w,pred_w = words(gt_title), words(pred)
        overlap = len(gt_w&pred_w)/max(1,len(gt_w))
        entry = {"label":norm,"gt":gt_title,"pred":pred,"overlap":overlap}
        (hits if overlap>=0.5 else wrong).append(entry)
    return hits, misses, wrong, no_gt

# ---------------------------------------------------------------------------
print("Loading image...")
img = load_preprocess(IMAGE_PATH)
orig_h, orig_w = img.shape[:2]
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
print(f"Original: {orig_w}x{orig_h}  |  Rotated: {rotated.shape[1]}x{rotated.shape[0]}")
orig_bytes = encode_jpeg(img)
rot_bytes  = encode_jpeg(rotated)
print(f"JPEG: orig={len(orig_bytes)//1024}KB  rot={len(rot_bytes)//1024}KB")
print()

# == Call 1: original image -> labels =======================================
print("=" * 60)
print("CALL 1: original image (finding label stickers)")
print("=" * 60)
data1, t1 = call_ocrspace(orig_bytes)
lines1 = parse_overlay(data1)
print(f"Time: {t1:.1f}s  |  {len(lines1)} overlay lines")

labels = []
for ln in lines1:
    x,y,w,h = ln["bbox"]
    if w < MIN_BOX_W: continue
    if not is_horiz(x,y,w,h): continue
    if has_white_bg(img, x, y, w, h):
        labels.append(ln)

print(f"Labels (horizontal + white-bg): {len(labels)}")
print("Label texts found:")
for lbl in labels:
    print(f"  {lbl['text']!r:30s}  bbox={lbl['bbox']}")
print()

# Rate limit sleep
print("Sleeping 65s for rate limit...")
time.sleep(65)

# == Call 2: rotated image -> spine titles ==================================
print()
print("=" * 60)
print("CALL 2: rotated image (finding spine titles)")
print("=" * 60)
data2, t2 = call_ocrspace(rot_bytes)
lines2 = parse_overlay(data2)
print(f"Time: {t2:.1f}s  |  {len(lines2)} overlay lines")

title_cands = []
for ln in lines2:
    rx,ry,rw,rh = ln["bbox"]
    if rw < MIN_BOX_W: continue
    if not is_horiz(rx,ry,rw,rh): continue
    if has_white_bg(rotated, rx,ry,rw,rh): continue
    ob = map_back_rotated(rx,ry,rw,rh, orig_h)
    title_cands.append({**ln, "bbox": ob})

print(f"Title candidates (horizontal + non-white in rotated): {len(title_cands)}")
print()

# == Attach titles ==========================================================
attach_titles(labels, title_cands)
print("label -> title pairs:")
for lbl in labels:
    print(f"  {lbl['text']!r:30s} -> {lbl.get('title','[none]')!r}")
print()

# == Score ==================================================================
hits, misses, wrong, no_gt = score_pairs(labels)
matched = len(hits)+len(misses)+len(wrong)
print("=" * 60)
print(f"SCORE: {len(hits)} correct / {matched} matched-to-GT labels")
print(f"       {len(wrong)} wrong title  |  {len(misses)} no title attached  |  {len(no_gt)} labels not in GT")
print()

if hits:
    print(f"Correct ({len(hits)}):")
    for h in hits:
        print(f"  [{h['overlap']:.0%}] {h['label']!r:20s}  GT={h['gt']!r:35s}  pred={h['pred']!r}")
print()

if wrong:
    print(f"Wrong title ({len(wrong)}):")
    for w in wrong:
        print(f"  {w['label']!r:20s}  GT={w['gt']!r:35s}  pred={w['pred']!r}  overlap={w['overlap']:.0%}")
print()

if misses:
    print(f"No title attached ({len(misses)}):")
    for m in misses:
        print(f"  {m['label']!r:20s}  GT={m['gt']!r}")
print()

# also show raw GT coverage: how many GT books got their label found at all?
all_found_labels = set(re.sub(r"\s+"," ",lbl["text"].strip().upper()) for lbl in labels)
gt_found = [gt for gt in GT_BY_LABEL if gt in all_found_labels or
            any(gt in f or f in gt for f in all_found_labels)]
print(f"GT label coverage: {len(gt_found)}/{len(GT_BY_LABEL)} GT labels recognized")
