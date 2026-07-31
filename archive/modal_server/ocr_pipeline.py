"""
Faithful Python port of the browser on-device OCR pipeline (ocr_local.js run()).

Same models (models/det.onnx + models/rec.fp32.onnx), same constants, same
det postprocess (thresholded connected components + height-fraction dilation),
same sticker-size gate, same rec preprocessing (grayscale on a white-padded
48px strip with a light-background gate), the same charset-masked CTC decode
(≈139 English-relevant classes), and the same line-join + dedupe.

Returns labels in the EXACT shape LocalOCR.run() returns to the app:
    labels:  [{ "text": str, "conf": float, "bboxFrac": [x, y, w, h] }]
    unread:  [{ "bboxFrac": [x, y, w, h] }]   # detected sticker, not recognised
bboxFrac values are fractions of the source image, so the browser's fusion,
ordering, and yellow-unread drawing consume them identically to the local
engine — nothing downstream changes.

Kept deliberately dependency-light (numpy + cv2 + onnxruntime) so the same file
runs locally on CPU for validation and on a Modal GPU container unchanged.
"""
import numpy as np
import cv2

# ── Constants — copied verbatim from ocr_local.js ─────────────────────────────
DET_MAX_SIDE      = 2048
DET_THRESH        = 0.3
DET_BOX_MIN_PROB  = 0.5
DET_DILATE        = 0.55     # box expansion, fraction of component height
REC_HEIGHT        = 48
REC_MAX_WIDTH     = 640
REC_MIN_CONF      = 0.5
STICK_H_MIN       = 0.004    # of max(imgW, imgH)
STICK_H_MAX       = 0.032
REC_WIDTH_BUCKETS = [64, 96, 128, 160, 224, 320, 480, 640]

_PUNCT = set(list(".,'-&/:;!?()\" "))


def build_allowed_classes(dict_chars):
    """Mirror ocr_local.js buildCharsetMask: class 0 (blank), the English/Latin
    dict classes, and the trailing space class."""
    def ok(ch):
        if not ch or len(ch) != 1:
            return False
        c = ord(ch)
        if 48 <= c <= 57:   return True   # 0-9
        if 65 <= c <= 90:   return True   # A-Z
        if 97 <= c <= 122:  return True   # a-z
        if ch in _PUNCT:    return True
        if 0xC0 <= c <= 0xFF and c not in (0xD7, 0xF7):
            return True                    # accented Latin (À..ÿ minus × ÷)
        return False
    allowed = [0]
    for i, ch in enumerate(dict_chars):
        if ok(ch):
            allowed.append(i + 1)          # class i>0 -> dict_chars[i-1]
    allowed.append(len(dict_chars) + 1)    # trailing space class
    return allowed


# ── Detection ─────────────────────────────────────────────────────────────────
def _det_preprocess(rgb):
    H, W = rgb.shape[:2]
    sc = min(1.0, DET_MAX_SIDE / max(H, W))
    nw = max(32, (round(W * sc) >> 5) << 5)
    nh = max(32, (round(H * sc) >> 5) << 5)
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    data = (small / 127.5 - 1.0).transpose(2, 0, 1)[None]   # [1,3,nh,nw] RGB
    return np.ascontiguousarray(data), nw, nh


def _det_postprocess(prob, nw, nh, srcW, srcH):
    """Thresholded connected components with per-component max-prob gate and
    height-fraction dilation — same output as the JS union-find version."""
    bmap = (prob > DET_THRESH).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bmap, connectivity=4)
    sx, sy = srcW / nw, srcH / nh
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 12 or w < 4 or h < 2:
            continue
        region = prob[y:y + h, x:x + w]
        mask = lab[y:y + h, x:x + w] == i
        if region[mask].max() < DET_BOX_MIN_PROB:
            continue
        ex = h * DET_DILATE
        boxes.append([(x - ex) * sx, (y - ex) * sy, (w + 2 * ex) * sx, (h + 2 * ex) * sy])
    return boxes


def _sticker_filter(boxes, W, H):
    m = max(W, H)
    return [b for b in boxes
            if STICK_H_MIN * m <= b[3] <= STICK_H_MAX * m and b[2] >= b[3] * 0.5]


# ── Recognition ───────────────────────────────────────────────────────────────
def _rec_preprocess(rgb, box):
    bx, by, bw, bh = box
    x = max(0, round(bx)); y = max(0, round(by))
    w = min(rgb.shape[1] - x, round(bw)); h = min(rgb.shape[0] - y, round(bh))
    if w < 4 or h < 6:
        return None
    nat = min(REC_MAX_WIDTH, max(16, round(w * REC_HEIGHT / h)))
    tw = next((b for b in REC_WIDTH_BUCKETS if b >= nat), REC_MAX_WIDTH)
    crop = rgb[y:y + h, x:x + w]
    resized = cv2.resize(crop, (nat, REC_HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32)
    lum = 0.299 * resized[:, :, 0] + 0.587 * resized[:, :, 1] + 0.114 * resized[:, :, 2]
    # Light-background gate: sticker text sits on a white label.
    light_frac = float((lum >= 120).sum()) / (nat * REC_HEIGHT)
    if light_frac < 0.30:
        return None
    strip = np.full((REC_HEIGHT, tw), 255.0, np.float32)   # white pad
    strip[:, :nat] = lum
    g = strip / 127.5 - 1.0
    return np.stack([g, g, g])[None], tw   # [1,3,48,tw]


def _ctc_decode(probs, allowed, dict_chars):
    """probs: (T, C) recogniser output (already softmaxed by the model)."""
    T, C = probs.shape
    allowed = [c for c in allowed if c < C]
    sub = probs[:, allowed]
    best_idx = sub.argmax(1)
    best_p = sub.max(1)
    chars, conf_sum, prev = [], 0.0, 0
    nd = len(dict_chars)
    for t in range(T):
        cls = allowed[best_idx[t]]
        if cls != 0 and cls != prev:
            ci = cls - 1
            chars.append(dict_chars[ci] if ci < nd else ' ')
            conf_sum += best_p[t]
        prev = cls
    text = ''.join(chars).strip()
    return text, (conf_sum / len(chars) if chars else 0.0)


# ── Line join + dedupe (ocr_local.js joinLines / dedupeLabels) ────────────────
def _join_lines(recs):
    n = len(recs)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(n):
        bi = recs[i]['box']
        for j in range(n):
            if i == j:
                continue
            bj = recs[j]['box']
            hmax = max(bi[3], bj[3])
            dy = (bj[1] + bj[3] / 2) - (bi[1] + bi[3] / 2)
            xov = min(bi[0] + bi[2], bj[0] + bj[2]) - max(bi[0], bj[0])
            if 0.4 * hmax < dy < 2.4 * hmax and xov > 0.4 * min(bi[2], bj[2]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(recs[i])
    labels = []
    for g in groups.values():
        g.sort(key=lambda r: r['box'][1] + r['box'][3] / 2)
        text = ' '.join(r['text'] for r in g)
        text = ' '.join(text.split())
        if len(text) < 2:
            continue
        x0 = min(r['box'][0] for r in g); y0 = min(r['box'][1] for r in g)
        x1 = max(r['box'][0] + r['box'][2] for r in g); y1 = max(r['box'][1] + r['box'][3] for r in g)
        conf = sum(r['conf'] for r in g) / len(g)
        labels.append({'text': text, 'conf': conf, 'box': [x0, y0, x1 - x0, y1 - y0]})
    return _dedupe(labels)


def _dedupe(labels):
    import re
    def norm(s):
        return re.sub(r'[^A-Za-z0-9]', '', s).upper()
    def xov(a, b):
        return min(a['box'][0] + a['box'][2], b['box'][0] + b['box'][2]) - max(a['box'][0], b['box'][0])
    def yov(a, b):
        return min(a['box'][1] + a['box'][3], b['box'][1] + b['box'][3]) - max(a['box'][1], b['box'][1])
    keep = sorted(labels, key=lambda l: -l['conf'])
    dead = set()
    for i in range(len(keep)):
        if i in dead:
            continue
        ni = norm(keep[i]['text'])
        for j in range(i + 1, len(keep)):
            if j in dead:
                continue
            if xov(keep[i], keep[j]) < 0.5 * min(keep[i]['box'][2], keep[j]['box'][2]):
                continue
            if yov(keep[i], keep[j]) < 0.4 * min(keep[i]['box'][3], keep[j]['box'][3]):
                continue
            nj = norm(keep[j]['text'])
            pre = min(len(ni), len(nj), 6)
            if pre >= 4 and ni[:pre] == nj[:pre]:
                dead.add(j)
    out = [keep[i] for i in range(len(keep)) if i not in dead]
    final = []
    for l in out:
        spanned = 0
        for o in out:
            if o is l or o['box'][2] >= l['box'][2]:
                continue
            if xov(l, o) > 0.6 * o['box'][2] and yov(l, o) > 0.5 * o['box'][3]:
                spanned += 1
        if spanned < 2:
            final.append(l)
    return final


# ── Public entry point ────────────────────────────────────────────────────────
def run_ocr(bgr, det_sess, rec_sess, dict_chars, allowed=None, rec_batch=16):
    """bgr: an OpenCV image (H,W,3). Returns (labels, unread) in browser shape."""
    if allowed is None:
        allowed = build_allowed_classes(dict_chars)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    det_in = det_sess.get_inputs()[0].name
    data, nw, nh = _det_preprocess(rgb)
    prob = det_sess.run(None, {det_in: data})[0][0, 0]        # (nh, nw)
    boxes = _det_postprocess(prob, nw, nh, W, H)
    stick = _sticker_filter(boxes, W, H)

    # Preprocess every sticker crop, grouped by padded width for batched rec.
    rec_in = rec_sess.get_inputs()[0].name
    by_w = {}
    for b in stick:
        pre = _rec_preprocess(rgb, b)
        if pre is None:
            continue
        tensor, tw = pre
        by_w.setdefault(tw, []).append((b, tensor))

    recs, read_boxes = [], []
    for tw, items in by_w.items():
        for s0 in range(0, len(items), rec_batch):
            chunk = items[s0:s0 + rec_batch]
            batch = np.concatenate([t for _, t in chunk], axis=0)
            out = rec_sess.run(None, {rec_in: batch})[0]        # (B, T, C)
            for k, (box, _) in enumerate(chunk):
                text, conf = _ctc_decode(out[k], allowed, dict_chars)
                if text and conf > REC_MIN_CONF:
                    recs.append({'text': text, 'conf': conf, 'box': box})
                    read_boxes.append(box)

    labels = _join_lines(recs)
    out_labels = [{'text': l['text'], 'conf': float(l['conf']),
                   'bboxFrac': [l['box'][0] / W, l['box'][1] / H, l['box'][2] / W, l['box'][3] / H]}
                  for l in labels]

    read_set = {id(b) for b in read_boxes}
    unread = [{'bboxFrac': [b[0] / W, b[1] / H, b[2] / W, b[3] / H]}
              for b in stick if id(b) not in read_set]
    return out_labels, unread
