"""
Validate the Python OCR port against ground truth on CPU, BEFORE deploying to
Modal. Proves the server pipeline reproduces the on-device engine's label
recall on the 10 test shelves. If this holds ~90%+ raw (the browser then adds
the same normalisation it already applies to local reads, reaching ~95%), the
port is faithful and safe to run server-side.

Usage:
  python validate_local.py [--side=2048] [--only=200004299] [--debug]
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort

import ocr_pipeline as P

MODELS = Path(r"C:\Users\krish\shelfcheck-ocr\models")
IMAGES = Path(r"C:\Users\krish\shelfcheck\test_items")
GT = json.load(open(IMAGES / "ground_truth_v2.json", encoding="utf-8"))

SIDE  = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--side=")), "2048"))
ONLY  = next((a.split("=")[1] for a in sys.argv if a.startswith("--only=")), "")
DEBUG = "--debug" in sys.argv
REC   = next((a.split("=")[1] for a in sys.argv if a.startswith("--rec=")), str(MODELS / "rec.fp32.onnx"))

P.DET_MAX_SIDE = SIDE

# ── strict matcher, same as site_suite.mjs / bench_js_port.py ─────────────────
def num_str(s):
    m = re.match(r"\s*(\d+(?:\.\d+)?)", str(s)); return m.group(1) if m else None
def cutter(s):
    m = re.match(r"\s*\d+(?:\.\d+)?\s*([A-Za-z]{2,})", str(s)); return m.group(1).upper() if m else ""
def surname(s):
    s = str(s).strip().upper()
    m = re.match(r"(VON\s+\w+|VAN\s+\w+|DE\s+\w+|LA\s+\w+)", s)
    if m: return m.group(1).replace(" ", "")
    return re.split(r"[,\s]+", s)[0] or ""
def label_match(det, gl, section):
    if section == "nonfiction":
        dn, gn, dc, gc = num_str(det), num_str(gl), cutter(det), cutter(gl)
        if not (dn and gn and dc and gc): return False
        return (dn.startswith(gn) or gn.startswith(dn)) and dc[:3] == gc[:3]
    ds, gs = surname(det), surname(gl); mn = min(len(ds), len(gs))
    if mn < 3: return ds == gs
    if mn >= 6: return sum(1 for i in range(mn) if ds[i] != gs[i]) <= 1
    return ds[:4] == gs[:4]

import cv2
det_sess = ort.InferenceSession(str(MODELS / "det.onnx"), providers=["CPUExecutionProvider"])
rec_sess = ort.InferenceSession(REC, providers=["CPUExecutionProvider"])
dict_chars = open(MODELS / "rec_dict.txt", encoding="utf-8").read().split("\n")
allowed = P.build_allowed_classes(dict_chars)
print(f"dict {len(dict_chars)} chars, {len(allowed)} allowed classes, det_side={SIDE}, rec={Path(REC).name}", file=sys.stderr)

total_hc = total_match = 0
t_sum = 0.0; n_img = 0
for name, gtv in GT.items():
    if name.startswith("_") or (ONLY and ONLY not in name):
        continue
    section = gtv["section"]
    gt_labels, unc = [], set()
    for r in gtv["rows"]:
        gt_labels += r["labels"]; unc |= set(r.get("uncertain", []))
    hc = [l for l in gt_labels if l not in unc]

    bgr = cv2.imread(str(IMAGES / name))
    t0 = time.time()
    labels, unread = P.run_ocr(bgr, det_sess, rec_sess, dict_chars, allowed)
    dt = time.time() - t0
    t_sum += dt; n_img += 1

    detected = [l["text"] for l in labels]
    used = [False] * len(detected); m = 0; missed = []
    for gl in hc:
        hit = False
        for k, c in enumerate(detected):
            if used[k]: continue
            if label_match(c, gl, section):
                used[k] = True; m += 1; hit = True; break
        if not hit: missed.append(gl)
    total_hc += len(hc); total_match += m
    short = name.replace("PXL_2026060", "").replace(".MP", "").replace(".jpg", "")
    print(f"{short:14} {section:11} {m}/{len(hc)} ({m/len(hc)*100:.0f}%)  "
          f"labels={len(detected)} unread={len(unread)}  {dt:.2f}s")
    if missed: print(f"    missed: {missed[:12]}")
    if DEBUG:
        for l in labels: print(f"      | {l['text']!r}  {l['conf']:.2f}")

print("=" * 70)
print(f"LABEL RECALL: {total_match}/{total_hc} = {total_match/total_hc*100:.1f}%   "
      f"avg {t_sum/n_img:.2f}s/img (CPU)")
