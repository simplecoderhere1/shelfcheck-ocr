"""
Compare recognition models for the on-device engine: recall + speed on the
hardest shelves. Same det model and same pipeline for all — only the rec model
(and its embedded dict) changes — so any difference is the rec model alone.

Goal: find a SMALLER/faster rec model that still holds >=90% on hard images,
which would cut the on-device scan time toward 5s with no server.

Usage: python bench_rec_models.py
"""
import re, time, sys
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
import ocr_pipeline as P

MODELS = Path(r"C:\Users\krish\shelfcheck-ocr\models")
IMAGES = Path(r"C:\Users\krish\shelfcheck\test_items")
import json
GT = json.load(open(IMAGES / "ground_truth_v2.json", encoding="utf-8"))

# The three hardest shelves (lowest raw recall in validate_local): two dense
# nonfiction, one partial fiction. --all runs the full set.
HARD = ["PXL_20260607_195858362.jpg", "PXL_20260607_195911082.jpg", "PXL_20260607_200006014.jpg"]
if "--all" in sys.argv:
    HARD = [k for k in GT if not k.startswith("_")]

# rec models to compare: (label, path, size). Add more here as we find them.
REC_MODELS = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--rec=")]
CANDIDATES = [("app v6 (21MB)", str(MODELS / "rec.fp32.onnx"))]
import glob, os
try:
    import rapidocr_onnxruntime as R
    rbase = os.path.dirname(R.__file__)
    for m in sorted(glob.glob(os.path.join(rbase, "models", "*rec*.onnx"))):
        CANDIDATES.append((f"{os.path.basename(m)[:18]} ({os.path.getsize(m)/1e6:.0f}MB)", m))
except Exception:
    pass
for extra in REC_MODELS:
    CANDIDATES.append((Path(extra).name + f" ({os.path.getsize(extra)/1e6:.0f}MB)", extra))

def num_str(s): m = re.match(r"\s*(\d+(?:\.\d+)?)", str(s)); return m.group(1) if m else None
def cutter(s): m = re.match(r"\s*\d+(?:\.\d+)?\s*([A-Za-z]{2,})", str(s)); return m.group(1).upper() if m else ""
def surname(s):
    s = str(s).strip().upper(); m = re.match(r"(VON\s+\w+|VAN\s+\w+|DE\s+\w+|LA\s+\w+)", s)
    return m.group(1).replace(" ", "") if m else (re.split(r"[,\s]+", s)[0] or "")
def label_match(det, gl, section):
    if section == "nonfiction":
        dn, gn, dc, gc = num_str(det), num_str(gl), cutter(det), cutter(gl)
        return bool(dn and gn and dc and gc) and (dn.startswith(gn) or gn.startswith(dn)) and dc[:3] == gc[:3]
    ds, gs = surname(det), surname(gl); mn = min(len(ds), len(gs))
    if mn < 3: return ds == gs
    if mn >= 6: return sum(1 for i in range(mn) if ds[i] != gs[i]) <= 1
    return ds[:4] == gs[:4]

imgs = {n: cv2.imread(str(IMAGES / n)) for n in HARD}

# det models to try (app only unless --dets given: #4 det-swap is deferred)
DETS = [("app v6 det (10MB)", str(MODELS / "det.onnx"))]
if "--dets" in sys.argv:
    try:
        for m in sorted(glob.glob(os.path.join(rbase, "models", "*det*.onnx"))):
            DETS.append((f"{os.path.basename(m)[:16]} ({os.path.getsize(m)/1e6:.0f}MB)", m))
    except Exception:
        pass

print(f"{'det':22} {'rec':26} {'recall':>10}  {'time/img':>9}")
print("-" * 74)
for det_label, det_path in DETS:
    det_sess = ort.InferenceSession(det_path, providers=["CPUExecutionProvider"])
    for label, path in CANDIDATES:
        rec_sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta = rec_sess.get_modelmeta().custom_metadata_map.get("character", "")
        dict_chars = meta.split("\n")
        allowed = P.build_allowed_classes(dict_chars)
        P.run_ocr(imgs[HARD[0]], det_sess, rec_sess, dict_chars, allowed)  # warm
        tot_hc = tot_m = 0; t_sum = 0.0; per = []
        for n in HARD:
            gtv = GT[n]; section = gtv["section"]
            gt, unc = [], set()
            for r in gtv["rows"]:
                gt += r["labels"]; unc |= set(r.get("uncertain", []))
            hc = [l for l in gt if l not in unc]
            t0 = time.time()
            labels, _ = P.run_ocr(imgs[n], det_sess, rec_sess, dict_chars, allowed)
            t_sum += time.time() - t0
            detected = [l["text"] for l in labels]; used = [False] * len(detected); m = 0
            for gl in hc:
                for k, c in enumerate(detected):
                    if not used[k] and label_match(c, gl, section): used[k] = True; m += 1; break
            tot_hc += len(hc); tot_m += m
            per.append(f"{m}/{len(hc)}")
        print(f"{det_label:22} {label:26} {tot_m}/{tot_hc} ({tot_m/tot_hc*100:.0f}%)  {t_sum/len(HARD):.2f}s  [{' '.join(per)}]")
