"""
ShelfCheck cloud OCR — Hugging Face Space (Gradio SDK, free CPU Basic: 2 vCPU).

Runs the SAME det + rec ONNX models the browser runs on-device, natively via
onnxruntime instead of WASM, so a phone doesn't have to do the inference.

Why CPU is enough (measured on the 10-image test set at intra_op_num_threads=2,
which is exactly this Space's hardware): 1.84s - 2.81s per shelf, avg 2.16s,
raw label recall 92.0% — identical to the on-device engine. The phone was slow
because of WASM, not because the model is heavy.

Why GRADIO and not Docker: Hugging Face made the Docker SDK paid-only. Gradio
Spaces remain free on personal accounts, so the same FastAPI-shaped service is
delivered as a Gradio app instead. `ocr_b64` is the machine endpoint the web app
calls; the Image tab is a human demo of the same function.
"""
import base64
import json
import os
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort

import ocr_pipeline as P

MODELS = Path(os.environ.get("MODELS_DIR", "models"))
REC_MODEL = os.environ.get("REC_MODEL", "rec.onnx")
THREADS = int(os.environ.get("ORT_THREADS", "2"))


def _session(path: Path) -> ort.InferenceSession:
    # Pinned so ORT doesn't oversubscribe the 2 shared vCPUs when two
    # volunteers scan at once — and so this matches the benchmark.
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


det = _session(MODELS / "det.onnx")
rec = _session(MODELS / REC_MODEL)
dict_chars = (MODELS / "rec_dict.txt").read_text(encoding="utf-8").split("\n")
allowed = P.build_allowed_classes(dict_chars)

# One warm pass so the first real request doesn't pay lazy kernel init.
try:
    P.run_ocr(np.zeros((64, 64, 3), np.uint8), det, rec, dict_chars, allowed)
except Exception:
    pass

print(f"shelfcheck OCR ready: rec={REC_MODEL} threads={THREADS} "
      f"classes={len(allowed)}", flush=True)


def _run(bgr):
    t0 = time.time()
    labels, unread = P.run_ocr(bgr, det, rec, dict_chars, allowed)
    return {"labels": labels, "unread": unread,
            "ms": int((time.time() - t0) * 1000), "provider": "CPU"}


def ocr_b64(image_b64: str) -> str:
    """Machine endpoint. In: base64 JPEG (bare or as a data: URI).
    Out: a JSON STRING, byte-compatible with what LocalOCR.run() produces in
    the browser, so every downstream step (fusion, ordering, the red-flag
    confidence gate, yellow unread boxes) is unchanged.

    A plain string in / string out is deliberate: it makes the HTTP contract
    unambiguous, instead of depending on how Gradio serialises file objects.
    """
    try:
        s = (image_b64 or "").strip()
        if not s:
            return json.dumps({"error": "empty input"})
        if s.startswith("data:"):
            s = s.split(",", 1)[-1]
        buf = np.frombuffer(base64.b64decode(s), np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return json.dumps({"error": "bad image"})
        return json.dumps(_run(bgr))
    except Exception as e:            # never 500 — the caller falls back on null
        return json.dumps({"error": str(e)})


def ocr_demo(img):
    """Human demo: draw the detected label boxes on the photo and show the reads."""
    if img is None:
        return None, "Upload a shelf photo."
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = _run(bgr)
    h, w = bgr.shape[:2]
    vis = img.copy()
    for l in out["labels"]:
        x, y, bw, bh = l["bboxFrac"]
        p1 = (int(x * w), int(y * h))
        p2 = (int((x + bw) * w), int((y + bh) * h))
        cv2.rectangle(vis, p1, p2, (34, 197, 94), max(2, w // 400))
    lines = [f"{l['text']}   ({l['conf']:.2f})" for l in out["labels"]]
    summary = (f"{len(out['labels'])} labels read · {len(out['unread'])} detected "
               f"but unreadable · {out['ms']}ms\n\n" + "\n".join(lines))
    return vis, summary


with gr.Blocks(title="ShelfCheck OCR") as demo:
    gr.Markdown(
        "# ShelfCheck OCR\n"
        "Reads library call-number labels off a shelf photo. Same PP-OCR "
        "det+rec models the [ShelfCheck](https://github.com/krish-sundareswar/shelfcheck-ocr) "
        "web app runs on-device — served here so phones don't have to.\n\n"
        "**API:** `POST /gradio_api/call/ocr` with `{\"data\": [\"<base64 jpeg>\"]}`."
    )
    with gr.Tab("Demo"):
        with gr.Row():
            inp = gr.Image(type="numpy", label="Shelf photo")
            out_img = gr.Image(type="numpy", label="Detected labels")
        out_txt = gr.Textbox(label="Reads", lines=14)
        gr.Button("Read shelf", variant="primary").click(
            ocr_demo, inputs=inp, outputs=[out_img, out_txt], api_name=False)
    with gr.Tab("API"):
        gr.Markdown("Endpoint used by the web app. Base64 JPEG in, JSON out.")
        b64_in = gr.Textbox(label="base64 JPEG", lines=3)
        json_out = gr.Textbox(label="JSON", lines=10)
        gr.Button("Run").click(ocr_b64, inputs=b64_in, outputs=json_out,
                               api_name="ocr")

# queue() is what makes the /gradio_api/call/<name> endpoints available.
demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=7860)
