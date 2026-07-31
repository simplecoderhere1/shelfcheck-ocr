"""
Modal deployment wrapper around the faithful OCR port (ocr_pipeline.py).

Runs the SAME det + rec ONNX models the browser runs, on a GPU, and returns the
SAME JSON shape LocalOCR.run() produces — so the web app treats this endpoint
as a drop-in for the on-device engine, and every downstream step (fusion,
ordering, the zero-false-positive red-flag rule, yellow unread boxes) is
unchanged.

Endpoints (both public HTTPS, no key — the models read library spines, nothing
sensitive; add a token later if you want):
  POST /ocr   body = raw JPEG/PNG bytes  ->  {"labels":[...], "unread":[...], "ms":N}
  GET  /warm  ->  {"ok": true}           warm-up ping; the browser hits this the
                                          moment the app opens so the container
                                          is hot by the time the photo arrives.

Deploy:  modal deploy modal_app.py      (see README_MODAL.md for the full walk-through)

Cost shape: scale-to-zero, billed per active GPU-second. Warm requests are
sub-second; a cold start reloads the models (~seconds). The browser's warm-on-
open ping hides most of that behind the time the volunteer spends framing.
"""
import time
from pathlib import Path

import modal

# ── Container image: CUDA base + onnxruntime-gpu + the pipeline & models ───────
# The nvidia/cuda base is the reliable way to satisfy onnxruntime-gpu's CUDA/
# cuDNN needs. Models (~24MB) are baked into the image so no volume is required.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "onnxruntime-gpu==1.20.1",
        "opencv-python-headless==4.10.0.84",
        "numpy==1.26.4",
        "fastapi[standard]==0.115.0",
    )
    .add_local_file("ocr_pipeline.py", "/root/ocr_pipeline.py")
    .add_local_dir("../models", "/root/models")   # det.onnx, rec.fp32.onnx, rec_dict.txt
)

app = modal.App("shelfcheck-ocr")

# GitHub Pages origin(s) allowed to call this endpoint from the browser.
ALLOWED_ORIGINS = ["*"]   # tighten to your Pages URL once it works


@app.cls(
    image=image,
    gpu="T4",              # cheapest CUDA GPU; ~50 GPU-hours fits the $30/mo free credit
    scaledown_window=60,   # stay warm 60s after the last request (covers a burst of scans)
    min_containers=0,      # scale to zero when idle so idle costs nothing
)
class OCR:
    @modal.enter()
    def load(self):
        import onnxruntime as ort
        import ocr_pipeline as P

        self.P = P
        models = Path("/root/models")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self.det = ort.InferenceSession(str(models / "det.onnx"), providers=providers)
        self.rec = ort.InferenceSession(str(models / "rec.fp32.onnx"), providers=providers)
        self.dict_chars = (models / "rec_dict.txt").read_text(encoding="utf-8").split("\n")
        self.allowed = P.build_allowed_classes(self.dict_chars)
        self.provider = self.det.get_providers()[0]
        # One warm inference so the first real request doesn't pay CUDA graph
        # init — a tiny black image through the full pipeline.
        import numpy as np
        try:
            self.P.run_ocr(np.zeros((64, 64, 3), np.uint8), self.det, self.rec,
                           self.dict_chars, self.allowed)
        except Exception:
            pass

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        import numpy as np
        import cv2

        api = FastAPI()
        api.add_middleware(
            CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
            allow_methods=["*"], allow_headers=["*"],
        )

        @api.get("/warm")
        def warm():
            return {"ok": True, "provider": self.provider}

        @api.post("/ocr")
        async def ocr(request: Request):
            body = await request.body()
            t0 = time.time()
            arr = np.frombuffer(body, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return JSONResponse({"error": "bad image"}, status_code=400)
            labels, unread = self.P.run_ocr(bgr, self.det, self.rec,
                                            self.dict_chars, self.allowed)
            return {"labels": labels, "unread": unread,
                    "ms": int((time.time() - t0) * 1000), "provider": self.provider}

        return api
