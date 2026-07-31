"""
Local stand-in for the Modal /ocr endpoint, for end-to-end testing WITHOUT
deploying. Serves the exact same pipeline (ocr_pipeline.run_ocr) over HTTP with
the same JSON contract and CORS/CORP headers a browser needs, so site_suite.mjs
can drive the real cloud code path (browser -> HTTP -> fusion) and measure
recall + latency. On Modal this same pipeline runs on a GPU; here it's CPU.

Run:  python local_ocr_server.py [--port=8799]
"""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
import ocr_pipeline as P

MODELS = Path(r"C:\Users\krish\shelfcheck-ocr\models")
PORT = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--port=")), "8799"))

det = ort.InferenceSession(str(MODELS / "det.onnx"), providers=["CPUExecutionProvider"])
rec = ort.InferenceSession(str(MODELS / "rec.fp32.onnx"), providers=["CPUExecutionProvider"])
dict_chars = (MODELS / "rec_dict.txt").read_text(encoding="utf-8").split("\n")
allowed = P.build_allowed_classes(dict_chars)
print(f"local OCR server ready on :{PORT} ({len(allowed)} classes)", flush=True)


class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        self.send_response(200); self._cors()
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "provider": "CPU-local"}).encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        t0 = time.time()
        arr = np.frombuffer(body, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        labels, unread = P.run_ocr(bgr, det, rec, dict_chars, allowed)
        out = json.dumps({"labels": labels, "unread": unread,
                          "ms": int((time.time() - t0) * 1000), "provider": "CPU-local"}).encode()
        self.send_response(200); self._cors()
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
