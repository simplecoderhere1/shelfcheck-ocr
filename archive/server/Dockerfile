# ── ShelfCheck OCR v2 backend ───────────────────────────────────────────────
# Works on any Docker host: Render, Fly.io, Railway, Cloud Run, plain VPS.
# HuggingFace Spaces also works — set sdk: docker, app_port: 7860 in README.
#
# Build:  docker build -t shelfcheck-ocr .
# Run:    docker run -p 7860:7860 shelfcheck-ocr
#
# Env vars (all optional):
#   PORT                 listening port (default 7860, override for non-HF hosts)
#   SHELFCHECK_API_KEY   simple API key guard
#   ALLOWED_ORIGINS      comma-separated CORS origins (default *)
#   MAX_SIDE             resize longest edge before OCR (default 1600)
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install system deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=7860
EXPOSE 7860

# Pre-download the RapidOCR ONNX models so first request isn't slow
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
