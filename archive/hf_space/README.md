---
title: ShelfCheck OCR
emoji: 📚
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# ShelfCheck cloud OCR (free CPU, no credit card)

Serves the same PP-OCR det + rec models the ShelfCheck web app otherwise runs
on-device, over HTTP, so a phone doesn't have to do the inference.

## Why this exists

The app's scan latency was ~9.6s median, and the dominant cost was the
on-device label pass (~6-7s). Every attempt to make that faster *on the phone*
was measured and failed: WebGPU was slower, INT8 was slower, and dropping the
detector to 1536px cost 19 points of recall.

The bottleneck turned out not to be the model but **ONNX under WASM on a mobile
CPU**. The identical graph, run natively, is about 3x faster — even on two weak
vCPUs. Measured on the 10-image test set with `intra_op_num_threads=2` (exactly
this Space's hardware):

| | |
|---|---|
| per shelf | 1.84s – 2.81s |
| average | **2.16s** (`rec.onnx`) |
| raw label recall | **92.0%** — identical to the on-device engine |

So this needs no GPU, which is what keeps it on a free tier with no card.

## Why Gradio and not Docker

This started as a Docker Space. Hugging Face has since made the **Docker SDK
paid-only** (PRO for personal accounts); Gradio Spaces are still free. Same
hardware, same models, different packaging.

## API

Gradio's two-step call protocol:

```bash
# 1. submit — returns {"event_id": "..."}
curl -X POST https://<space>.hf.space/gradio_api/call/ocr \
     -H 'Content-Type: application/json' \
     -d '{"data": ["<base64 jpeg>"]}'

# 2. collect — SSE stream ending in the result
curl -N https://<space>.hf.space/gradio_api/call/ocr/<event_id>
```

The result is a JSON **string**: `{"labels":[...],"unread":[...],"ms":N}`,
byte-compatible with `LocalOCR.run()` in `ocr_local.js`. The browser feeds it
into the same `rawLabelsToBooks()` normalisation, so two-engine fusion, the
red-flag confidence gate and the yellow unread boxes are all untouched.

Base64 in (rather than a file upload) is deliberate: a plain string in and out
makes the contract independent of how Gradio serialises file objects.

## Deploying

1. Create a Space: **Gradio** SDK, **CPU Basic** (free, 2 vCPU / 16GB).
2. Clone it, then from this repo:
   ```bash
   bash hf_space/prepare_space.sh /path/to/space-clone
   ```
   That stages the pipeline + models and writes `.gitignore` / `.gitattributes`
   (LFS for the `.onnx` files, `eol=lf`).
3. `git lfs install --local && git add -A && git commit && git push`.
4. Put the Space URL into `CLOUD_OCR_URL_DEFAULT` in `index.html`.

## Operational notes

- **Free Spaces sleep after 48h idle.** The first request after that pays a
  container boot. The app pings `/warm` on open and on the Take Photo tap,
  which covers the ordinary case; a Space that has slept overnight will still
  be slow for whoever scans first. On-device OCR remains the automatic
  fallback, so a cold or down Space degrades to today's behaviour rather than
  breaking a scan.
- **2 vCPU is shared across concurrent requests.** ORT is pinned to 2 intra-op
  threads so simultaneous scans queue instead of thrashing. A volunteer shift
  with a few phones is fine; a classroom of 30 is not.
- **Upload size is now the latency budget**, not inference. Keep the posted
  JPEG near a 2048px long edge — that is the detector's input resolution, and
  sending more just costs upload time.
