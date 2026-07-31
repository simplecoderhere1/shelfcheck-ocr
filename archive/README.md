# Archive — the readers ShelfCheck no longer uses

Everything here was part of the app before it standardised on **Azure AI Vision**
as the reader. None of it is loaded by `index.html` any more. It is kept because
each piece encodes a measurement that is expensive to redo, not because any of it
is expected to run again.

Deleting this directory is safe. Nothing in the shipping app imports it.

## What's here

| Path | What it was | Why it went |
|---|---|---|
| `ocr_local.js` | PP-OCRv6 det+rec in the browser via onnxruntime-web | ~6–7s per scan on a phone, against Azure's ~1–1.4s. Free and offline, but it sat on the blocking path and dominated scan latency. |
| `models/` | The det/rec ONNX weights (`det.onnx`, `rec.onnx`, int8) | Only `ocr_local.js` loaded them. 43MB of the deploy. |
| `vendor/ort/` | onnxruntime-web runtime + WASM | Same — only needed by the on-device engine. 39MB. |
| `coi-serviceworker.min.js` | COOP/COEP shim so GitHub Pages could give the WASM engine multi-threading | Only existed to make `ocr_local.js` fast. Cost every visitor a reload on first visit. |
| `worker/` | Cloudflare Worker proxying **ocr.space** | ocr.space measured 2.8s and truncated the deep decimals (`635.048` → `635.04`) that are exactly what hides a misshelved book. |
| `server/`, `modal_server/`, `hf_space/`, `render.yaml` | Python backends running the same PP-OCR models server-side (Render / Modal GPU / Hugging Face Space) | Attempts to move the on-device cost to a server. The HF Space ran past a 45s client timeout on hardware that reads in ~2s locally; Modal was never deployed. |
| `ocr_compare.py`, `ocr_pipeline_validate.py` | Bench scripts for the above | Nothing left to compare. |

## Also removed, but not stored here

The **live AR camera mode** (~820 lines of `index.html`: continuous scanning with
no shutter, multi-frame vote tracking, motion-compensated overlay). It depended on
the on-device engine for its fast loop, and one Azure call per scan cycle would
have burned the 5,000-reads/month allowance in a single session. Recover it from
git history if it's ever wanted:

```
git log --oneline -- index.html          # find a commit before 2026-07-29
git show <commit>:index.html > old.html
```

`models/` and `vendor/` are likewise recoverable from history rather than being
carried in the working tree:

```
git checkout df7b6d9 -- models vendor
```

## What replaced all of it

One reader: **Azure AI Vision Image Analysis 4.0**, behind a Cloudflare Worker
that holds the subscription key (`workers/azure-ocr-worker.js`). Four calls per
nonfiction shelf — full frame, two native-resolution vertical strips, and one
"closer look" sheet of the stickers the first pass was shaky on.

**Gemini** stays wired as the standby in `index.html` (`fallbackReader`), used
only when Azure cannot be reached at all. It reads by inference where Azure
reports only what the camera captured, so it is strictly a better-than-nothing
path — never the default.
