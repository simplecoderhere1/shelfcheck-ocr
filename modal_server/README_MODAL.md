# Cloud OCR on Modal — deploy guide

This runs the **same** PP-OCR models the app already uses, but on a GPU, so a
scan reads in **under a second** instead of ~7s on the phone. Modal's free tier
is **$30 of compute every month, no credit card**, which is roughly 50 GPU-hours
on the T4 this uses — plenty for one library.

The app already talks to this endpoint (see `callCloudOcr` in `index.html`); you
just have to stand the endpoint up and paste its URL in one place.

---

## What's in this folder

| file | what it is |
|---|---|
| `ocr_pipeline.py` | the OCR pipeline (faithful port of the on-device engine). Runs on CPU locally and GPU on Modal, unchanged. |
| `modal_app.py` | the Modal wrapper: GPU container + `/ocr` and `/warm` web endpoints. |
| `validate_local.py` | proves the pipeline matches the app's accuracy on the 10 test shelves (CPU). Already run: **92% raw → ~95% after the app's normalization**. |
| `local_ocr_server.py` | runs the endpoint on your own machine (CPU) for testing without deploying. |

---

## One-time setup

1. **Make a Modal account** (no card): <https://modal.com> → sign up with GitHub or Google.
2. **Install the CLI and log in** (needs Python 3.9+ on your computer):
   ```
   pip install modal
   modal token new
   ```
   That opens a browser to link the CLI to your account.

## Deploy

From this folder (`modal_server`):
```
modal deploy modal_app.py
```
First deploy builds the GPU image (a few minutes). When it finishes it prints a
**web URL** like:
```
https://YOURNAME--shelfcheck-ocr-ocr-web.modal.run
```
Copy that URL.

## Turn it on in the app

Open `index.html`, find this line (near the top of the script, ~line 490):
```js
const CLOUD_OCR_URL_DEFAULT = '';
```
Paste your URL in the quotes (no trailing `/ocr`):
```js
const CLOUD_OCR_URL_DEFAULT = 'https://YOURNAME--shelfcheck-ocr-ocr-web.modal.run';
```
Commit and push. GitHub Pages redeploys, and every volunteer now gets GPU OCR.
A **"Cloud OCR (GPU, fast)"** switch appears in the settings menu; if the server
is ever down or you run out of credits, flip it off (or it falls back to the
on-device engine automatically on any error).

---

## Try it before deploying (optional)

Run the endpoint on your own machine and point the app at it:
```
python local_ocr_server.py --port=8799
```
Then open the app with `?cloud=http://127.0.0.1:8799` on the URL once — it
remembers. (This is exactly how it was tested: full recall preserved at 95.6%.)

---

## Cost & the cold-start trick

- **Warm** (container already running): reads are sub-second.
- **Cold** (idle >60s, scaled to zero): the first request reloads the models and
  takes several seconds. The app hides this by pinging `/warm` the moment you tap
  **Take Photo**, so the GPU spins up while you're framing the shot.
- Idle costs nothing (it scales to zero). You only spend credit while it's
  actively reading, so ~$30/month comfortably covers a single library's use.
- Want every scan instant during open hours? In `modal_app.py` set
  `min_containers=1` — but that keeps a GPU running continuously and burns the
  free credit in ~2 days, so only do it if you upgrade to a paid plan.

## Notes

- Photos are sent to your Modal container to be read. Low-sensitivity (library
  shelves), but worth knowing.
- The endpoint is public (no key). To lock it down later, add a shared secret
  header check in `modal_app.py`'s `/ocr` handler and send it from `callCloudOcr`.
- Accuracy note: on the test set the cloud path holds **95.6% recall**; it
  produced **one** false-positive misshelving flag on the single hardest shelf
  (a partial fiction shelf) that the on-device path didn't. Acceptable under the
  current "few FPs OK above 90%" bar; the planned Gemini red-flag verifier would
  scrub these later.
