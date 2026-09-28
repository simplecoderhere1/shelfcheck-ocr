# ShelfCheck

Find misshelved library books from a phone photo. Live at
https://simplecoderhere1.github.io/shelfcheck-ocr/

Pick **Fiction** (author surname, then given name) or **Nonfiction** (Dewey
number, then cutter), photograph a shelf straight on, and the app marks:

- **red, numbered** — the book is in the wrong place; the list below the photo
  says where it goes ("goes after 616.8527 RAM");
- **yellow** — its sticker could not be read, check it by hand;
- **no box** — in order.

A shelf of Dewey numbers is read as nonfiction even from the Fiction tab (and
vice versa).

## How a photo is read

```
phone ──(one JPEG, 3072px q0.75, ~700 KB)──> shelfcheck-read Worker
                                              ├─ Gemini proxy  (x2, raced)  ─┐
                                              └─ Azure Read worker           ─┤
phone <───────── both answers ───────────────────────────────────────────────┘
      fuse.js: align + reconcile  →  ordering.js: check order  →  boxes
```

- **Gemini** (`gemini-3.1-flash-lite`, fallback `gemini-2.5-flash` on a 429)
  lists every sticker row by row, in order. It reads a whole shelf well, but it
  *conforms outliers to their neighbourhood* — it read "WERTH, Janet" as
  "WESLEY, Janet", and moved a stranded TATA back into the TATA run in its list.
  Those are exactly the books the app exists to find.
- **Azure Read** copies characters literally and gives exact word boxes, but on
  its own cannot reliably tell stickers from cover text or group words into books.
- **`fuse.js`** finds the physical sticker rows from Azure's words, assigns the
  model's rows to them, and reconciles each book: agreement anchors a row;
  Azure's literal text wins where the model copied a neighbour and the sticker's
  other half (given name / call number) confirms it; a book the model relocated
  goes back where Azure saw it; a book only the model reports is never accused.
- **`ordering.js`** is the order check (longest in-order run, then demotions that
  keep a correctly shelved book from being accused). It trusts reads both engines
  confirmed (`_azConfirmed`, `_numConfirmed`).

The fan-out Worker (`workers/read-worker.js`) exists for latency: uploading the
photo once as binary instead of twice (base64 to Gemini + binary to Azure) took
the throttled-phone profile from 5.7–12.6s to under 5s, and racing two Gemini
requests cuts its long tail (its latency is bimodal and random per request).
The page falls back to calling both proxies directly if the Worker is down, and
to a Gemini-only read (approximate box positions) if Azure is.

## Measured (2026-09-27, 36 photos in the test corpus)

| | |
|---|---|
| Latency, phone profile (4× CPU, 6 Mbps up) | mean 3.9s, median 3.8s, 32/36 under 5s |
| Latency, desktop | ~2–3s |
| Known misfiles flagged red | 16–18 of 24 per read |
| False red flags | 0–2 per read, ~1,250 books |
| Sort-relevant reading (10 fully transcribed shelves) | 91–94% |

The misfiles not flagged are ones whose deciding digits are hidden on the
sticker (wrapped round the spine or cut by the frame) or unreadable by both
engines; those stay unmarked rather than guessed.

## Limits worth knowing

- Free tiers: Gemini allows 15 requests/minute per model (the app uses 2 per
  photo, so ~7 photos/minute across all volunteers, then it switches model);
  Azure F0 allows 5,000 reads/month (1 per photo), metered by the Azure Worker.
  Enabling billing on the Gemini key (well under a cent per photo) removes the
  per-minute ceiling and usually lowers latency.
- Photograph a shelf straight on; a face-out display book at the end of a row
  is sometimes read as part of it.

## Files

| | |
|---|---|
| `index.html` | the page |
| `reader.js` | encode, call the fan-out Worker (or both proxies), fuse |
| `fuse.js` | prompt, parser, two-engine alignment and reconciliation |
| `ordering.js` | order check |
| `workers/read-worker.js` | one-upload fan-out Worker (`npx wrangler deploy` from `workers/`) |
| `workers/azure/` | key-holding Azure Read Worker with its monthly quota counter (deployed as `shelfcheck-azure-ocr`) |
| `test/` | ordering test against hand-verified ground truth |
| `archive/` | retired readers (Azure-only assembly, on-device OCR), kept for their measurements |
