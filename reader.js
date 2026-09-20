// Single vision-model reader for one whole-shelf photo.
//
// Replaces the old Azure-OCR + geometry-assembly + Gemini-fusion pipeline
// with one model call that reads every book directly. The model is asked for
// each book's shelf ROW as well as its box, so one photo of a whole shelf
// (several stacked rows) reads and orders in a single call — no per-segment
// capture or cross-segment stitching. Validated 2026-09-20 on the 10 real
// ground-truth library shelves in test_items, fed whole (downscaled to
// 3072px, exactly what a phone upload becomes): gemini-3.1-flash-lite read
// ~71% of spines exactly in 5-12s per shelf and its ordering false-reds were
// the same dense-nonfiction cutter cases seen on close crops. The tradeoff
// (chosen with the user): whole-shelf capture is simpler and reads sparse
// shelves cleanly, but on a densely packed shelf shot from normal distance it
// can miss roughly half the spines to the resolution limit — a spine it does
// not read is silently skipped. Candidates tested and rejected: 3.5-flash-
// lite misread a name despite being "faster" on paper (matches this project's
// own prior finding that newer/faster is not automatically better here);
// 2.5-flash silently dropped duplicate-author books and was slower; 3.8-flash
// and a couple of name variants 503'd repeatedly, consistent with its free
// tier being far too small (~20 req/day) for real volunteer use.
//
// The model IS asked for a bounding box per book. An earlier version skipped
// it (evenly dividing the segment width by book count) to shave latency, but
// an even split points a volunteer at the wrong spine whenever books vary in
// width — which they always do — so a red flag has to land on the actual
// book, not an approximate slot. Measured cost of adding the box: median read
// ~3.8s, worst ~6s on a ~15-book segment (vs ~2-3s without), still within the
// usable range for the close-up segments this app captures. The box also
// feeds ordering.js's geometry-based dedup (overlapping re-reads of one
// sticker), which the even split could not.
//
// Box format from the model: [ymin, xmin, ymax, xmax], each 0-1000 normalized
// to the segment image. Stored two ways: `_box01` (normalized [x,y,w,h] in
// 0..1) for drawing the overlay at any display size, and `_bbox` (pixel
// [x,y,w,h] in the sent image's own space) for ordering.js. When the model
// omits or malforms a box for a row, that one book falls back to an even slot
// so it still gets drawn.

const GEMINI_MODEL = 'gemini-3.1-flash-lite';
const READ_MAX_DIM = 3072;   // a whole-shelf phone photo is downscaled to this before sending
const READ_QUALITY = 0.9;
const LOW_CONF = 0.5;        // below this, sortKey() in ordering.js won't sort the book

// Output shape chosen for latency. The response time is dominated by output-
// token generation, so the format is kept as compact as possible WITHOUT
// changing what the model reads (measured 2026-09-20: this compact shape is
// ~20% faster than one object-per-book with a full 4-number box, at equal read
// accuracy). Two savings: (1) each book is a positional array, not an object,
// so the long key names are not re-emitted ~45 times; (2) the vertical box
// coords are given ONCE PER ROW as a y-band, not per book — a book only needs
// its horizontal extent (xmin,xmax) plus which row it's in, and the row's band
// supplies the vertical extent. A full 4-number box per book cost the most
// tokens and, on dense fiction, sometimes degraded the read; this shape avoids
// both. Drawing then frames a flagged book as (its x-extent) x (its row band),
// which points at the spine more clearly than the old per-book sticker boxes.
function buildPrompt(section) {
  const convention = section === 'fiction'
    ? 'Each book has a small white sticker printing the author\'s SURNAME then given name(s), e.g. "WALLACE, David Foster".'
    : section === 'nonfiction'
    ? 'Each book has a small white sticker printing a Dewey call number then a short cutter code, e.g. "635.04 HEM".'
    : 'Each book has a small white sticker with either an author surname or a Dewey call number + cutter code.';
  return `You are looking at a photo of one or more full rows (shelves) of library book spines, in physical order: within a row left-to-right, and top row before bottom row. ${convention} Read every sticker you can see.

Return ONLY a JSON object with two keys:
{"rows": [[ymin, ymax], ...], "books": [["<exact sticker text, on ONE line, spaces not line breaks>", <confidence 0 to 1>, <row index>, <xmin>, <xmax>], ...]}
- "rows": one [ymin, ymax] per physical shelf row, top row first, giving that row's top and bottom edge as integers 0 to 1000 normalized to image height.
- "books": one array per book in reading order (each row left to right, top row first). The row index is 0-based into "rows"; xmin and xmax are the spine's left and right edges as integers 0 to 1000 normalized to image width, tight to that one book.

Copy each sticker exactly as printed — copy only the digits you can actually see and do not pad a call number to match its neighbours. If a sticker is partly cut off at the edge of the photo, or blurry, or you are unsure of the text, still include the book with a LOW confidence (well below 0.5) rather than guessing or omitting it — a low-confidence entry is fine, a confident wrong guess is the worst outcome. No other text, no markdown fences.`;
}

function stepDownscale(srcCanvas, tW, tH) {
  let cur = srcCanvas, cw = cur.width, ch = cur.height;
  while (cw / tW > 2 || ch / tH > 2) {
    const nw = Math.max(tW, Math.ceil(cw / 2)), nh = Math.max(tH, Math.ceil(ch / 2));
    const step = document.createElement('canvas');
    step.width = nw; step.height = nh;
    step.getContext('2d').drawImage(cur, 0, 0, nw, nh);
    cur = step; cw = nw; ch = nh;
  }
  if (cw !== tW || ch !== tH) {
    const fin = document.createElement('canvas');
    fin.width = tW; fin.height = tH;
    fin.getContext('2d').drawImage(cur, 0, 0, tW, tH);
    return fin;
  }
  return cur;
}

// canvasSrc: a <canvas> already holding the decoded segment photo (native
// pixels — do the decode/orientation-fix upstream, same as the old
// loadPhoto -> snapshot canvas pattern).
function encodeSegment(canvasSrc, maxDim = READ_MAX_DIM, quality = READ_QUALITY) {
  const W = canvasSrc.width, H = canvasSrc.height;
  const scale = Math.min(1, maxDim / Math.max(W, H));
  const sentW = Math.round(W * scale), sentH = Math.round(H * scale);
  const resized = stepDownscale(canvasSrc, sentW, sentH);
  return new Promise((resolve, reject) => {
    resized.toBlob(blob => {
      if (!blob) { reject(new Error('segment image encoding failed')); return; }
      resolve({ blob, sentW, sentH });
    }, 'image/jpeg', quality);
  });
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result).split(',')[1] || '');
    fr.onerror = () => reject(fr.error || new Error('read failed'));
    fr.readAsDataURL(blob);
  });
}

// Median latency measured against real shelf photos is 2-4s for a close crop,
// but a whole-shelf read runs 5-12s here on a fast connection and a single
// call was once observed taking ~20s (real network/API variance, not a bug —
// see [[shelfcheck-latency]]: output-token generation is the slow part). A
// phone on library Wi-Fi is slower still and drops requests, so the timeout is
// generous and network failures are retried a few times with backoff before
// the volunteer sees an error. `TIMEOUT` gets its own message because "wait
// longer / better signal" is different advice from "the reader is down".
const READ_TIMEOUT_MS = 40000;
const MAX_ATTEMPTS = 3;

const sleep = ms => new Promise(r => setTimeout(r, ms));

// Errors are tagged so callGemini knows what to retry and what to tell the
// user. NETWORK (fetch could not complete — the "Failed to fetch" the phone
// hit) and TIMEOUT and HTTP-5xx are transient and worth retrying; HTTP-4xx and
// a malformed body are not.
class ReaderError extends Error {
  constructor(kind, message, retryable) { super(message); this.kind = kind; this.retryable = retryable; }
}

async function callGeminiOnce(geminiUrl, base64, section, model, externalSignal) {
  const body = {
    model,
    generationConfig: {
      temperature: 0,
      response_mime_type: 'application/json',
      thinkingConfig: { thinkingBudget: 0 },
    },
    contents: [{ parts: [
      { text: buildPrompt(section) },
      { inline_data: { mime_type: 'image/jpeg', data: base64 } },
    ]}],
  };
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, READ_TIMEOUT_MS);
  const onExternalAbort = () => ctrl.abort();
  if (externalSignal) externalSignal.addEventListener('abort', onExternalAbort);
  let res;
  try {
    res = await fetch(geminiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
  } catch (e) {
    if (externalSignal?.aborted) throw new ReaderError('ABORTED', 'cancelled', false);
    if (timedOut) throw new ReaderError('TIMEOUT', 'the reader took too long to answer', true);
    // A fetch that rejects (rather than returning a response) is a network
    // failure: no connection, DNS, or the request was blocked/dropped.
    throw new ReaderError('NETWORK', 'could not reach the reader (no network response)', true);
  } finally {
    clearTimeout(timer);
    if (externalSignal) externalSignal.removeEventListener('abort', onExternalAbort);
  }
  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new ReaderError('HTTP', `reader service returned HTTP ${res.status} ${errText.slice(0, 160)}`.trim(), res.status >= 500 || res.status === 429);
  }
  const json = await res.json().catch(() => null);
  const parts = json?.candidates?.[0]?.content?.parts || [];
  const text = parts.map(p => p.text || '').join('');
  // The model usually returns a clean JSON object, but occasionally wraps it in
  // stray text or a code fence; take the outermost {...} before parsing.
  const start = text.indexOf('{'), end = text.lastIndexOf('}');
  const slice = start >= 0 && end > start ? text.slice(start, end + 1) : text;
  let parsed;
  try { parsed = JSON.parse(slice); }
  catch (e) { throw new ReaderError('PARSE', `reader returned an unreadable response: ${text.slice(0, 160)}`, false); }
  if (!parsed || !Array.isArray(parsed.books)) throw new ReaderError('PARSE', 'reader did not return a list of books', false);
  return parsed;   // { rows: [[ymin,ymax],...], books: [[label,conf,row,xmin,xmax],...] }
}

async function callGemini(geminiUrl, base64, section, model, externalSignal) {
  let lastErr;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      return await callGeminiOnce(geminiUrl, base64, section, model, externalSignal);
    } catch (err) {
      lastErr = err;
      if (externalSignal?.aborted) throw err;              // the caller cancelled
      if (!(err instanceof ReaderError) || !err.retryable) throw err;
      if (attempt < MAX_ATTEMPTS) await sleep(700 * attempt);  // 0.7s, 1.4s backoff
    }
  }
  throw lastErr;
}

const clamp01 = v => Math.max(0, Math.min(1000, Number(v))) / 1000;

// Build a normalized [x, y, w, h] box (0..1) from a book's horizontal extent
// (xmin, xmax) and its row's vertical band ([ymin, ymax], from the "rows"
// array). Falls back to an even horizontal slot and/or full height when a
// piece is missing, so a book is always drawable. Returns null only if there
// is genuinely nothing usable.
function buildBox01(xmin, xmax, band, slotX, slotW) {
  let x0 = Number.isFinite(Number(xmin)) ? clamp01(xmin) : slotX;
  let x1 = Number.isFinite(Number(xmax)) ? clamp01(xmax) : slotX + slotW;
  if (x1 < x0) [x0, x1] = [x1, x0];
  let w = x1 - x0;
  if (w <= 0) { x0 = slotX; w = slotW; }
  let y0 = 0, y1 = 1;
  if (Array.isArray(band) && band.length === 2) {
    let a = clamp01(band[0]), b = clamp01(band[1]);
    if (b < a) [a, b] = [b, a];
    if (b > a) { y0 = a; y1 = b; }
  }
  const h = y1 - y0;
  return [x0, y0, w, h];
}

// One label per line, spaces not line breaks: the model sometimes returns the
// call number and cutter on separate lines ("635\nGAR"), which then displays
// with a break and would split wrong in any code that tokenizes on whitespace.
function cleanLabel(s) {
  return String(s || '').replace(/\s+/g, ' ').trim();
}

// The public entry point. canvasSrc holds the decoded whole-shelf photo at
// native (or near-native) resolution. section is 'fiction' | 'nonfiction'.
// Returns { books, sentW, sentH, model, elapsedMs }. Each book carries
// spine_label, _score, _src, shelfRow (the model's 0-based row index),
// _box01 (normalized [x,y,w,h] for drawing at any display size) and _bbox
// (pixel [x,y,w,h] in the sent image's own space, for ordering.js's geometry
// rules). The caller passes books straight to checkOrder.
export async function readShelf(canvasSrc, section, {
  geminiUrl,
  model = GEMINI_MODEL,
  signal,
} = {}) {
  if (!geminiUrl) throw new Error('reader: geminiUrl is required');
  const t0 = performance.now();
  const { blob, sentW, sentH } = await encodeSegment(canvasSrc);
  const base64 = await blobToBase64(blob);
  const parsed = await callGemini(geminiUrl, base64, section, model, signal);
  const bands = Array.isArray(parsed.rows) ? parsed.rows : [];
  // Each book is [label, confidence, rowIndex, xmin, xmax].
  const rows = parsed.books;
  const n = rows.length;
  const books = rows.map((r, i) => {
    const label = cleanLabel(Array.isArray(r) ? r[0] : r?.label);
    const rawScore = Array.isArray(r) ? r[1] : r?.confidence;
    const score = typeof rawScore === 'number' ? Math.max(0, Math.min(1, rawScore)) : 0;
    const rowIdx = Number.isFinite(Number(Array.isArray(r) ? r[2] : r?.shelfRow)) ? Number(Array.isArray(r) ? r[2] : r?.shelfRow) : 0;
    const xmin = Array.isArray(r) ? r[3] : undefined;
    const xmax = Array.isArray(r) ? r[4] : undefined;
    // A book's vertical extent comes from its row's band; horizontal from its
    // own xmin/xmax, with an even slot as fallback for either.
    const box01 = buildBox01(xmin, xmax, bands[rowIdx], n ? i / n : 0, n ? 1 / n : 1);
    const bbox = [
      Math.round(box01[0] * sentW), Math.round(box01[1] * sentH),
      Math.round(box01[2] * sentW), Math.round(box01[3] * sentH),
    ];
    const book = {
      spine_label: label,
      _score: score,
      _box01: box01,
      _bbox: bbox,
      _src: 'gemini',
      shelfRow: rowIdx,
    };
    if (score < LOW_CONF) book.confidence = 'low';
    return book;
  }).filter(b => b.spine_label);
  return { books, sentW, sentH, model, elapsedMs: Math.round(performance.now() - t0) };
}
