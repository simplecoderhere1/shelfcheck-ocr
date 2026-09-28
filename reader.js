// Reads one whole-shelf photo: a vision model (Gemini) lists every book on each
// row in order, and an OCR engine (Azure Read) reads the same pixels literally
// with exact word boxes. The two run IN PARALLEL on the same JPEG, so the wall
// time is the slower of the two (almost always Gemini), not their sum.
// fuse.js reconciles them; see the header there for why both are needed.
//
// Latency, measured 2026-09-27 on the 36-photo corpus through the same proxies
// the page uses: Azure 0.8-1.8s; Gemini 2-4s typical with a long tail (a
// minority of calls take 6-12s, and a few never finish). The long tail is what
// breaks a 5-second average, so a Gemini call that has not answered after
// HEDGE_MS gets a second, identical request racing it; whichever answers first
// wins and the other is aborted.
//
// Two Gemini failure modes seen on this corpus, both handled here:
//  - a runaway repetition loop at temperature 0 on long same-author runs
//    ("AUSTER, Paul" x 1500 tokens, 21s) -> a small temperature plus an output
//    cap, and a retry if the cap is hit;
//  - finishReason RECITATION (the output resembles a list the model has seen)
//    with no text -> retried at a higher temperature.
// A 429 (the free tier allows 15 requests/minute per model) switches to a
// second model, which has its own quota.

import { labelsPrompt, parseLabels, fuse, toBooks } from './fuse.js';

const MODELS = ['gemini-3.1-flash-lite', 'gemini-2.5-flash'];
const SEND_MAX_DIM = 3072;       // measured: 1536px lost 12 points of reading vs 3072px
const SEND_QUALITY = 0.75;      // measured: 727 KB vs ~1 MB at 0.85, no loss in reading (93.0% vs 90.5% compat, noise)
const HEDGE_MS = 4000;
const GEMINI_TIMEOUT_MS = 20000;
const AZURE_TIMEOUT_MS = 9000;
const MAX_OUTPUT_TOKENS = 1500;  // a 60-book shelf needs ~600

function stepDownscale(src, tW, tH) {
  let cur = src, cw = cur.width, ch = cur.height;
  const draw = (ctx, img, ww, hh) => { ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high'; ctx.drawImage(img, 0, 0, ww, hh); };
  while (cw / tW > 2 || ch / tH > 2) {
    const nw = Math.max(tW, Math.ceil(cw / 2)), nh = Math.max(tH, Math.ceil(ch / 2));
    const c = document.createElement('canvas'); c.width = nw; c.height = nh;
    draw(c.getContext('2d'), cur, nw, nh);
    cur = c; cw = nw; ch = nh;
  }
  if (cw !== tW || ch !== tH) {
    const c = document.createElement('canvas'); c.width = tW; c.height = tH;
    draw(c.getContext('2d'), cur, tW, tH);
    return c;
  }
  return cur;
}

// src: an ImageBitmap or canvas holding the decoded photo. 4080->3072 is a
// small step, so one high-quality draw is enough (stepDownscale halves only
// when the source is more than twice the target).
function encode(src) {
  const s = Math.min(1, SEND_MAX_DIM / Math.max(src.width, src.height));
  const w = Math.round(src.width * s), h = Math.round(src.height * s);
  const c = stepDownscale(src, w, h);
  return new Promise((res, rej) => c.toBlob(b => b ? res({ blob: b, w, h }) : rej(new Error('image encoding failed')), 'image/jpeg', SEND_QUALITY));
}

function toBase64(blob) {
  return new Promise((res, rej) => {
    const fr = new FileReader();
    fr.onload = () => res(String(fr.result).split(',')[1] || '');
    fr.onerror = () => rej(fr.error || new Error('read failed'));
    fr.readAsDataURL(blob);
  });
}

export class ReaderError extends Error {
  constructor(kind, message, retryable) { super(message); this.kind = kind; this.retryable = retryable; }
}

function linkAbort(outer, inner) {
  if (!outer) return () => {};
  const f = () => inner.abort();
  if (outer.aborted) inner.abort(); else outer.addEventListener('abort', f);
  return () => outer.removeEventListener('abort', f);
}

async function geminiOnce(url, base64, section, { model, temperature, signal }) {
  const ctrl = new AbortController();
  const unlink = linkAbort(signal, ctrl);
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, GEMINI_TIMEOUT_MS);
  try {
    let res;
    try {
      res = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ctrl.signal,
        body: JSON.stringify({
          model,
          generationConfig: { temperature, maxOutputTokens: MAX_OUTPUT_TOKENS, thinkingConfig: { thinkingBudget: 0 } },
          contents: [{ parts: [{ text: labelsPrompt(section) }, { inline_data: { mime_type: 'image/jpeg', data: base64 } }] }],
        }),
      });
    } catch (e) {
      if (signal?.aborted) throw new ReaderError('ABORTED', 'cancelled', false);
      if (timedOut) throw new ReaderError('TIMEOUT', 'the reader took too long to answer', true);
      throw new ReaderError('NETWORK', 'could not reach the reader', true);
    }
    if (!res.ok) {
      const t = await res.text().catch(() => '');
      throw new ReaderError(res.status === 429 ? 'QUOTA' : 'HTTP', `reader returned HTTP ${res.status} ${t.slice(0, 120)}`.trim(), res.status >= 500 || res.status === 429);
    }
    const j = await res.json().catch(() => null);
    const cand = j?.candidates?.[0];
    const text = (cand?.content?.parts || []).map(p => p.text || '').join('');
    const finish = cand?.finishReason || '';
    const rows = parseLabels(text);
    if (!rows.length) throw new ReaderError('EMPTY', `reader returned no books (${finish || 'no text'})`, true);
    // A hit output cap means the model looped. The rows before the loop are still
    // usable, but a clean re-read is better when there is time for one.
    return { rows, text, finish, model, capped: finish === 'MAX_TOKENS' };
  } finally { clearTimeout(timer); unlink(); }
}

// One Gemini attempt chain: retries transient failures, switches model on a 429.
async function geminiChain(url, base64, section, signal) {
  let model = MODELS[0], temperature = 0.2, last;
  for (let i = 0; i < 3; i++) {
    try {
      const r = await geminiOnce(url, base64, section, { model, temperature, signal });
      if (r.capped && i < 2) { last = r; temperature = 0.5; continue; }
      return r;
    } catch (e) {
      if (signal?.aborted || !(e instanceof ReaderError) || !e.retryable) throw e;
      last = e;
      if (e.kind === 'QUOTA') model = MODELS[(MODELS.indexOf(model) + 1) % MODELS.length];
      if (e.kind === 'EMPTY') temperature = 0.5;
      await new Promise(r => setTimeout(r, 400 * (i + 1)));
    }
  }
  if (last && last.rows) return last;
  throw last;
}

// One logical read: the chain above, raced by a hedge chain if it is slow.
function geminiRead(url, base64, section, signal) {
  return new Promise((resolve, reject) => {
    const a = new AbortController(), b = new AbortController();
    const unA = linkAbort(signal, a), unB = linkAbort(signal, b);
    let settled = false, running = 1, hedged = false, lastErr;
    const finish = () => { unA(); unB(); };
    const ok = (r, loser) => { if (settled) return; settled = true; loser.abort(); finish(); resolve(r); };
    const bad = e => { lastErr = e; if (--running === 0 && !settled) { settled = true; finish(); reject(lastErr); } };
    const hedge = () => {
      if (settled || hedged) return;
      hedged = true; running++;
      geminiChain(url, base64, section, b.signal).then(r => ok(r, a), bad);
    };
    geminiChain(url, base64, section, a.signal).then(r => ok(r, b), e => { hedge(); bad(e); });
    setTimeout(hedge, HEDGE_MS);
  });
}

async function azureRead(url, blob, signal) {
  const ctrl = new AbortController();
  const unlink = linkAbort(signal, ctrl);
  const timer = setTimeout(() => ctrl.abort(), AZURE_TIMEOUT_MS);
  try {
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: blob, signal: ctrl.signal });
    if (!res.ok) return null;
    const j = await res.json().catch(() => null);
    return j && j.readResult ? j : null;
  } catch { return null; }
  finally { clearTimeout(timer); unlink(); }
}

// Majority vote on what the stickers look like: a shelf of Dewey numbers is
// nonfiction whatever the toggle says (a volunteer on the wrong tab would
// otherwise get every book sorted by the wrong rules).
export function detectSection(rows) {
  const labels = rows.flat().filter(l => l && l !== '?');
  if (labels.length < 4) return null;
  const dewey = labels.filter(l => /^\s*\d{3}/.test(l)).length / labels.length;
  return dewey > 0.6 ? 'nonfiction' : dewey < 0.15 ? 'fiction' : null;
}

// One upload through the fan-out Worker (workers/read-worker.js), which calls
// both engines server-side and runs the retry/hedge policy there.
async function readViaFanout(url, blob, section, signal) {
  const ctrl = new AbortController();
  const unlink = linkAbort(signal, ctrl);
  const timer = setTimeout(() => ctrl.abort(), 45000);
  try {
    let res;
    try {
      res = await fetch(`${url}?section=${section}`, {
        method: 'POST', body: blob, signal: ctrl.signal,
        headers: { 'Content-Type': 'image/jpeg', 'X-Prompt': encodeURIComponent(labelsPrompt(section)) },
      });
    } catch (e) {
      if (signal?.aborted) throw new ReaderError('ABORTED', 'cancelled', false);
      throw new ReaderError('NETWORK', 'could not reach the reader', true);
    }
    const j = await res.json().catch(() => null);
    if (!res.ok || !j || !j.gemini) throw new ReaderError(j?.kind === 'QUOTA' ? 'QUOTA' : 'HTTP', `reader: ${j?.message || j?.error || res.status}`, res.status >= 500);
    const rows = parseLabels(j.gemini.text);
    if (!rows.length) throw new ReaderError('EMPTY', 'reader returned no books', true);
    return { gem: { rows, text: j.gemini.text, model: j.gemini.model }, az: j.azure && j.azure.readResult ? j.azure : null, ms: j.ms || {} };
  } finally { clearTimeout(timer); unlink(); }
}

// Returns { books, section, sentW, sentH, timings, degraded, model, raw, via }.
export async function readShelf(source, section, { readUrl, geminiUrl, azureUrl, signal } = {}) {
  const t0 = performance.now();
  const { blob, w, h } = await encode(source);
  const tEnc = performance.now();
  if (readUrl) {
    try {
      let r = await readViaFanout(readUrl, blob, section, signal);
      let usedSection = section;
      const detected = detectSection(r.gem.rows);
      if (detected && detected !== section) {
        usedSection = detected;
        try { r = await readViaFanout(readUrl, blob, detected, signal); } catch { /* keep the first read */ }
      }
      const fused = fuse(r.gem.rows, r.az || { metadata: { width: w, height: h }, readResult: { blocks: [] } }, usedSection);
      return {
        books: toBooks(fused, usedSection), section: usedSection, sentW: w, sentH: h, degraded: !r.az, model: r.gem.model,
        raw: { text: r.gem.text, az: r.az }, via: 'fanout',
        timings: { encode: Math.round(tEnc - t0), gemini: r.ms.gemini, azure: r.ms.azure, total: Math.round(performance.now() - t0) },
      };
    } catch (e) {
      if (e instanceof ReaderError && e.kind === 'ABORTED') throw e;
      // otherwise fall through to calling the two engines directly
    }
  }
  const base64 = await toBase64(blob);
  let azMs = 0;
  const azP = azureUrl ? azureRead(azureUrl, blob, signal).then(r => { azMs = performance.now() - tEnc; return r; }) : Promise.resolve(null);
  let gem = await geminiRead(geminiUrl, base64, section, signal);
  const gMs = performance.now() - tEnc;
  // wrong tab: read again with the right conventions only when the photo is plainly the other kind
  const detected = detectSection(gem.rows);
  let usedSection = section;
  if (detected && detected !== section) {
    usedSection = detected;
    try { gem = await geminiRead(geminiUrl, base64, detected, signal); } catch { /* keep the first read */ }
  }
  const az = await azP;
  const fused = fuse(gem.rows, az || { metadata: { width: w, height: h }, readResult: { blocks: [] } }, usedSection);
  const books = toBooks(fused, usedSection);
  return {
    books, section: usedSection, sentW: w, sentH: h, degraded: !az, model: gem.model, raw: { text: gem.text, az }, via: 'direct',
    timings: { encode: Math.round(tEnc - t0), gemini: Math.round(gMs), azure: Math.round(azMs), total: Math.round(performance.now() - t0) },
  };
}
