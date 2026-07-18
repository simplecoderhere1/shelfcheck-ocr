// ShelfCheck local OCR engine — PP-OCRv6 det+rec (ONNX) running fully in the
// browser via onnxruntime-web. No network, no quota: models are fetched once
// from this origin and cached in the Cache API.
//
// Pipeline (validated offline in test_items/bench_js_port.py — 96.0% recall
// on the 10-image ground truth, ~2.1s native CPU):
//   1. downscale frame to <=2400 max side (multiples of 32), normalize
//   2. DBNet det -> prob map -> threshold 0.3 -> connected components ->
//      bounding rects dilated by ~0.55*h (approximates DBNet unclip 1.6)
//   3. sticker-size filter (line height 0.4%..3.2% of max dim, w >= 0.5h)
//   4. recognition on full-res crops (48px-high grayscale), CTC decode
//   5. vertical join of stacked sticker lines (call number above cutter,
//      SURNAME above first name) into one label per sticker
//
// Public API (window.LocalOCR):
//   available()        -> bool (WebAssembly present)
//   preload(onProgress)-> Promise<void>  idempotent warm-up (models+sessions)
//   run(sourceCanvas, {signal}) -> Promise<Array<{text, conf, bboxFrac}>>
//      bboxFrac = [x, y, w, h] as fractions of the source canvas.
//   detect(sourceCanvas, {signal}) -> Promise<Array<{bboxFrac}>>
//      detection-only (single cheap det pass, no recognition) — used by the
//      live AR loop to paint "seen, still reading" boxes near-instantly.
(function () {
  'use strict';

  const ORT_DIR = 'vendor/ort/';
  const DET_URL = 'models/det.onnx';
  const REC_URL = 'models/rec.onnx';
  const DICT_URL = 'models/rec_dict.txt';
  const CACHE_NAME = 'shelfcheck-ocr-models-v1';

  const DET_MAX_SIDE = 2048;
  const DET_THRESH = 0.3;
  const DET_BOX_MIN_PROB = 0.5;
  const DET_DILATE = 0.55;      // box expansion, fraction of component height
  const REC_HEIGHT = 48;
  const REC_MAX_WIDTH = 640;
  const REC_MIN_CONF = 0.5;
  const STICK_H_MIN = 0.004;    // of max(imgW, imgH)
  const STICK_H_MAX = 0.032;

  let ortReady = null;          // Promise for ort script load
  let engineReady = null;       // Promise for sessions + dict
  let detSess = null, recSess = null, dictChars = null;
  let allowedClasses = null;    // Int32Array of CTC classes ctcDecode argmaxes over
  let activeEP = 'none';

  function available() {
    return typeof WebAssembly === 'object' && typeof fetch === 'function';
  }

  function loadOrtScript() {
    if (ortReady) return ortReady;
    ortReady = new Promise((resolve, reject) => {
      if (window.ort) return resolve();
      const s = document.createElement('script');
      s.src = ORT_DIR + 'ort.all.min.js';
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('failed to load onnxruntime'));
      document.head.appendChild(s);
    });
    return ortReady;
  }

  // Fetch through the Cache API so the 31MB of models downloads exactly once
  // per device (GitHub Pages serves them same-origin).
  async function cachedFetch(url, onProgress) {
    let cache = null;
    try { cache = await caches.open(CACHE_NAME); } catch { /* private mode */ }
    if (cache) {
      const hit = await cache.match(url);
      if (hit) return hit.arrayBuffer();
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
    if (cache) { try { await cache.put(url, res.clone()); } catch { /* quota */ } }
    if (onProgress && res.headers.get('content-length') && res.body) {
      const total = +res.headers.get('content-length');
      const reader = res.body.getReader();
      const chunks = []; let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length;
        onProgress(url, got, total);
      }
      const buf = new Uint8Array(got); let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }
      return buf.buffer;
    }
    return res.arrayBuffer();
  }

  let webgpuOk = null;
  async function wantWebgpu() {
    // Threaded WASM is the default: it's fast enough (~2-3s/scan) and its
    // latency is predictable. The WebGPU (JSEP) EP compiles shaders per
    // tensor shape — measured multi-second stalls on first inference — so
    // it's opt-in via ?ep=webgpu until it earns trust, and even then only
    // on a real (non-fallback) adapter.
    if (webgpuOk !== null) return webgpuOk;
    try {
      if (new URLSearchParams(location.search).get('ep') !== 'webgpu') return (webgpuOk = false);
      if (!navigator.gpu) return (webgpuOk = false);
      const a = await navigator.gpu.requestAdapter();
      webgpuOk = !!a && !(a.info && a.info.isFallbackAdapter) && !a.isFallbackAdapter;
    } catch { webgpuOk = false; }
    return webgpuOk;
  }

  async function createSession(buf) {
    const eps = [];
    if (await wantWebgpu()) eps.push('webgpu');
    eps.push('wasm');
    for (const ep of eps) {
      try {
        const sess = await ort.InferenceSession.create(buf, {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
        });
        activeEP = ep;
        return sess;
      } catch (e) {
        console.warn(`[localocr] ${ep} EP failed, trying next:`, e.message || e);
      }
    }
    throw new Error('no usable onnxruntime execution provider');
  }

  function preload(onProgress) {
    if (engineReady) return engineReady;
    engineReady = (async () => {
      const t0 = performance.now();
      await loadOrtScript();
      // absolute URL: ort's proxy/thread workers resolve paths against their
      // own blob origin, so a relative path would 404 there
      ort.env.wasm.wasmPaths = new URL(ORT_DIR, location.href).href;
      const threads = (self.crossOriginIsolated && navigator.hardwareConcurrency)
        ? Math.min(4, navigator.hardwareConcurrency) : 1;
      ort.env.wasm.numThreads = threads;
      try { ort.env.wasm.proxy = true; } catch { /* keep main thread */ }
      const [detBuf, recBuf, dictBuf] = await Promise.all([
        cachedFetch(DET_URL, onProgress),
        cachedFetch(REC_URL, onProgress),
        cachedFetch(DICT_URL, onProgress),
      ]);
      dictChars = new TextDecoder('utf-8').decode(dictBuf).split('\n');
      buildCharsetMask();
      [detSess, recSess] = await Promise.all([createSession(detBuf), createSession(recBuf)]);
      console.log(`[perf] localocr: engine ready in ${Math.round(performance.now() - t0)}ms ` +
                  `(ep=${activeEP}, threads=${ort.env.wasm.numThreads}, isolated=${!!self.crossOriginIsolated})`);
    })();
    engineReady.catch(() => { engineReady = null; }); // allow retry after failure
    return engineReady;
  }

  function throwIfAborted(signal) {
    if (signal && signal.aborted) {
      const e = new Error('aborted'); e.name = 'AbortError'; throw e;
    }
  }

  // ---- det ----------------------------------------------------------------

  function detPreprocess(src) {
    const W = src.width, H = src.height;
    const sc = Math.min(1, DET_MAX_SIDE / Math.max(W, H));
    const nw = Math.max(32, (Math.round(W * sc) >> 5) << 5);
    const nh = Math.max(32, (Math.round(H * sc) >> 5) << 5);
    const c = document.createElement('canvas');
    c.width = nw; c.height = nh;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src, 0, 0, W, H, 0, 0, nw, nh);
    const px = ctx.getImageData(0, 0, nw, nh).data;
    const n = nw * nh;
    const data = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      data[i]         = px[o]     / 127.5 - 1;   // R
      data[n + i]     = px[o + 1] / 127.5 - 1;   // G
      data[2 * n + i] = px[o + 2] / 127.5 - 1;   // B
    }
    return { data, nw, nh };
  }

  // Two-pass connected components (4-connectivity) with union-find over the
  // thresholded probability map. Returns dilated boxes in SOURCE pixel coords.
  function detPostprocess(prob, nw, nh, srcW, srcH) {
    const n = nw * nh;
    const lab = new Int32Array(n);          // 0 = background
    const parent = [0];
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    let next = 1;
    for (let y = 0; y < nh; y++) {
      const row = y * nw;
      for (let x = 0; x < nw; x++) {
        const i = row + x;
        if (prob[i] <= DET_THRESH) continue;
        const left = x > 0 ? lab[i - 1] : 0;
        const up = y > 0 ? lab[i - nw] : 0;
        if (!left && !up) { lab[i] = next; parent[next] = next; next++; }
        else if (left && up) {
          const rl = find(left), ru = find(up);
          lab[i] = rl;
          if (rl !== ru) parent[ru] = rl;
        } else lab[i] = left || up;
      }
    }
    const stats = new Map();  // root -> [minx,miny,maxx,maxy,area,maxProb]
    for (let y = 0; y < nh; y++) {
      const row = y * nw;
      for (let x = 0; x < nw; x++) {
        const l = lab[row + x];
        if (!l) continue;
        const r = find(l);
        let s = stats.get(r);
        if (!s) { s = [x, y, x, y, 0, 0]; stats.set(r, s); }
        if (x < s[0]) s[0] = x;
        if (y < s[1]) s[1] = y;
        if (x > s[2]) s[2] = x;
        if (y > s[3]) s[3] = y;
        s[4]++;
        const p = prob[row + x];
        if (p > s[5]) s[5] = p;
      }
    }
    const sx = srcW / nw, sy = srcH / nh;
    const boxes = [];
    for (const s of stats.values()) {
      const w = s[2] - s[0] + 1, h = s[3] - s[1] + 1;
      if (s[4] < 12 || w < 4 || h < 2 || s[5] < DET_BOX_MIN_PROB) continue;
      const ex = h * DET_DILATE;
      boxes.push([
        (s[0] - ex) * sx,
        (s[1] - ex) * sy,
        (w + 2 * ex) * sx,
        (h + 2 * ex) * sy,
      ]);
    }
    return boxes;
  }

  // ---- rec ----------------------------------------------------------------

  // Fixed rec input widths: WebGPU compiles shaders per tensor shape, so
  // arbitrary crop widths would recompile on nearly every line. Crops are
  // drawn at their natural width and right-padded with white (sticker
  // background) up to the bucket width; CTC emits blanks over the padding.
  const REC_WIDTH_BUCKETS = [64, 96, 128, 160, 224, 320, 480, 640];

  function recPreprocess(src, box, allowDark) {
    const [bx, by, bw, bh] = box;
    const x = Math.max(0, Math.round(bx)), y = Math.max(0, Math.round(by));
    const w = Math.min(src.width - x, Math.round(bw));
    const h = Math.min(src.height - y, Math.round(bh));
    if (w < 4 || h < 6) return null;
    const nat = Math.min(REC_MAX_WIDTH, Math.max(16, Math.round(w * REC_HEIGHT / h)));
    const tw = REC_WIDTH_BUCKETS.find(b => b >= nat) || REC_MAX_WIDTH;
    const c = document.createElement('canvas');
    c.width = tw; c.height = REC_HEIGHT;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, tw, REC_HEIGHT);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src, x, y, w, h, 0, 0, nat, REC_HEIGHT);
    const px = ctx.getImageData(0, 0, tw, REC_HEIGHT).data;
    const n = tw * REC_HEIGHT;
    const data = new Float32Array(3 * n);
    let light = 0;
    const natPx = nat * REC_HEIGHT;
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      const lum = 0.299 * px[o] + 0.587 * px[o + 1] + 0.114 * px[o + 2];
      // grayscale replicated to 3 channels — matches the validated pipeline
      const g = lum / 127.5 - 1;
      data[i] = g; data[n + i] = g; data[2 * n + i] = g;
      if (lum >= 120 && (i % tw) < nat) light++;
    }
    // Sticker text sits on a white/light label; spine-title text does not.
    // Rejecting dark-background lines is what keeps junk like publisher
    // names out of the results (precision), at negligible recall cost.
    // The title-reading pass sets allowDark to keep coloured-spine text.
    const lightFrac = light / natPx;
    if (!allowDark && lightFrac < 0.30) return null;
    return { data, tw };
  }

  // The rec dict is the full multilingual PP-OCR charset (~18.7k glyphs: CJK,
  // emoji, Cyrillic, symbols). This app only ever reads English library-spine
  // text (call numbers, author names, English titles), so every non-Latin
  // class is pure downside: it can only (a) steal an argmax win on a noisy
  // glyph and corrupt the read (dragging confidence below the flag bars) or
  // (b) waste decode time. buildCharsetMask precomputes ONE list of the CTC
  // classes worth argmaxing over — the blank, the trailing space class, and
  // the dict entries that are English letters/digits, common punctuation, or
  // accented Latin letters (so author names like BRONTË / GARCÍA survive).
  // ctcDecode then scans ~120 classes per timestep instead of ~18.7k.
  function buildCharsetMask() {
    if (!dictChars) return;
    const punct = new Set(['.', ',', "'", '-', '&', '/', ':', ';', '!', '?', '(', ')', '"', ' ']);
    const isAllowedChar = (ch) => {
      if (!ch || ch.length !== 1) return false;   // '' padding / multi-codepoint emoji
      const code = ch.charCodeAt(0);
      if (code >= 48 && code <= 57) return true;   // 0-9
      if (code >= 65 && code <= 90) return true;   // A-Z
      if (code >= 97 && code <= 122) return true;  // a-z
      if (punct.has(ch)) return true;              // common ASCII punctuation
      // Latin-1 accented letters À(0xC0)..ÿ(0xFF), minus the × and ÷ math signs
      if (code >= 0xC0 && code <= 0xFF && code !== 0xD7 && code !== 0xF7) return true;
      return false;
    };
    const allowed = [0];   // CTC blank (class 0) — always keep
    for (let i = 0; i < dictChars.length; i++) {
      if (isAllowedChar(dictChars[i])) allowed.push(i + 1);   // class i>0 -> dictChars[i-1]
    }
    allowed.push(dictChars.length + 1);   // trailing space class (one past the dict)
    allowedClasses = Int32Array.from(allowed);
    console.log(`[perf] localocr: charset mask ${allowedClasses.length}/${dictChars.length + 2} classes`);
  }

  function ctcDecode(logits, T, C) {
    const allowed = allowedClasses;
    let prev = 0;
    const chars = []; let confSum = 0;
    for (let t = 0; t < T; t++) {
      const base = t * C;
      let best = 0, bestP = -Infinity;
      for (let k = 0; k < allowed.length; k++) {
        const c2 = allowed[k];
        if (c2 >= C) continue;           // guard: space class beyond model output
        const p = logits[base + c2];
        if (p > bestP) { bestP = p; best = c2; }
      }
      if (best !== 0 && best !== prev) {
        const ci = best - 1;
        if (ci < dictChars.length) { chars.push(dictChars[ci]); confSum += bestP; }
        else { chars.push(' '); confSum += bestP; }   // space class (last)
      }
      prev = best;
    }
    const text = chars.join('').trim();
    return { text, conf: chars.length ? confSum / chars.length : 0 };
  }

  // ---- join stacked sticker lines into labels -----------------------------

  function joinLines(recs) {
    const nRec = recs.length;
    const parent = Array.from({ length: nRec }, (_, i) => i);
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    for (let i = 0; i < nRec; i++) {
      const bi = recs[i].box;
      for (let j = 0; j < nRec; j++) {
        if (i === j) continue;
        const bj = recs[j].box;
        const hMax = Math.max(bi[3], bj[3]);
        const dy = (bj[1] + bj[3] / 2) - (bi[1] + bi[3] / 2);
        const xov = Math.min(bi[0] + bi[2], bj[0] + bj[2]) - Math.max(bi[0], bj[0]);
        if (dy > 0.4 * hMax && dy < 2.4 * hMax && xov > 0.4 * Math.min(bi[2], bj[2])) {
          const ri = find(i), rj = find(j);
          if (ri !== rj) parent[rj] = ri;
        }
      }
    }
    const groups = new Map();
    for (let i = 0; i < nRec; i++) {
      const r = find(i);
      if (!groups.has(r)) groups.set(r, []);
      groups.get(r).push(recs[i]);
    }
    const labels = [];
    for (const g of groups.values()) {
      g.sort((a, b) => (a.box[1] + a.box[3] / 2) - (b.box[1] + b.box[3] / 2));
      const text = g.map(r => r.text).join(' ').replace(/\s+/g, ' ').trim();
      if (!text || text.length < 2) continue;
      const x0 = Math.min(...g.map(r => r.box[0]));
      const y0 = Math.min(...g.map(r => r.box[1]));
      const x1 = Math.max(...g.map(r => r.box[0] + r.box[2]));
      const y1 = Math.max(...g.map(r => r.box[1] + r.box[3]));
      const conf = g.reduce((s, r) => s + r.conf, 0) / g.length;
      labels.push({ text, conf, box: [x0, y0, x1 - x0, y1 - y0] });
    }
    return dedupeLabels(labels);
  }

  // Post-join cleanup. Real duplicate copies of a book sit SIDE BY SIDE
  // (boxes don't overlap), so both rules below key on box overlap and never
  // remove them:
  //  1. fuzzy-duplicate: two labels reading (nearly) the same text over the
  //     same spot are one sticker detected twice — keep the higher-conf one.
  //  2. container: a label whose box spans two or more other labels is a det
  //     line that bridged adjacent stickers — the individual reads win.
  function dedupeLabels(labels) {
    const norm = s => s.replace(/[^A-Za-z0-9]/g, '').toUpperCase();
    const xov = (a, b) => Math.min(a.box[0] + a.box[2], b.box[0] + b.box[2]) - Math.max(a.box[0], b.box[0]);
    const yov = (a, b) => Math.min(a.box[1] + a.box[3], b.box[1] + b.box[3]) - Math.max(a.box[1], b.box[1]);
    const keep = labels.slice().sort((a, b) => b.conf - a.conf);
    const dead = new Set();
    for (let i = 0; i < keep.length; i++) {
      if (dead.has(keep[i])) continue;
      const ni = norm(keep[i].text);
      for (let j = i + 1; j < keep.length; j++) {
        if (dead.has(keep[j])) continue;
        if (xov(keep[i], keep[j]) < 0.5 * Math.min(keep[i].box[2], keep[j].box[2])) continue;
        if (yov(keep[i], keep[j]) < 0.4 * Math.min(keep[i].box[3], keep[j].box[3])) continue;
        const nj = norm(keep[j].text);
        const pre = Math.min(ni.length, nj.length, 6);
        if (pre >= 4 && ni.slice(0, pre) === nj.slice(0, pre)) dead.add(keep[j]);
      }
    }
    let out = keep.filter(l => !dead.has(l));
    out = out.filter(l => {
      let spanned = 0;
      for (const o of out) {
        if (o === l || o.box[2] >= l.box[2]) continue;
        if (xov(l, o) > 0.6 * o.box[2] && yov(l, o) > 0.5 * o.box[3]) spanned++;
      }
      return spanned < 2;
    });
    return out;
  }

  // ---- main ---------------------------------------------------------------

  // Sticker-size gate, shared by run() and detect() so both use byte-identical
  // logic: keep boxes whose height is 0.4%..3.2% of the frame's max dimension
  // (the physical size range of a call-number/author sticker at shelf distance)
  // and at least half as wide as tall.
  function stickerFilter(boxes, W, H) {
    const maxDim = Math.max(W, H);
    return boxes.filter(b =>
      b[3] >= STICK_H_MIN * maxDim && b[3] <= STICK_H_MAX * maxDim && b[2] >= b[3] * 0.5);
  }

  // Detection only — one cheap DBNet pass, no recognition. Returns sticker-
  // sized boxes as { bboxFrac:[x,y,w,h] } fractions of the source canvas. The
  // live AR loop calls this every few hundred ms to show grey "seen, still
  // reading" boxes long before the full det+rec+fusion cycle finishes.
  async function detect(src, opts = {}) {
    const { signal } = opts;
    await preload();
    throwIfAborted(signal);
    const t0 = performance.now();
    const W = src.width, H = src.height;
    const { data, nw, nh } = detPreprocess(src);
    throwIfAborted(signal);
    const detOut = await detSess.run({ x: new ort.Tensor('float32', data, [1, 3, nh, nw]) });
    const prob = detOut[detSess.outputNames[0]].data;
    throwIfAborted(signal);
    const stick = stickerFilter(detPostprocess(prob, nw, nh, W, H), W, H);
    console.log(`[perf] localocr: det-only ${Math.round(performance.now() - t0)}ms (${stick.length} boxes)`);
    return stick.map(b => ({ bboxFrac: [b[0] / W, b[1] / H, b[2] / W, b[3] / H] }));
  }

  async function run(src, opts = {}) {
    const { signal } = opts;
    await preload();
    throwIfAborted(signal);
    const t0 = performance.now();
    const W = src.width, H = src.height;

    const { data, nw, nh } = detPreprocess(src);
    throwIfAborted(signal);
    const detOut = await detSess.run({ x: new ort.Tensor('float32', data, [1, 3, nh, nw]) });
    const prob = detOut[detSess.outputNames[0]].data;
    const tDet = performance.now();
    throwIfAborted(signal);

    const allBoxes = detPostprocess(prob, nw, nh, W, H);
    const stick = stickerFilter(allBoxes, W, H);

    // Batch recognition per width bucket: one inference call per bucket
    // instead of one per line — per-call overhead (worker round trips)
    // otherwise dominates rec time on 100+ lines.
    const byWidth = new Map();
    for (const box of stick) {
      const pre = recPreprocess(src, box);
      if (!pre) continue;
      if (!byWidth.has(pre.tw)) byWidth.set(pre.tw, []);
      byWidth.get(pre.tw).push({ box, data: pre.data });
    }
    const recs = [];
    const REC_BATCH = 16;
    for (const [tw, items] of byWidth) {
      for (let s0 = 0; s0 < items.length; s0 += REC_BATCH) {
        throwIfAborted(signal);
        const chunk = items.slice(s0, s0 + REC_BATCH);
        const per = 3 * REC_HEIGHT * tw;
        const batch = new Float32Array(chunk.length * per);
        chunk.forEach((it, i) => batch.set(it.data, i * per));
        const out = await recSess.run({
          x: new ort.Tensor('float32', batch, [chunk.length, 3, REC_HEIGHT, tw]),
        });
        const t = out[recSess.outputNames[0]];
        const [B, T, C] = t.dims;
        for (let i = 0; i < B; i++) {
          const { text, conf } = ctcDecode(t.data.subarray(i * T * C, (i + 1) * T * C), T, C);
          if (text && conf > REC_MIN_CONF) recs.push({ text, conf, box: chunk[i].box });
        }
      }
    }
    const tRec = performance.now();

    const labels = joinLines(recs).map(l => ({
      text: l.text,
      conf: l.conf,
      bboxFrac: [l.box[0] / W, l.box[1] / H, l.box[2] / W, l.box[3] / H],
    }));
    console.log(`[perf] localocr: det ${Math.round(tDet - t0)}ms (${allBoxes.length} boxes, ` +
                `${stick.length} sticker-sized), rec ${Math.round(tRec - tDet)}ms ` +
                `(${recs.length} lines) -> ${labels.length} labels, ep=${activeEP}`);
    return labels;
  }

  // Focused re-read of one sticker at 2x zoom (det + rec on a padded crop).
  // Used to double-check a label before the app accuses its book of being
  // misshelved. Returns { text, conf } or null if nothing readable.
  async function readRegion(src, boxPx) {
    await preload();
    const [bx, by, bw, bh] = boxPx;
    const pad = Math.max(8, bh * 0.3, bw * 0.12);
    const x = Math.max(0, Math.round(bx - pad)), y = Math.max(0, Math.round(by - pad));
    const w = Math.min(src.width - x, Math.round(bw + 2 * pad));
    const h = Math.min(src.height - y, Math.round(bh + 2 * pad));
    if (w < 12 || h < 12) return null;
    const scale = Math.min(3, 1920 / Math.max(w, h));
    const cw = Math.max(32, (Math.round(w * scale) >> 5) << 5);
    const ch = Math.max(32, (Math.round(h * scale) >> 5) << 5);
    const c = document.createElement('canvas');
    c.width = cw; c.height = ch;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src, x, y, w, h, 0, 0, cw, ch);
    const px = ctx.getImageData(0, 0, cw, ch).data;
    const n = cw * ch;
    const data = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      data[i] = px[o] / 127.5 - 1;
      data[n + i] = px[o + 1] / 127.5 - 1;
      data[2 * n + i] = px[o + 2] / 127.5 - 1;
    }
    const detOut = await detSess.run({ x: new ort.Tensor('float32', data, [1, 3, ch, cw]) });
    const prob = detOut[detSess.outputNames[0]].data;
    const lines = detPostprocess(prob, cw, ch, cw, ch)
      .filter(b => b[3] >= 10)
      .sort((a, b) => (a[1] + a[3] / 2) - (b[1] + b[3] / 2));
    const parts = []; let confMin = 1;
    for (const b of lines.slice(0, 4)) {
      // Extend each line box a little rightward: truncations lose TRAILING
      // glyphs to a clipped det box, and rec/CTC reads through trailing
      // background harmlessly. Keep the extension modest so a horizontal
      // neighbor's sticker doesn't bleed into the read.
      const ext = Math.max(12, b[3] * 1.2);
      const x0 = Math.max(0, b[0] - 6);
      const bFull = [x0, b[1], Math.min(cw - x0, b[2] + (b[0] - x0) + ext), b[3]];
      const pre = recPreprocess(c, bFull) || recPreprocess(c, b);
      if (!pre) continue;
      const out = await recSess.run({ x: new ort.Tensor('float32', pre.data, [1, 3, REC_HEIGHT, pre.tw]) });
      const t = out[recSess.outputNames[0]];
      const [, T, C] = t.dims;
      const { text, conf } = ctcDecode(t.data, T, C);
      if (text && conf > REC_MIN_CONF) { parts.push(text); confMin = Math.min(confMin, conf); }
    }
    if (!parts.length) return null;
    return { text: parts.join(' ').replace(/\s+/g, ' ').trim(), conf: confMin };
  }

  // ---- title reading ------------------------------------------------------

  // Rotate a canvas 90 degrees. Spine titles are printed vertically, so the
  // rec model (which expects horizontal lines) needs the strip turned upright.
  function rotate90(srcCanvas, clockwise) {
    const w = srcCanvas.width, h = srcCanvas.height;
    const out = document.createElement('canvas');
    out.width = h; out.height = w;
    const ctx = out.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    if (clockwise) { ctx.translate(h, 0); ctx.rotate(Math.PI / 2); }
    else { ctx.translate(0, w); ctx.rotate(-Math.PI / 2); }
    ctx.drawImage(srcCanvas, 0, 0);
    return out;
  }

  // Full det+rec over an arbitrary canvas, returning recognized lines. Unlike
  // run(), no sticker-size filter — the title-reading pass wants the large
  // spine text run() deliberately discards. allowDark keeps coloured-spine
  // text past the light-background filter.
  async function recognizeLines(canvas, allowDark, maxLines) {
    const { data, nw, nh } = detPreprocess(canvas);
    const detOut = await detSess.run({ x: new ort.Tensor('float32', data, [1, 3, nh, nw]) });
    const prob = detOut[detSess.outputNames[0]].data;
    const boxes = detPostprocess(prob, nw, nh, canvas.width, canvas.height)
      .filter(b => b[3] >= 8 && b[2] >= 8)
      .sort((a, b) => b[2] * b[3] - a[2] * a[3])   // largest first: title text
      .slice(0, maxLines || 8);
    const recs = [];
    for (const b of boxes) {
      const pre = recPreprocess(canvas, b, allowDark);
      if (!pre) continue;
      const out = await recSess.run({ x: new ort.Tensor('float32', pre.data, [1, 3, REC_HEIGHT, pre.tw]) });
      const t = out[recSess.outputNames[0]];
      const [, T, C] = t.dims;
      const { text, conf } = ctcDecode(t.data, T, C);
      if (text && conf > REC_MIN_CONF) recs.push({ text, conf, box: b });
    }
    return recs;
  }

  // Read the title/author text off the spine above a sticker. boxPx is the
  // sticker bbox in source pixels; the spine strip is that x-range extended
  // upward. Tries both 90-degree rotations and keeps whichever orientation
  // yields more confident characters. Returns { title, conf } or null.
  async function readTitle(src, boxPx) {
    await preload();
    const [sx, sy, sw, sh] = boxPx;
    const padX = sw * 0.25;
    const x = Math.max(0, Math.round(sx - padX));
    const w = Math.min(src.width - x, Math.round(sw + 2 * padX));
    // the spine rises from the sticker; cap the strip so it can't run off into
    // the shelf above (extra background just yields no det boxes anyway)
    const upH = Math.min(sy, Math.max(sw * 5, sh * 12));
    const y = Math.max(0, Math.round(sy - upH));
    const h = Math.round(sy - y);
    if (w < 12 || h < 40) return null;
    const scale = Math.min(3, 1400 / Math.max(w, h));
    const cw0 = Math.max(16, Math.round(w * scale));
    const ch0 = Math.max(16, Math.round(h * scale));
    const crop = document.createElement('canvas');
    crop.width = cw0; crop.height = ch0;
    const cctx = crop.getContext('2d', { willReadFrequently: true });
    cctx.imageSmoothingQuality = 'high';
    cctx.drawImage(src, x, y, w, h, 0, 0, cw0, ch0);
    // English spine text conventionally reads bottom-to-top (rotate clockwise
    // to make it upright), so try that first and only pay for the other
    // orientation when the first read is weak — halves the average cost.
    let best = null;
    for (const clockwise of [true, false]) {
      const rot = rotate90(crop, clockwise);
      const recs = await recognizeLines(rot, true, 5);
      if (recs.length) {
        recs.sort((a, b) => (a.box[1] + a.box[3] / 2) - (b.box[1] + b.box[3] / 2) || a.box[0] - b.box[0]);
        const text = recs.map(r => r.text).join(' ').replace(/\s+/g, ' ').trim();
        const conf = recs.reduce((s, r) => s + r.conf, 0) / recs.length;
        const score = conf * text.replace(/[^A-Za-z0-9]/g, '').length;
        if (!best || score > best.score) best = { title: text, conf, score };
      }
      // a confident, several-character read is almost certainly the right way
      // up — skip the second orientation
      if (best && best.conf >= 0.85 && best.score >= 8) break;
    }
    if (!best || best.title.replace(/[^A-Za-z0-9]/g, '').length < 3) return null;
    return { title: best.title, conf: best.conf };
  }

  window.LocalOCR = { available, preload, run, detect, readRegion, readTitle, get ep() { return activeEP; } };
})();
