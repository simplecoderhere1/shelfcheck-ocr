// Shelf reading: one vision-model pass (Gemini) for WHICH books are on each
// row and in what order, fused with one OCR pass (Azure Read) for WHERE each
// sticker is and exactly WHAT it prints.
//
// Why two engines. The vision model reads a whole shelf well and knows the
// sticker conventions, but it conforms outliers to their neighbourhood: on the
// audit shelves it read "WERTH, Janet" as "WESLEY, Janet" and "WEISBERGER,
// Lauren" as "WEST, Lauren" -- exactly the misfiled books this app exists to
// find. It also invents evenly spaced coordinates when asked for positions in
// a compact format. Azure copies characters literally and gives exact word
// boxes, but on its own cannot tell a sticker from cover text or group words
// into books reliably (that assembly was the old app's weak point). Each
// covers the other's blind spot: the model's ordered list is aligned to
// Azure's sticker words; agreement anchors the row, and where they disagree
// on a sticker whose other half (given name / call number) Azure confirms,
// Azure's literal text wins.
//
// Shared by the page (reader.js) and the offline evaluation harness, so the
// numbers measured are the code that ships.

export function labelsPrompt(section) {
  const conv = section === 'fiction'
    ? 'the author\'s SURNAME (sometimes cut short) then given name(s), e.g. "WALLACE, David Foster" or "VONNE, Kurt"'
    : 'a Dewey call number then a cutter, e.g. "635.04 HEM" (sometimes a year or volume below)';
  return `Library book spines standing on shelf rows. Each book has a small white sticker printing ${conv}.
Transcribe every sticker, row by row, top row first. Output plain text only, one line per shelf row:
ROW: <sticker of book 1> | <sticker of book 2> | ... (that row's books left to right)
Rules:
- Each book's WHOLE sticker as one item (number and letters together).
- Only rows whose stickers are visible; skip a row cut off by the photo edge.
- A book displayed face-out (its front cover toward the camera, often past a bookend at a row's end) or lying flat is not in the row: skip it. List only spines standing in the row.
- Copy the characters exactly as printed. Never pad or complete a number or name to match neighbouring books.
- Write ? for each character you cannot read, or just ? for a book whose sticker is unreadable (blurred, turned away, half cut off).`;
}

export function parseLabels(text) {
  const rows = [];
  for (const raw of String(text || '').split(/\r?\n/)) {
    let line = raw.trim().replace(/^```\w*|```$/g, '').trim();
    if (!line) continue;
    line = line.replace(/^ROW\s*\d*\s*[:.-]?\s*/i, '');
    const items = line.split('|').map(s => s.trim().replace(/\s+/g, ' ')
      .replace(/^(\d{3})\s*\.\s+(\d)/, '$1.$2')).filter(Boolean);   // some stickers print "364. 1066"
    if (items.length) rows.push(items);
  }
  for (const r of rows) for (let i = 0; i < r.length - 1; i++)
    if (/^\d{3}(\.[\d?]*)?$/.test(r[i]) && !/^\d/.test(r[i + 1]) && r[i + 1] !== '?') { r[i] += ' ' + r[i + 1]; r.splice(i + 1, 1); }
  return rows;
}

const U = s => String(s || '').toUpperCase();
const letters = s => U(s).replace(/[^A-Z]/g, '');
// Every word with its orientation (spine titles/authors run vertically).
export function allWords(az) {
  const out = [];
  for (const b of az?.readResult?.blocks || []) for (const l of b.lines) for (const w of l.words) {
    const p = w.boundingPolygon; if (!p || p.length < 4) continue;
    const dx = p[1].x - p[0].x, dy = p[1].y - p[0].y;
    const xs = p.map(q => q.x), ys = p.map(q => q.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    out.push({ text: w.text, T: U(w.text).replace(/[^A-Z0-9.]/g, ''), conf: w.confidence, x0, x1, y0, y1, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, vertical: Math.abs(dx) < Math.abs(dy) });
  }
  return out;
}

export function azureWords(az) {
  const out = [];
  for (const b of az?.readResult?.blocks || []) for (const l of b.lines) for (const w of l.words) {
    const p = w.boundingPolygon; if (!p || p.length < 4) continue;
    const dx = p[1].x - p[0].x, dy = p[1].y - p[0].y;
    if (Math.abs(dx) < Math.abs(dy)) continue;
    const xs = p.map(q => q.x), ys = p.map(q => q.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    out.push({ text: w.text, T: U(w.text).replace(/[^A-Z0-9.]/g, ''), conf: w.confidence, x0, x1, y0, y1, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, h: y1 - y0 });
  }
  return out;
}

const median = a => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };

function ed(a, b) {
  const m = a.length, n = b.length; if (!m) return n; if (!n) return m;
  let prev = Array.from({ length: n + 1 }, (_, j) => j);
  for (let i = 1; i <= m; i++) { const cur = [i]; for (let j = 1; j <= n; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)); prev = cur; }
  return prev[n];
}

// Parse a Gemini label into its sort-relevant parts.
export function keyOf(label, section) {
  const s = String(label || '').trim();
  if (section === 'fiction') {
    let sur, giv;
    if (s.includes(',')) { [sur, giv] = [s.slice(0, s.indexOf(',')), s.slice(s.indexOf(',') + 1)]; }
    else { const w = s.split(/\s+/); let k = 0; while (k < w.length && w[k] === U(w[k]) && /[A-Z]/.test(w[k])) k++; if (!k) k = 1; sur = w.slice(0, k).join(' '); giv = w.slice(k).join(' '); }
    return { tok: letters(sur), sur: sur.trim(), giv: (giv || '').trim() };
  }
  const m = /^\s*(\d{3}(?:\.[\d?]*)?)\s*(.*)$/.exec(s);
  if (!m) return { tok: letters(s.split(/\s+/)[0]), num: null, rest: s };
  const cut = (U(m[2]).match(/[A-Z?]{2,}/) || [''])[0];
  return { tok: cut.includes('?') ? '' : cut, num: m[1], rest: m[2] };
}

// Azure sticker tokens: the first line of each sticker (surname / cutter).
function stickerTokens(words, section) {
  let toks = [];
  for (const w of words) {
    const L = letters(w.T);
    if (section === 'fiction') {
      const raw = w.text.replace(/[^A-Za-z]/g, '');
      if (L.length < 2 || raw !== raw.toUpperCase()) continue;       // surname line is capitalised
    } else {
      if (!/^[A-Z]{2,4}\d{0,4}$/.test(w.T.replace(/[.,]/g, ''))) continue; // cutter
    }
    toks.push({ ...w, L });
  }
  // Azure often returns two overlapping reads of ONE sticker ("THOMAS" 0.09 and
  // "THOMAS." 0.92 at nearly the same box); keep the more confident one, or the
  // spare copy occupies a slot that belongs to a different book.
  toks.sort((a, b) => b.conf - a.conf);
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]; if (t.dup) continue;
    for (let j = i + 1; j < toks.length; j++) {
      const o = toks[j]; if (o.dup) continue;
      const ox = Math.min(t.x1, o.x1) - Math.max(t.x0, o.x0), oy = Math.min(t.y1, o.y1) - Math.max(t.y0, o.y0);
      if (ox > 0.5 * Math.min(t.x1 - t.x0, o.x1 - o.x0) && oy > 0.3 * Math.min(t.h, o.h)) o.dup = true;
    }
  }
  toks = toks.filter(t => !t.dup);
  for (const t of toks) t.complete = /,\s*$/.test(t.text);
  // drop tokens that sit directly under another token (second sticker line, e.g. "JN" under "WELSH,")
  return toks.filter(t => !toks.some(o => o !== t && t.cx > o.x0 && t.cx < o.x1 && t.cy - o.cy > t.h * 0.8 && t.cy - o.cy < t.h * 2.2));
}

function similar(tok, L, section, complete = false) {
  if (!tok || !L) return 0;
  if (tok === L) return 3;
  if (section === 'fiction') {
    // prefix: the model's name may be the sticker's truncation (VONNE / VONNEGUT) or
    // Azure's word may be cut short, but not when Azure saw the name end in a comma
    if (tok.length >= 4 && L.startsWith(tok)) return 3;
    if (L.length >= 4 && tok.startsWith(L) && !complete) return 3;
    if (Math.min(tok.length, L.length) >= 5 && Math.abs(tok.length - L.length) <= 1 && ed(tok, L) <= 1) return 2;
    return 0;
  }
  if (L.length === tok.length + 1 && L.startsWith(tok)) return 3;
  return L.length === tok.length && ed(tok, L) <= 1 ? 2 : 0;
}

const GIVEN_STOP = new Set(['THE', 'A', 'AN', 'OF', 'AND', 'NOVEL']);
function givenAgree(a, b) {
  const x = letters(a), y = letters(b);
  if (!x || !y) return 0;
  if (x.length >= 3 && y.length >= 3) return x.slice(0, 3) === y.slice(0, 3) ? 1 : -1;
  return x === y || x[0] === y[0] ? 1 : -1;
}
// Book (model read) vs sticker token (Azure). On a fiction shelf the given name
// breaks the ties that a long single-author run creates (a dozen WALKERs).
function scoreBT(key, t, section) {
  let sc = similar(key.tok, t.L, section, t.complete);
  if (!sc) return 0;
  if (section === 'fiction' && t.giv && key.giv) sc = Math.max(1, sc + (givenAgree(key.giv, t.giv) > 0 ? 1 : -2));
  return sc;
}

function fitRowLine(hits, tol) {
  let best = null;
  const H = hits.slice(0, 300);
  for (let a = 0; a < H.length; a++) for (let b = a + 1; b < H.length; b++) {
    const p = H[a], q = H[b];
    if (p.i === q.i || (p.i < q.i) !== (p.w.cx < q.w.cx) || Math.abs(q.w.cx - p.w.cx) < 1) continue;
    const slope = (q.w.cy - p.w.cy) / (q.w.cx - p.w.cx);
    if (Math.abs(slope) > 0.25) continue;
    // count distinct WORDS on the line, not book indices: one "THOMAS" word matches
    // every THOMAS book in the row and would make any line through it look best
    const set = new Set();
    for (const h of hits) if (Math.abs(h.w.cy - (p.w.cy + slope * (h.w.cx - p.w.cx))) < tol) set.add(h.w);
    if (!best || set.size > best.n) best = { n: set.size, x0: p.w.cx, y0: p.w.cy, slope };
  }
  if (!best && hits.length) best = { n: 1, x0: hits[0].w.cx, y0: hits[0].w.cy, slope: 0 };
  if (!best) return null;
  const inl = hits.filter(h => Math.abs(h.w.cy - (best.y0 + best.slope * (h.w.cx - best.x0))) < tol);
  if (inl.length >= 3) {
    const mx = inl.reduce((s, h) => s + h.w.cx, 0) / inl.length, my = inl.reduce((s, h) => s + h.w.cy, 0) / inl.length;
    let sxx = 0, sxy = 0; for (const h of inl) { sxx += (h.w.cx - mx) ** 2; sxy += (h.w.cx - mx) * (h.w.cy - my); }
    const slope = sxx ? sxy / sxx : 0;
    if (Math.abs(slope) <= 0.25) best = { n: best.n, x0: mx, y0: my, slope };
  }
  return { ...best, yAt: x => best.y0 + best.slope * (x - best.x0) };
}

// The physical sticker rows, found from Azure's words alone: stickers on one shelf
// sit on one (tilted, slightly curved) line. Peel off the best-supported line,
// refit it through its words, repeat. Cover text scattered over the spines rarely
// lines up four-deep, so what remains are the shelves.
function detectBands(toks, hMed) {
  let rest = toks.slice();
  const bands = [];
  const tol = hMed * 1.3;
  while (rest.length >= 4 && bands.length < 8) {
    let best = null;
    const P = rest.length > 220 ? rest.filter((_, i) => i % Math.ceil(rest.length / 220) === 0) : rest;
    for (let a = 0; a < P.length; a++) for (let b = a + 1; b < P.length; b++) {
      const p = P[a], q = P[b];
      const dx = q.cx - p.cx; if (Math.abs(dx) < hMed * 2) continue;
      const slope = (q.cy - p.cy) / dx; if (Math.abs(slope) > 0.25) continue;
      let n = 0; for (const t of rest) if (Math.abs(t.cy - (p.cy + slope * (t.cx - p.cx))) < tol) n++;
      if (!best || n > best.n) best = { n, p, slope };
    }
    if (!best || best.n < 4) break;
    let inl = rest.filter(t => Math.abs(t.cy - (best.p.cy + best.slope * (t.cx - best.p.cx))) < tol);
    let line = fitPoly(inl, inl.length >= 10 ? 2 : 1) || { yAt: x => best.p.cy + best.slope * (x - best.p.cx) };
    inl = rest.filter(t => Math.abs(t.cy - line.yAt(t.cx)) < tol * 1.2);
    line = fitPoly(inl, inl.length >= 10 ? 2 : 1) || line;
    bands.push({ line, toks: inl });
    const set = new Set(inl); rest = rest.filter(t => !set.has(t));
  }
  return bands;
}

// Least-squares polynomial y(x) through sticker words (degree 1 or 2).
function fitPoly(pts, deg) {
  const n = pts.length; if (n < deg + 2) return null;
  const mx = pts.reduce((a, p) => a + p.cx, 0) / n;
  const sx = Math.max(1, Math.sqrt(pts.reduce((a, p) => a + (p.cx - mx) ** 2, 0) / n));
  const X = pts.map(p => (p.cx - mx) / sx), Y = pts.map(p => p.cy);
  const m = deg + 1, A = Array.from({ length: m }, () => new Array(m + 1).fill(0));
  for (let k = 0; k < n; k++) {
    const pw = [1, X[k], X[k] * X[k]];
    for (let r = 0; r < m; r++) { for (let c = 0; c < m; c++) A[r][c] += pw[r] * pw[c]; A[r][m] += pw[r] * Y[k]; }
  }
  for (let c = 0; c < m; c++) {
    let piv = c; for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    if (Math.abs(A[c][c]) < 1e-9) return null;
    for (let r = 0; r < m; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let k = c; k <= m; k++) A[r][k] -= f * A[c][k]; }
  }
  const co = A.map((r, i) => r[m] / r[i]);
  // evaluate only inside the fitted span: a quadratic extrapolated past its last
  // anchor bends away fast and would sweep into the next row
  const lo = Math.min(...pts.map(p => p.cx)), hi = Math.max(...pts.map(p => p.cx));
  const f = x => { const t = (x - mx) / sx; return co[0] + (co[1] || 0) * t + (co[2] || 0) * t * t; };
  const sl = deg === 1 ? (co[1] || 0) / sx : 0;
  return { yAt: x => x < lo ? f(lo) + sl * (x - lo) : x > hi ? f(hi) + sl * (x - hi) : f(x) };
}

function lcsAlign(n, m, score) {
  const dp = Array.from({ length: n + 1 }, () => new Int16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) {
    const s = score(i, j); dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1], s ? s + dp[i + 1][j + 1] : 0);
  }
  const pairs = []; let i = 0, j = 0;
  while (i < n && j < m) {
    const s = score(i, j);
    if (s && dp[i][j] === s + dp[i + 1][j + 1]) { pairs.push([i, j, s]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) i++; else j++;
  }
  return pairs;
}

// Words of the same sticker below a token (given name / year).
function belowWords(words, t) {
  return words.filter(q => q !== t && q.cy > t.cy + t.h * 0.4 && q.cy < t.cy + t.h * 2.4 && q.x0 < t.x1 + t.h * 0.5 && q.x1 > t.x0 - t.h * 0.5)
    .sort((a, b) => a.cy - b.cy);
}
// The given-name line of a fiction sticker: directly beneath the surname, not
// itself a capitalised surname (initials like "JN" / "A.J." are allowed).
function givenBelow(words, t) {
  const w = t.x1 - t.x0;
  return words.filter(q => {
    if (q === t) return false;
    const raw = q.text.replace(/[^A-Za-z]/g, '');
    if (!raw) return false;
    if (raw === raw.toUpperCase() && raw.length > 3) return false;
    return q.cx > t.x0 - w * 0.3 && q.cx < t.x1 + w * 0.3 && q.cy > t.cy + t.h * 0.5 && q.cy < t.cy + t.h * 2.6;
  }).sort((a, b) => a.cy - b.cy)[0] || null;
}
function numberAbove(words, t) {
  const head = words.filter(q => /^\d{3}(\.\d*)?$/.test(q.T) && q.cy < t.cy && t.cy - q.cy < t.h * 3 && q.x0 < t.x1 + t.h && q.x1 > t.x0 - t.h)
    .sort((a, b) => (t.cy - a.cy) - (t.cy - b.cy))[0];
  if (!head) return null;
  // Azure splits "616.8527" into "616." and "8527" on the same line: join the digits
  let T = head.T, conf = head.conf, x1 = head.x1;
  if (/\.$/.test(T) || !T.includes('.')) {
    const tail = words.filter(q => q !== head && /^\.?\d{1,5}$/.test(q.T) && Math.abs(q.cy - head.cy) < head.h * 0.5 &&
      q.x0 > head.x1 - head.h * 0.3 && q.x0 - x1 < head.h * 1.2).sort((a, b) => a.x0 - b.x0)[0];
    if (tail) { T = (T.endsWith('.') || tail.T.startsWith('.') ? T : T + '.') + tail.T; T = T.replace('..', '.'); conf = Math.min(conf, tail.conf); }
  }
  return { ...head, T, conf };
}

export function fuse(gemRows, az, section, opts = {}) {
  const W = az?.metadata?.width || 1, H = az?.metadata?.height || 1;
  const words = azureWords(az);
  const toks = stickerTokens(words, section);
  if (section === 'fiction') for (const t of toks) { const g = givenBelow(words, t); t.giv = g ? g.text : ''; t.givConf = g ? g.conf : 0; }
  // spine text (author names printed vertically on the spine) -- an independent third read
  const spine = allWords(az).filter(w => w.vertical && letters(w.T).length >= 4);
  const hMed = median(toks.map(w => w.h)) || median(words.map(w => w.h)) || H * 0.01;
  const rows = [];
  for (const labels of gemRows) {
    const readable = labels.filter(l => letters(l).length >= 2).length;
    if (readable < Math.max(2, labels.length * 0.4)) continue;         // a row cut off by the frame
    const books = labels.map(label => ({ label, key: keyOf(label, section) }));
    rows.push({ books, line: null });
  }
  const spineAt = (cx, cy, pitch) => spine.filter(v => Math.abs(v.cx - cx) < pitch * 0.6 && v.cy < cy).map(v => letters(v.T));
  const nameMatch = (v, name) => !!name && name.length >= 4 && (v === name || v.startsWith(name) || (v.length >= 5 && name.startsWith(v)));
  const used = new Set();
  // Assign the model's rows (top to bottom) to the physical sticker bands (top to
  // bottom) so the total text agreement is maximal. Two rows by one author
  // (WILLIAMS over WILLIAMS) no longer compete for the same band, and a model row
  // with no band is a shelf whose stickers are not in the frame.
  const bands = detectBands(toks, hMed).sort((a, b) => a.line.yAt(W / 2) - b.line.yAt(W / 2));
  const S = rows.map(r => bands.map(bd => {
    const c = bd.toks.slice().sort((a, b) => a.cx - b.cx);
    return lcsAlign(r.books.length, c.length, (i, j) => scoreBT(r.books[i].key, c[j], section)).reduce((a, q) => a + q[2], 0);
  }));
  const R = rows.length, K = bands.length;
  const f = Array.from({ length: R + 1 }, () => new Array(K + 1).fill(0));
  for (let r = 1; r <= R; r++) for (let k = 1; k <= K; k++)
    f[r][k] = Math.max(f[r - 1][k], f[r][k - 1], f[r - 1][k - 1] + (S[r - 1][k - 1] >= 6 ? S[r - 1][k - 1] : -1e9));
  for (let r = R, k = K; r > 0 && k > 0;) {
    if (f[r][k] === f[r - 1][k]) r--;
    else if (f[r][k] === f[r][k - 1]) k--;
    else { rows[r - 1].line = bands[k - 1].line; rows[r - 1].band = k - 1; r--; k--; }
  }
  for (const row of rows) {
    const { books } = row;
    const line = row.line;
    if (opts.debug) console.log('row', rows.indexOf(row), 'band', row.band, 'scores', S[rows.indexOf(row)]);
    if (!line) continue;
    const candsFor = ln => toks.filter(w => !used.has(w) && Math.abs(w.cy - ln.yAt(w.cx)) < hMed * 3.5).sort((a, b) => a.cx - b.cx);
    let cands = candsFor(line);
    // pass 1: anchors (agreeing text, order-preserving)
    let pairs = lcsAlign(books.length, cands.length, (i, j) => scoreBT(books[i].key, cands[j], section));
    // Shelves photograph tilted and slightly curved: refit the row through the
    // anchors (quadratic once they span the row) and align again, so the far end
    // of a long row is not lost to a line fitted on its near end.
    for (let pass = 0; pass < 2 && pairs.length >= 4; pass++) {
      const pts = pairs.map(([, j]) => cands[j]);
      const ln2 = fitPoly(pts, pts.length >= 8 ? 2 : 1);
      if (!ln2) break;
      const c2 = candsFor(ln2);
      const p2 = lcsAlign(books.length, c2.length, (i, j) => scoreBT(books[i].key, c2[j], section));
      if (p2.reduce((a, q) => a + q[2], 0) <= pairs.reduce((a, q) => a + q[2], 0)) break;
      cands = c2; pairs = p2; row.line = ln2;
    }
    const pairOf = new Map(pairs.map(([i, j, s]) => [i, { j, s }]));
    const candUsed = new Set(pairs.map(p => p[1]));
    for (const [i, j] of pairs) Object.assign(books[i], { tokW: cands[j], agree: similar(books[i].key.tok, cands[j].L, section, cands[j].complete) >= 3 ? 'exact' : 'fuzzy', cx: cands[j].cx });
    // pitch from anchors
    const anc = pairs.map(([i, j]) => [i, cands[j].cx]);
    const gaps = []; for (let k = 1; k < anc.length; k++) gaps.push((anc[k][1] - anc[k - 1][1]) / (anc[k][0] - anc[k - 1][0]));
    const pitch = median(gaps.filter(g => g > 0)) || W / Math.max(10, books.length);
    // pass 2: reconcile each gap between anchors
    const bounds = [[-1, -Infinity], ...anc, [books.length, Infinity]];
    const inserts = [];
    for (let k = 0; k + 1 < bounds.length; k++) {
      const [iL, xL] = bounds[k], [iR, xR] = bounds[k + 1];
      const gb = []; for (let i = iL + 1; i < iR; i++) gb.push(i);
      const lo = isFinite(xL) ? xL + pitch * 0.3 : -Infinity, hi = isFinite(xR) ? xR - pitch * 0.3 : Infinity;
      const gc = cands.map((c, j) => j).filter(j => !candUsed.has(j) && cands[j].cx > lo && cands[j].cx < hi);
      // expected x for gap books
      const expX = gb.map((i, n) => isFinite(xL) && isFinite(xR) ? xL + (xR - xL) * (n + 1) / (gb.length + 1)
        : isFinite(xL) ? xL + pitch * (n + 1) : isFinite(xR) ? xR - pitch * (gb.length - n) : (n + 0.5) * W / gb.length);
      if (gb.length && gc.length) {
        // order-preserving pairing by position (small DP on |dx|)
        const n = gb.length, m = gc.length, cost = (a, b) => Math.abs(expX[a] - cands[gc[b]].cx) / pitch;
        const dp = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(Infinity)); const ch = Array.from({ length: n + 1 }, () => new Int8Array(m + 1));
        for (let a = 0; a <= n; a++) for (let b = 0; b <= m; b++) {
          if (!a && !b) { dp[0][0] = 0; continue; }
          const opts = [];
          if (a && b && cost(a - 1, b - 1) < 0.75) opts.push([dp[a - 1][b - 1] + cost(a - 1, b - 1) - 1, 1]);
          if (a) opts.push([dp[a - 1][b], 2]);
          if (b) opts.push([dp[a][b - 1], 3]);
          for (const [v, c] of opts) if (v < dp[a][b]) { dp[a][b] = v; ch[a][b] = c; }
        }
        let a = n, b = m; const pr = [];
        while (a || b) { const c = ch[a][b]; if (c === 1) { pr.push([a - 1, b - 1]); a--; b--; } else if (c === 2) a--; else b--; }
        for (const [ga, gcb] of pr) {
          const i = gb[ga], j = gc[gcb], w = cands[j];
          candUsed.add(j);
          Object.assign(books[i], { tokW: w, agree: 'disagree', cx: w.cx });
        }
      }
      // leftover Azure tokens in the gap: books the model skipped?
      for (const j of gc) {
        if (candUsed.has(j)) continue;
        const w = cands[j];
        const below = belowWords(words, w);
        // Only a nonfiction sticker has an unmistakable signature (a Dewey number
        // stacked over a cutter). Fiction "extra" surname words were measured to be
        // re-reads of a neighbour's sticker (THRAS beside THRASHER), so never inserted.
        const nbT = [books[iL]?.key.tok, books[iR]?.key.tok].filter(Boolean);
        const dupOfNeighbour = nbT.some(t => t === w.L || t.startsWith(w.L) || w.L.startsWith(t));
        const looksSticker = section !== 'fiction' && !!numberAbove(words, w) && w.conf >= 0.9 && !dupOfNeighbour;
        if (!looksSticker || opts.insert === false) continue;
        candUsed.add(j);
        inserts.push({ afterIdx: iL, w, below });
      }
    }
    // Relocation repair. The model does not only rewrite a misfiled book's text, it
    // can MOVE it in its list to where it belongs: a TATA stranded after the TAWADA
    // run came back as an eighth TATA at the start of the row, and the stranded
    // sticker vanished from the model's order. Azure still sees that sticker where
    // it physically is. When a leftover Azure sticker carries exactly the name of a
    // model book that found no partner in order, the book goes where Azure saw it.
    {
      const leftovers = cands.filter((c, j) => !candUsed.has(j) && c.conf >= 0.75 && c.L.length >= 3 &&
        !(c.x0 < W * 0.01 || c.x1 > W * 0.99));
      for (const c of leftovers) {
        const strongSig = c.complete || !!c.giv || c.L.length >= 5 || (section !== 'fiction' && !!numberAbove(words, c));
        let best = -1, bestD = Infinity;
        books.forEach((b, i) => {
          if (b.tokW || !b.key.tok) return;
          const exact = b.key.tok === c.L;
          if (!exact && !(strongSig && similar(b.key.tok, c.L, section, c.complete) >= 3)) return;
          if (section === 'fiction' && c.giv && b.key.giv && givenAgree(b.key.giv, c.giv) < 0) return;
          // prefer the unanchored copy nearest the sticker's slot
          const at = books.findIndex(o => o.cx != null && o.cx > c.cx);
          const d = Math.abs((at < 0 ? books.length : at) - i);
          if (d < bestD) { bestD = d; best = i; }
        });
        if (best < 0 || bestD < 2) continue;          // nearby: pass 2 already had its chance
        const [m] = books.splice(best, 1);
        Object.assign(m, { tokW: c, agree: 'exact', cx: c.cx, relocated: true });
        const at = books.findIndex(o => o.cx != null && o.cx > c.cx);
        books.splice(at < 0 ? books.length : at, 0, m);
        candUsed.add(cands.indexOf(c));
      }
    }
    // A run of identical model labels that swallowed a different sticker: the model
    // conforms an outlier to its neighbours ("THOMAS, Jodi" x14 where two are
    // "THORPE, Rufi"). Give each leftover strong Azure sticker inside the run one of
    // the run's unanchored members; they are interchangeable, so the order holds.
    {
      const leftovers = cands.filter((c, j) => !candUsed.has(j) && c.conf >= 0.6 && c.L.length >= 3)
        .filter(c => section === 'fiction' ? !!givenBelow(words, c) : !!numberAbove(words, c));
      for (const c of leftovers) {
        const anchored = books.map((b, i) => [b, i]).filter(([b]) => b.tokW && b.cx != null);
        const L = anchored.filter(([b]) => b.cx < c.cx).pop(), R = anchored.find(([b]) => b.cx > c.cx);
        if (!L || !R || U(L[0].label) !== U(R[0].label)) continue;
        if (similar(L[0].key.tok, c.L, section, c.complete)) continue;
        const lab = U(L[0].label);
        let s0 = L[1]; while (s0 > 0 && U(books[s0 - 1].label) === lab) s0--;
        let e0 = R[1]; while (e0 < books.length - 1 && U(books[e0 + 1].label) === lab) e0++;
        const spare = []; for (let i = s0; i <= e0; i++) if (!books[i].tokW) spare.push(i);
        if (c.x0 < W * 0.01 || c.x1 > W * 0.99) continue;
        if (!spare.length) {
          // The model listed the run without this book at all. A complete, different
          // sticker (its own given name) inside the run is a book, and it is exactly the
          // kind the model hides; insert it unless the spine above says otherwise.
          if (section !== 'fiction' || c.conf < 0.7 || !c.giv) continue;
          if (givenAgree(c.giv, L[0].key.giv) > 0) continue;
          const sp = spineAt(c.cx, c.cy, pitch);
          if (sp.some(v => nameMatch(v, L[0].key.tok))) continue;
          const label = `${c.text.replace(/[^A-Za-z' -]+$/, '')}, ${c.giv}`;
          const nbk = { label, key: keyOf(label, section), tokW: c, cx: c.cx, agree: 'azonly', inserted: true };
          const at = books.findIndex(b => b.cx != null && b.cx > c.cx);
          books.splice(at < 0 ? books.length : at, 0, nbk);
          candUsed.add(cands.indexOf(c));
          continue;
        }
        const [m] = books.splice(spare[spare.length - 1], 1);
        Object.assign(m, { tokW: c, agree: 'disagree', cx: c.cx, runSwap: true });
        const at = books.findIndex(b => b.cx != null && b.cx > c.cx);
        books.splice(at < 0 ? books.length : at, 0, m);
        candUsed.add(cands.indexOf(c));
      }
    }
    // weld fix-up: a token "XY" paired to the book whose own name is Y really starts on the previous book's sticker
    if (section === 'fiction') for (let i = 1; i < books.length; i++) {
      const b = books[i], p = books[i - 1], w = b.tokW;
      if (!w || b.agree !== 'disagree' || !b.key.tok || b.key.tok.length < 3) continue;
      if (!(w.L.length > b.key.tok.length + 2 && w.L.endsWith(b.key.tok))) continue;
      if (p.tokW && p.agree !== 'disagree') continue;
      p.tokW = w; p.agree = 'disagree'; p.cx = w.x0 + (w.x1 - w.x0) * 0.25;
      b.tokW = null; b.agree = undefined; b.cx = w.x0 + (w.x1 - w.x0) * 0.75; b.weldRight = true;
    }
    for (const i in books) if (books[i].tokW) used.add(books[i].tokW);
    const rowPitch = pitch;
    // Reconcile each paired book's two reads.
    books.forEach((b, i) => {
      const w = b.tokW; if (!w) { if (b.weldRight) b.az = true; return; }
      if (section === 'nonfiction') {
        const num = numberAbove(words, w);
        if (num) { b.azNum = num.T.replace(/\.$/, ''); b.azNumConf = num.conf; }
      } else {
        const g = givenBelow(words, w); if (g) b.azGiv = g.text;
      }
      if (section === 'fiction' && GIVEN_STOP.has(letters(b.key.giv))) {
        b.label = `${b.key.sur}${b.azGiv ? ', ' + b.azGiv : ''}`; b.key = keyOf(b.label, section);
      }
      // Same surname, different given name: the model paired this sticker with a
      // neighbour's given name (it read "WILSON, Hattie" off Lauren Wilson's sticker).
      // Azure's given-name line is a literal read of THIS sticker; take it when sure.
      if (section === 'fiction' && (b.agree === 'exact' || b.agree === 'fuzzy') && w.giv && (w.givConf || 0) >= 0.85 &&
          letters(w.giv).length >= 3 && b.key.giv && givenAgree(b.key.giv, w.giv) < 0 && /[a-z]/.test(w.giv)) {
        b.gemLabel = b.label; b.givFixed = true;
        b.label = `${b.key.sur}, ${w.giv}`; b.key = keyOf(b.label, section);
      }
      if (b.agree === 'fuzzy' && section === 'fiction' && w.conf >= 0.85 && w.complete &&
          !(w.x0 < W * 0.01 || w.x1 > W * 0.99) && w.L.length >= 4 && w.L.slice(0, 2) === (b.key.tok || '').slice(0, 2) &&
          !spineAt(b.cx, w.cy, rowPitch).some(v => nameMatch(v, b.key.tok))) {
        const azText0 = w.text.replace(/[^A-Za-z' -]+$/, '');
        b.override = true; b.gemLabel = b.label;
        b.label = `${azText0}, ${b.key.giv || b.azGiv || ''}`.replace(/,\s*$/, ''); b.key = keyOf(b.label, section);
        return;
      }
      if (b.agree !== 'disagree') return;
      if (w.x0 < W * 0.01 || w.x1 > W * 0.99) { b.agree = 'edge'; return; }   // a sticker cut by the frame
      if (/(\w{4,}).*\1/.test(w.L) || /([A-Za-z]{4,})\1/.test(b.azGiv || '') || /^(\w{3,})\1/.test(w.L)) {
        // Two identical stickers welded into one word ("WAGAWAGAMES", "RichardRichard"):
        // the longer half is the name both books carry. Fix this book and an unanchored
        // neighbour when the model misread them as something close ("WAGAN").
        b.agree = 'weld';
        if (section === 'fiction') {
          let Q = null;
          for (let k = 3; k <= w.L.length / 2; k++) if (w.L.slice(k).startsWith(w.L.slice(0, k))) { Q = w.L.slice(k); break; }
          if (Q && Q.length >= 4) for (const o of [b, books[i - 1], books[i + 1]]) {
            if (!o || (o !== b && o.tokW) || !o.key.tok || o.key.tok === Q || o.key.tok.slice(0, 3) !== Q.slice(0, 3)) continue;
            o.gemLabel = o.label; o.override = true;
            o.label = `${Q}${o.key.giv ? ', ' + o.key.giv : ''}`; o.key = keyOf(o.label, section);
          }
        }
        return;
      }
      let A = w.L, azText = w.text.replace(/[^A-Za-z' -]+$/, '');
      const G = b.key.tok || '';
      if (section === 'fiction') {
        // a weld of two stickers ("WEISBERGERWEST"): keep the part that is not the next book's name
        const nxt = books[i + 1]?.key.tok;
        if (nxt && nxt.length >= 3 && A.length > nxt.length + 2 && A.endsWith(nxt)) {
          const xCut = w.x0 + (w.x1 - w.x0) * (A.length - nxt.length) / A.length;
          A = A.slice(0, A.length - nxt.length); azText = A; b.cx = (w.x0 + xCut) / 2;
          const gw = words.filter(q => q.cy > w.cy + w.h * 0.3 && q.cy < w.cy + w.h * 2.4 && q.cx > w.x0 - w.h && q.cx < xCut && /[a-z]/.test(q.text));
          b.azGiv = gw[0]?.text; b.weld = w.conf >= 0.8;
        }
      }
      // Azure read only the tail of a name a neighbour carries in full ("GARBER" beside
      // "WEISGARBER"): it is that name, partly read -- complete it before judging.
      if (section === 'fiction' && A.length >= 4) {
        const full = [books[i - 2], books[i - 1], books[i + 1], books[i + 2]].map(o => o?.key.tok)
          .find(t => t && t.length > A.length + 1 && t.endsWith(A));
        if (full) { A = full; azText = full; }
      }
      const adopt = (giv) => {
        b.override = true; b.gemLabel = b.label;
        if (section === 'fiction') b.label = `${azText}, ${giv ?? (b.key.giv || b.azGiv || '')}`.replace(/,\s*$/, '');
        else b.label = `${b.key.num || b.azNum || ''} ${A}`.trim();
        b.key = keyOf(b.label, section);
      };
      if (!G) { if (A.length >= 3 && w.conf >= 0.6) adopt(); return; }
      const surWords = U(b.key.sur || '').split(/[\s-]+/).map(letters);
      // (a) Azure saw part of the same word (a dropped leading/trailing letter, or one word of "VON ARNIM")
      // (also: A is one word of a multi-word surname, possibly with a neighbour's letters welded on --
      //  "REINHOLDEN" for VON REINHOLD)
      const wordHit = surWords.some(sw => sw.length >= 4 && (A.startsWith(sw) || sw.startsWith(A) || ed(A, sw) <= 1));
      if ((A.length >= 3 && (G.includes(A) || A.startsWith(G) || surWords.includes(A) || (surWords.length > 1 && wordHit))) ||
          (A.length >= 2 && (G.startsWith(A) || G.endsWith(A)))) { b.agree = 'partial'; return; }
      const full = ed(A, G), pre = ed(A, G.slice(0, A.length));
      // (b) one-letter slip: the literal engine wins when it is sure (the model inserts letters too: "THEVALL")
      //     Fiction only: on a 3-letter cutter one letter is a third of the word and OCR
      //     confuses exactly those glyphs (GRO read "GRC"), so the model keeps its cutter.
      if (full <= 1) { if (w.conf >= 0.85 && section === 'fiction' && A.slice(0, 2) === G.slice(0, 2)) adopt(); else b.agree = 'partial'; return; }
      // (c) Azure's word is a slipped truncation of the model's
      if (A.length < G.length && pre <= 1) { b.agree = 'partial'; return; }
      // The spine usually prints the author's name: an independent read that settles
      // which engine is right (Azure's "TAVI" under a spine reading "BRAD TAYLOR").
      if (section === 'fiction') {
        const sp = spineAt(b.cx, w.cy, rowPitch);
        const forG = sp.some(v => nameMatch(v, G)), forA = sp.some(v => nameMatch(v, A));
        if (forG && !forA) { b.agree = 'partial'; b.spine = 'model'; return; }
        if (forA && !forG && A.length >= 3) { b.spine = 'azure'; adopt(); return; }
      }
      // (d) genuinely different text. The model conforms outliers to their neighbourhood, so Azure's
      //     literal read wins when the SAME sticker is confirmed by its other half, or when the model's
      //     label is an exact copy of a neighbour's.
      let confirmed = false;
      if (section === 'fiction') {
        const azG = letters(b.azGiv || ''), gG = letters(b.key.giv || '');
        confirmed = !!b.weld || (azG.length >= 3 && gG.length >= 3 && azG.slice(0, 3) === gG.slice(0, 3)) || (azG.length >= 2 && azG === gG);
      } else {
        confirmed = !!b.azNum && !!b.key.num && w.conf >= 0.9 &&
          (b.azNum.startsWith(b.key.num.replace(/\.$/, '')) || b.key.num.startsWith(b.azNum));
      }
      const copiesLabel = [books[i - 1], books[i + 1]].filter(Boolean).some(o => U(o.label) === U(b.label));
      // a sticker confirmed by its other half needs less from the word itself (WEISBERGER read
      // at 0.58, "Lauren" under it at 0.99); a bare neighbour-copy needs a confident word
      if (A.length >= 3 && ((confirmed && w.conf >= 0.5) || (copiesLabel && w.conf >= 0.75))) {
        // a copied label's given name is the neighbour's, not this book's
        adopt(confirmed ? undefined : (section === 'fiction' ? (b.azGiv || '') : undefined));
      }
    });
    // literal digits: the model flattens trailing decimals; take Azure's number when it is a confident
    // extension or a genuinely different number where the model copied a neighbour's
    if (section === 'nonfiction') books.forEach((b, i) => {
      if (!b.azNum || !b.key.num || b.override) return;
      const g = b.key.num.replace(/\.$/, ''), a = b.azNum;
      if (g === a || b.azNumConf < (opts.numConf ?? 0.9)) return;
      const nb = [books[i - 1], books[i + 1]].filter(Boolean);
      const copies = nb.some(o => (o.key.num || '').replace(/\.$/, '') === g);
      const ext = a.startsWith(g) || g.startsWith(a);
      if (a.split('.')[0] !== g.split('.')[0]) return;   // a different class is a fragment, not a flattened decimal
      if ((ext && a.length > g.length && opts.extend) || (!ext && copies && opts.numFix)) {
        b.numFixed = g; b.label = `${a} ${b.key.rest}`.trim(); b.key = keyOf(b.label, section);
      }
    });
    // inserted books
    for (const ins of inserts.sort((a, b) => b.afterIdx - a.afterIdx)) {
      const w = ins.w; let label;
      if (section === 'fiction') label = `${w.text.replace(/[^A-Za-z' -]+$/, '')}${w.giv ? ', ' + w.giv : ''}`;
      else label = `${numberAbove(words, w).T} ${w.L}`;
      const nb = { label, key: keyOf(label, section), tokW: w, cx: w.cx, agree: 'azonly', inserted: true };
      let at = books.findIndex(b => b.cx != null && b.cx > w.cx); if (at < 0) at = books.length;
      books.splice(at, 0, nb);
    }
  }
  // A model row that Azure's sticker words barely support is not a row of visible
  // stickers: on a shelf cut off by the frame the model reads the author names off
  // the spines instead ("NICO WALKER") and presents them as stickers.
  if (words.length >= 20 && !opts.keepRows) {
    for (let r = rows.length - 1; r >= 0; r--) {
      if (bands.length && rows[r].band == null) { rows.splice(r, 1); continue; }
      const bs = rows[r].books;
      const good = bs.filter(b => b.tokW && ['exact', 'fuzzy', 'partial'].includes(b.agree) || b.override).length;
      if (good < Math.max(3, bs.length * 0.3)) rows.splice(r, 1);
    }
  }
  // positions for books without a token; boxes
  for (const row of rows) {
    const bs = row.books;
    const idx = bs.map((b, i) => b.cx != null ? i : -1).filter(i => i >= 0);
    const g = []; for (let k = 1; k < idx.length; k++) g.push((bs[idx[k]].cx - bs[idx[k - 1]].cx) / (idx[k] - idx[k - 1]));
    const pitch = median(g.filter(v => v > 0)) || W / Math.max(10, bs.length);
    for (let i = 0; i < bs.length; i++) {
      if (bs[i].cx != null) continue;
      const L = idx.filter(m => m < i).pop(), R = idx.find(m => m > i);
      bs[i].cx = L != null && R != null ? bs[L].cx + (bs[R].cx - bs[L].cx) * (i - L) / (R - L)
        : L != null ? bs[L].cx + pitch * (i - L) : R != null ? bs[R].cx - pitch * (R - i) : (i + 0.5) * W / bs.length;
      bs[i].interp = true;
      bs[i].extrap = !(L != null && R != null);   // beyond the last sticker Azure placed: position is a guess
    }
    const hh = median(bs.filter(b => b.tokW).map(b => b.tokW.h)) || hMed;
    bs.forEach((b, i) => {
      const y = b.tokW ? b.tokW.cy : row.line ? row.line.yAt(b.cx) : H * (rows.indexOf(row) + 0.5) / rows.length;
      const left = i ? (bs[i - 1].cx + b.cx) / 2 : b.cx - pitch / 2, right = i < bs.length - 1 ? (b.cx + bs[i + 1].cx) / 2 : b.cx + pitch / 2;
      b.x0 = Math.max(0, left); b.x1 = Math.min(W, right); b.y0 = Math.max(0, y - hh * 11); b.y1 = Math.min(H, y + hh * 3);
      b.az = !!b.tokW && b.agree !== 'disagree';
    });
  }
  return { rows: rows.map(r => ({ books: r.books, line: r.line })), W, H, words };
}

export function cleanForOrder(label, section) {
  let s = String(label || '').trim(), trunc = false, unreadable = false;
  if (section === 'fiction') {
    s = s.replace(/,\s*\?+$/, '').replace(/\s*\?+$/, '');           // "WALKER, ?" -> surname only
    s = s.replace(/,\s*(the|a|an|of|and)\s*$/i, '');                 // "WALKER, The" -> surname only
    if (/\?/.test(s.split(',')[0]) || !/[A-Za-z]{2}/.test(s)) unreadable = true;
    s = s.replace(/\?/g, '');
  } else {
    const m = /^(\d{3}(?:\.[\d?]*)?)\s*(.*)$/.exec(s);
    if (!m) unreadable = true;
    else {
      let num = m[1], rest = m[2];
      if (num.includes('?')) { num = num.slice(0, num.indexOf('?')).replace(/\.$/, ''); trunc = true; }
      if (num.endsWith('.')) { num = num.slice(0, -1); trunc = true; }   // "796. CHE": decimals wrapped out of view
      if (/\?/.test(rest.split(/\s+/)[0] || '?')) unreadable = true;
      s = `${num} ${rest.replace(/\?/g, '')}`.trim();
    }
  }
  return { s, trunc, unreadable };
}

// Book objects for ordering.js. _score is not a model confidence (the model's
// own number was measured useless: 97% of reads >= 0.9, 74% right); it encodes
// what the two engines established about the read:
//   0.97 both engines agree, or Azure's literal text replaced a model copy
//   0.90 paired with an Azure word that neither confirms nor contradicts it
//   0.55 model only (Azure saw no sticker there) -- sortable, never red: a
//        book only the model reports may be one it imagined (measured: a
//        third WELSH invented at a row end, accused)
//   0.85 the engines disagree and the model's text was kept
//   0.90 a sticker only Azure saw (the model skipped the book); inserted only
//        on a strong signature (a Dewey number over a cutter, or a complete
//        surname + given name inside a run the model flattened)
//   0.30 unreadable -> never sorted, shown as "couldn't read"
export function toBooks(fused, section) {
  const books = [];
  // Azure unavailable (outage, monthly budget spent): there is no second engine to
  // require, so the model's read may accuse on its own, as the previous app did.
  const noAzure = !fused.words || fused.words.length < 5;
  fused.rows.forEach((row, ri) => row.books.forEach(b => {
    const c = cleanForOrder(b.label, section);
    const score = c.unreadable ? 0.3
      : section === 'nonfiction' && b.agree === 'fuzzy' && !b.override ? 0.5   // engines differ by a letter (HAG / HAR)
      : b.agree === 'exact' || b.agree === 'fuzzy' || b.override ? 0.97
      : b.agree === 'disagree' ? 0.85 : b.inserted ? 0.9 : (b.tokW || noAzure ? 0.9 : 0.55);   // 0.55: below RED_MIN_SCORE -- a sticker Azure never saw is not accused
    books.push({
      spine_label: c.s || b.label, shelfRow: ri, _src: 'gemini', _score: score,
      ...(c.unreadable ? { confidence: 'low', _unreadable: true } : {}),
      // "couldn't read" is a claim about THIS sticker: the model said so ("?" / a clipped
      // number), or Azure saw a sticker there that did not parse. A model-only label
      // that does not parse is more likely spine text than a sticker, and gets no box.
      _readTrunc: c.trunc, _saidUnreadable: /\?/.test(b.label) || !!b.tokW || c.trunc,
      ...(c.trunc ? { _truncated: true } : {}),
      // a surname with no readable given name files under its author but not within it
      ...(section === 'fiction' && !c.unreadable && !/,\s*\S/.test(c.s) ? { _noGiven: true } : {}),
      _bbox: [b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0],
      _box01: [b.x0 / fused.W, b.y0 / fused.H, (b.x1 - b.x0) / fused.W, (b.y1 - b.y0) / fused.H],
      _raw: b.label, _agree: b.agree || null, _override: !!b.override, _inserted: !!b.inserted,
      // where the box can honestly be drawn: not extrapolated past the placed stickers,
      // and not a spine cut by the photo's edge (it belongs to the next photo)
      _extrap: !!b.extrap, _edge: b.x0 <= fused.W * 0.015 || b.x1 >= fused.W * 0.985,
      // both engines read this sticker's name/cutter the same way (or Azure's literal
      // read replaced the model's on confirmed evidence): not a garbled read
      ...(!c.unreadable && (b.agree === 'exact' || b.agree === 'partial' || b.override) ? { _azConfirmed: true } : {}),
      // nonfiction: Azure read this exact call number, digit for digit, with confidence
      ...(section === 'nonfiction' && b.azNum && b.azNumConf >= 0.9 && b.key.num &&
          b.azNum === String(b.key.num).replace(/\.$/, '') && b.azNum.includes('.') ? { _numConfirmed: true } : {}),
    });
  }));
  return books;
}
