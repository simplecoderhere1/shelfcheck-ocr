/**
 * Is the "number cut off" flag TRUE?
 *
 * The other suites measure whether a label was read. This one measures whether
 * the app was right to decline to judge it — the flag that turns a book yellow
 * and sends a volunteer to check it by hand. A flag on a number that was in
 * fact read in full is pure noise, and noise is what makes a volunteer stop
 * trusting the app.
 *
 * Section A additionally classifies every recall miss whose number AND author
 * tag are both sitting in the Azure word list, i.e. the ones the assembler lost
 * rather than the reader. It prints the pairing geometry for each so the cause
 * is visible instead of guessed at.
 *
 *   node trunc_accuracy.mjs                    # both sections
 *   node trunc_accuracy.mjs --ink=0.6 --run=4  # sweep inkRightOfNumber
 *
 * Like the other harnesses the app code is EXTRACTED from index.html, never
 * copied, and real luminance is served from _lum/*.pgm — a dimensions-only stub
 * ctx would silently disable the geometry test and flatter the result.
 */
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { loadGT } from './gt_load.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(HERE, '..', '..', 'shelfcheck-ocr', 'index.html'), 'utf-8');
const start = html.indexOf('// Fraction of pixels in a region');
const end   = html.indexOf('// ── Gemini fusion (pure;');
let _s = html.slice(start, end);
const _a = k => (process.argv.find(x => x.startsWith(`--${k}=`)) || '').split('=')[1];
if (_a('ink'))  _s = _s.replace('AZ_INK_MIN = 0.70', `AZ_INK_MIN = ${_a('ink')}`);
if (_a('wall')) _s = _s.replace('AZ_INK_WALL = 0.85', `AZ_INK_WALL = ${_a('wall')}`);
if (_a('run'))  _s = _s.replace('AZ_INK_WALL_RUN = 3', `AZ_INK_WALL_RUN = ${_a('run')}`);
const fns = new Function(_s + `
  return { parseAzureWords, reconstructNonfictionAzure, mergeAzureWordSets, inkRightOfNumber };
`)();

function pgm(path) {
  const b = readFileSync(path);
  let i = 0; const tok = [];
  while (tok.length < 4) {
    while (b[i] === 32 || b[i] === 10 || b[i] === 13 || b[i] === 9) i++;
    if (b[i] === 35) { while (b[i] !== 10) i++; continue; }
    const s = i; while (b[i] > 32) i++; tok.push(b.toString('ascii', s, i));
  }
  return { w: +tok[1], h: +tok[2], d: b.subarray(i + 1) };
}
function lumCtx(L) {
  return {
    canvas: { width: L.w, height: L.h },
    getImageData(x, y, gw, gh) {
      const data = new Uint8ClampedArray(gw * gh * 4);
      for (let j = 0; j < gh; j++) for (let i2 = 0; i2 < gw; i2++) {
        const yy = y + j, xx = x + i2;
        const v = (yy >= 0 && yy < L.h && xx >= 0 && xx < L.w) ? L.d[yy * L.w + xx] : 0;
        const o = (j * gw + i2) * 4;
        data[o] = data[o + 1] = data[o + 2] = v; data[o + 3] = 255;
      }
      return { data };
    },
  };
}
const numStr = s => { const m = /^\s*(\d+(?:\.\d+)?)/.exec(String(s)); return m ? m[1] : null; };
function parseDeweyStr(s) {
  const m = /^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]{2,})?/.exec(String(s));
  return m ? { num: parseFloat(m[1]), cut: (m[2] || '').toUpperCase() } : null;
}
function labelMatch(d, g) {
  const dn = numStr(d), gn = numStr(g), dm = parseDeweyStr(d), gm = parseDeweyStr(g);
  if (!(dn && gn && dm && gm)) return false;
  const numOk = dn.startsWith(gn) || gn.startsWith(dn);
  const dc = (dm.cut || '').slice(0, 3), gc = (gm.cut || '').slice(0, 3);
  return numOk && !!dc && !!gc && dc === gc;
}
function wordsFor(rec) {
  const full = fns.parseAzureWords(rec.full);
  if (!rec.tiles?.length) return full;
  const tw = [];
  for (const t of rec.tiles) for (const w of fns.parseAzureWords(t.raw)) { w.x += t.xoff; tw.push(w); }
  return fns.mergeAzureWordSets(full, tw, rec.w / rec.natW);
}

const GT = loadGT();
console.log('=== A. assembly misses ===');
let badTrunc = 0, goodTrunc = 0, totalMatched = 0;
const truncRows = [];
for (const [img, g] of Object.entries(GT)) {
  if (img.startsWith('_') || g.section !== 'nonfiction') continue;
  const p = join(HERE, 'azure_tiled', img + '.json');
  if (!existsSync(p)) continue;
  const rec = JSON.parse(readFileSync(p, 'utf-8'));
  const words = wordsFor(rec);
  const ctx = lumCtx(pgm(join(HERE, '_lum', img + '.pgm')));
  const labels = fns.reconstructNonfictionAzure(words, ctx);
  const texts = labels.map(l => l.text);

  const unc = new Set(); const all = [];
  for (const r of g.rows || []) { all.push(...r.labels); (r.uncertain || []).forEach(u => unc.add(u)); }
  const hc = all.filter(l => !unc.has(l));

  const used = new Set(); const missed = [];
  for (const gl of hc) {
    let ok = false;
    for (let i = 0; i < texts.length; i++) {
      if (used.has(i)) continue;
      if (labelMatch(texts[i], gl)) {
        used.add(i); ok = true;
        // B. truncation correctness: our number vs GT number
        const l = labels[i];
        const dn = numStr(l.text), gn = numStr(gl);
        const complete = dn === gn;
        totalMatched++;
        if (l.truncated) {
          if (complete) { badTrunc++; truncRows.push([img, gl, l.text, 'FALSE-CUTOFF']); }
          else goodTrunc++;
        } else if (!complete) {
          truncRows.push([img, gl, l.text, 'MISSED-CUTOFF']);
        }
        break;
      }
    }
    if (!ok) missed.push(gl);
  }

  // diagnose each miss
  const short = img.replace('PXL_2026060', '').replace('.MP', '').replace('.jpg', '');
  const maxH = Math.max(ctx.canvas.width, ctx.canvas.height) * 0.022;
  for (const gl of missed) {
    const m = /^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]{2,})/.exec(gl);
    if (!m) continue;
    const c3 = m[2].toUpperCase().slice(0, 3);
    const nums = words.filter(w => /^\d/.test(w.t.replace(',', '.')) &&
      (w.t.replace(',', '.').startsWith(m[1]) || m[1].startsWith(w.t.replace(',', '.'))));
    const cuts = words.filter(w => /^[A-Za-z]{2,4}$/.test(w.t) && w.t.toUpperCase().slice(0, 3) === c3);
    if (!nums.length || !cuts.length) continue;  // Azure gap, not assembly
    // Only numbers plausibly on the SAME sticker as one of the candidate
    // cutters — otherwise every 641.568 on the shelf prints.
    const near = nums.filter(n => cuts.some(c =>
      Math.abs((c.x + c.w/2) - (n.x + n.w/2)) < n.h * 8 && c.y - n.y > -n.h && c.y - n.y < n.h * 6));
    if (!near.length) continue;
    console.log(`\n${short}  MISS "${gl}"`);
    for (const n of near) console.log(`   num  "${n.t}" x=${n.x|0} y=${n.y|0} w=${n.w|0} h=${n.h|0} c=${(n.c??.9).toFixed(2)}${n.h>maxH?' [TOO TALL]':''}${n.w < n.h*0.6?' [TOO NARROW]':''}`);
    for (const c of cuts) console.log(`   cut  "${c.t}" x=${c.x|0} y=${c.y|0} w=${c.w|0} h=${c.h|0} c=${(c.c??.9).toFixed(2)}${c.h>maxH?' [TOO TALL]':''}${c.w < c.h*0.6?' [TOO NARROW]':''}`);
    for (const n of near) for (const c of cuts) {
      const dy = c.y - (n.y + n.h);
      const dx = Math.abs((c.x + c.w/2) - (n.x + n.w/2));
      const xw = Math.max(n.w, c.w) * 1.2 + n.h * 3.0;
      console.log(`   pair ${n.t}/${c.t}: dy=${dy.toFixed(0)} (${(dy/n.h).toFixed(2)}h, need -0.8..3.0) dx=${dx.toFixed(0)} (win ${xw.toFixed(0)}) ${(dy>=-n.h*0.8&&dy<=n.h*3&&dx<=xw)?'OK->stolen/other':'REJECTED'}`);
    }
    // who took the cutter?
    for (const c of cuts) {
      const owner = labels.find(l => l.cut === c.t.toUpperCase() &&
        l.cutBox && Math.abs(l.cutBox[0]-c.x)<2 && Math.abs(l.cutBox[1]-c.y)<2);
      if (owner) console.log(`   -> cutter "${c.t}" was taken by "${owner.text}"`);
    }
  }
}
console.log(`\n=== B. truncation flags on matched labels (${totalMatched} matched) ===`);
for (const r of truncRows) console.log(`  ${r[3].padEnd(14)} ${r[0].replace('PXL_2026060','').replace('.jpg','')}  GT="${r[1]}" got="${r[2]}"`);
console.log(`\n  correct cut-off flags : ${goodTrunc}`);
console.log(`  FALSE cut-off flags   : ${badTrunc}`);
