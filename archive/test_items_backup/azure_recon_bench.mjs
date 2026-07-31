/**
 * Offline bench for the AZURE nonfiction assembly step.
 *
 * Runs the app's REAL code (extracted from index.html, never copied) over the
 * cached Azure responses and scores label recall/precision against
 * ground_truth_v2 with the same strict matcher site_suite uses. No API calls,
 * so an assembly change can be measured in ~1s instead of a ~6-minute live run.
 *
 * The cache in azure_tiled/ mirrors what the SHIPPING app sends: a full frame
 * encoded to <=2400px plus two native-resolution vertical strips with 5%
 * overlap. mergeAzureWordSets folds them together exactly as runOcr does, so
 * the number here tracks the live suite instead of drifting from it.
 * Rebuild the cache with _build_tilecache.py if the tiling scheme changes.
 *
 *   node azure_recon_bench.mjs              # score every image
 *   node azure_recon_bench.mjs --dump=195911082
 *   node azure_recon_bench.mjs --why        # classify every miss by cause
 */
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { loadGT } from './gt_load.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const APP  = join(HERE, '..', '..', 'shelfcheck-ocr', 'index.html');
const arg = (k, d = '') =>
  (process.argv.find(a => a.startsWith(`--${k}=`)) || '').split('=').slice(1).join('=') || d;
const DUMP = arg('dump');
const WHY  = process.argv.includes('--why');
const NOZOOM = process.argv.includes('--nozoom');
let zoomFixed = 0, zoomTried = 0;

const html = readFileSync(APP, 'utf-8');
const start = html.indexOf('// Fraction of pixels in a region');
const end   = html.indexOf('// ── Gemini fusion (pure;');
if (start < 0 || end < 0) { console.error('extraction markers not found'); process.exit(1); }

let src = html.slice(start, end);
// Test-only constant rewrites for sweeps; they patch the extracted copy, never
// the file.
const DXH = arg('dxh'), DYH = arg('dy');
if (DXH) src = src.replace(/AZ_CUT_DX_H = [\d.]+/, `AZ_CUT_DX_H = ${DXH}`);
if (DYH) src = src.replace(/dy > a\.h \* 3\.0/, `dy > a.h * ${DYH}`);
const TOL = arg('tol');
const DED = arg('ded');
if (DED) src = src.replace('w * h) >= 0.4;', `w * h) >= ${DED};`);
if (TOL) src = src.replace('iou(o, t) >= 0.5', `iou(o, t) >= ${TOL}`);
const ZM = arg('zm');
if (ZM) src = src.replace('AZ_ZOOM_MIN_CONF = 0.80', `AZ_ZOOM_MIN_CONF = ${ZM}`);
const ZCT = arg('zct'), ZDX = arg('zdx');
if (ZCT) src = src.replace('AZ_ZOOM_CENTRE = 0.22', `AZ_ZOOM_CENTRE = ${ZCT}`);
if (ZDX) src = src.replace('AZ_ZOOM_CUT_DX = 1.5', `AZ_ZOOM_CUT_DX = ${ZDX}`);
const WD = arg('wd');
if (WD) src = src.replace('overlapMin(o, w) >= 0.55', `overlapMin(o, w) >= ${WD}`);
// --wdtie=conf ranks repeat reads of one word by Azure confidence first instead
// of by how many characters came back.
if (arg('wdtie') === 'conf') src = src.replace(
  `const better = w.t.length !== dup.t.length
        ? w.t.length > dup.t.length
        : (w.c ?? 0.9) > (dup.c ?? 0.9);`,
  `const better = (w.c ?? 0.9) !== (dup.c ?? 0.9)
        ? (w.c ?? 0.9) > (dup.c ?? 0.9)
        : w.t.length > dup.t.length;`);
if (process.argv.includes('--nowd')) src = src.replace('dedupeWordReads(words.filter', '((x)=>x)(words.filter');

const fns = new Function(src + `
  return { parseAzureWords, reconstructNonfictionAzure, mergeAzureWordSets,
           zoomCandidates, readZoomCells, mergeZoomReads };
`)();

// The Azure reconstructor samples pixels for the truncation-geometry test
// (inkRightOfNumber), so the bench serves real luminance from _lum/*.pgm â€”
// rasters at the shipping full-frame scale, built by build_lum.py. A
// dimensions-only stub would silently disable that test and flatter the score.
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

// â”€â”€ Strict matcher, identical to site_suite.mjs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function parseDeweyStr(s) {
  const m = /^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]{2,})?/.exec(String(s));
  return m ? { num: parseFloat(m[1]), cut: (m[2] || '').toUpperCase() } : null;
}
const numStr = s => { const m = /^\s*(\d+(?:\.\d+)?)/.exec(String(s)); return m ? m[1] : null; };
function labelMatch(detected, gt) {
  const dn = numStr(detected), gn = numStr(gt);
  const dm = parseDeweyStr(detected), gm = parseDeweyStr(gt);
  if (!(dn && gn && dm && gm)) return false;
  const numOk = dn.startsWith(gn) || gn.startsWith(dn);
  const dc = (dm.cut || '').slice(0, 3), gc = (gm.cut || '').slice(0, 3);
  return numOk && !!dc && !!gc && dc === gc;
}

function wordsFor(rec) {
  const full = fns.parseAzureWords(rec.full);
  if (!rec.tiles?.length) return full;
  const tw = [];
  for (const t of rec.tiles) {
    for (const w of fns.parseAzureWords(t.raw)) { w.x += t.xoff; tw.push(w); }
  }
  return fns.mergeAzureWordSets(full, tw, rec.w / rec.natW);
}

const GT = loadGT();
let hit = 0, den = 0, det = 0;
const rows = [];
const causes = { assembly: 0, noNum: 0, noCut: 0, neither: 0 };

for (const [img, g] of Object.entries(GT)) {
  if (img.startsWith('_') || g.section !== 'nonfiction') continue;
  const p = join(HERE, 'azure_tiled', img + '.json');
  if (!existsSync(p)) continue;

  const lp = join(HERE, '_lum', img + '.pgm');
  if (!existsSync(lp)) { console.error(`missing ${lp} ï¿½ run build_lum.py`); process.exit(1); }
  const rec = JSON.parse(readFileSync(p, 'utf-8'));
  const words = wordsFor(rec);
  // keepUnpaired + filter-after mirrors runOcr exactly: the closer look must
  // be able to see anchors that never found an author tag.
  const labels = fns.reconstructNonfictionAzure(words, lumCtx(pgm(lp)), !NOZOOM);
  // Closer-look pass, replayed from azure_zoom/ (built by build_zoomcache.py).
  // The candidate list is recomputed from the CURRENT labels and matched to the
  // cached cells by sticker text, so the cache stays valid as the assembler
  // changes; cells it cannot match are skipped rather than misapplied.
  const zp = join(HERE, process.env.ZDIR || 'azure_zoom', img + '.json');
  if (!NOZOOM && existsSync(zp)) {
    const z = JSON.parse(readFileSync(zp, 'utf-8'));
    if (z.raw) {
      const cells = fns.readZoomCells(fns.parseAzureWords(z.raw), z.rects);
      const idxs = [], reads = [];
      for (const li of fns.zoomCandidates(labels)) {
        const ci = z.cands.findIndex(c => c.text === labels[li].text);
        if (ci < 0 || !cells[ci]) continue;
        idxs.push(li); reads.push(cells[ci]);
      }
      zoomFixed += fns.mergeZoomReads(labels, idxs, reads);
      zoomTried += idxs.length;
    }
  }
  if (!NOZOOM) { const k = labels.filter(l => l.cut); labels.length = 0; labels.push(...k); }
  const texts = labels.map(l => l.text);

  const unc = new Set(); const all = [];
  for (const r of g.rows || []) { all.push(...r.labels); (r.uncertain || []).forEach(u => unc.add(u)); }
  const hc = all.filter(l => !unc.has(l));

  const used = new Set();
  let n = 0;
  const missed = [];
  for (const gl of hc) {
    let ok = false;
    for (let i = 0; i < texts.length; i++) {
      if (used.has(i)) continue;
      if (labelMatch(texts[i], gl)) { used.add(i); n++; ok = true; break; }
    }
    if (!ok) missed.push(gl);
  }
  hit += n; den += hc.length; det += texts.length;
  rows.push([img, hc.length, texts.length, n]);

  // Why did each miss happen? If BOTH parts are sitting in the word list as
  // exact tokens, the reader did its job and the assembler lost the label â€”
  // that is ours to fix. Otherwise Azure never returned the text.
  if (WHY) {
    const toks = new Set(words.map(w => w.t.replace(',', '.').toUpperCase()));
    for (const gl of missed) {
      const m = /^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]{2,})/.exec(gl);
      if (!m) { causes.neither++; continue; }
      // Match the SCORER's rules, not stricter ones: it accepts a number that
      // is a prefix of GT (or vice versa) and compares only the cutter's first
      // 3 chars. Counting exact tokens here blamed Azure for cutters it had
      // actually read well enough to score.
      const c3 = m[2].toUpperCase().slice(0, 3);
      const hasN = [...toks].some(t => /^\d/.test(t) && (t.startsWith(m[1]) || m[1].startsWith(t)));
      const hasC = [...toks].some(t => /^[A-Z]{2,4}$/.test(t) && t.slice(0, 3) === c3);
      if (hasN && hasC) causes.assembly++;
      else if (!hasN && hasC) causes.noNum++;
      else if (hasN && !hasC) causes.noCut++;
      else causes.neither++;
    }
  }

  if (DUMP && img.includes(DUMP)) {
    console.log(`--- ${img} ---`);
    console.log('detected:', texts.join(' | '));
    console.log('missed  :', missed.join(' | '));
  }
}

console.log(`${'image'.padEnd(34)}${'GT'.padStart(4)}${'found'.padStart(7)}${'matched'.padStart(9)}${'recall'.padStart(9)}`);
for (const [img, d, f, n] of rows) {
  const s = img.replace('PXL_2026060', '').replace('.MP', '').replace('.jpg', '');
  console.log(`${s.padEnd(34)}${String(d).padStart(4)}${String(f).padStart(7)}${String(n).padStart(9)}` +
              `${(100 * n / d).toFixed(1).padStart(8)}%`);
}
console.log('-'.repeat(63));
console.log(`recall    ${hit}/${den} = ${(100 * hit / den).toFixed(1)}%`);
console.log(`closer look ${zoomFixed} label(s) changed of ${zoomTried} re-read`);
console.log(`precision ${hit}/${det} = ${(100 * hit / det).toFixed(1)}%`);
if (WHY) {
  const tot = Object.values(causes).reduce((a, b) => a + b, 0);
  console.log(`\nwhy the ${tot} misses happened:`);
  console.log(`  assembly (both parts read, label lost) : ${causes.assembly}`);
  console.log(`  number never read                      : ${causes.noNum}`);
  console.log(`  cutter never read                      : ${causes.noCut}`);
  console.log(`  neither read                           : ${causes.neither}`);
}
