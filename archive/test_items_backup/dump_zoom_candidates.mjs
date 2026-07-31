/**
 * Emit the stickers that need a closer look, in NATIVE photo coordinates.
 *
 * Step 1 of rebuilding the closer-look cache:
 *   node dump_zoom_candidates.mjs      # -> _cands.json
 *   python build_zoomcache.py          # -> azure_zoom/
 *
 * Only needed when the assembler changes enough that the cached crops no longer
 * line up — azure_recon_bench matches cached cells to current labels by sticker
 * text and skips any it cannot match, so a stale cache understates the pass
 * rather than corrupting it.
 *
 * The candidate rule and the crop box are the app's own (zoomCandidates, and
 * the padding in encodeZoomSheet); this file mirrors the box arithmetic because
 * encodeZoomSheet needs a canvas and cannot run under node.
 */
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { loadGT } from './gt_load.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(HERE, '..', '..', 'shelfcheck-ocr', 'index.html'), 'utf-8');
const src = html.slice(html.indexOf('// Fraction of pixels in a region'),
                       html.indexOf('// ── Gemini fusion (pure;'));
const fns = new Function(src + `
  return { parseAzureWords, reconstructNonfictionAzure, mergeAzureWordSets, zoomCandidates };`)();

function pgm(p) {
  const b = readFileSync(p); let i = 0; const t = [];
  while (t.length < 4) {
    while (b[i] === 32 || b[i] === 10 || b[i] === 13 || b[i] === 9) i++;
    if (b[i] === 35) { while (b[i] !== 10) i++; continue; }
    const s = i; while (b[i] > 32) i++; t.push(b.toString('ascii', s, i));
  }
  return { w: +t[1], h: +t[2], d: b.subarray(i + 1) };
}
function lumCtx(L) {
  return { canvas: { width: L.w, height: L.h },
    getImageData(x, y, gw, gh) {
      const d = new Uint8ClampedArray(gw * gh * 4);
      for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
        const yy = y + j, xx = x + i;
        const v = (yy >= 0 && yy < L.h && xx >= 0 && xx < L.w) ? L.d[yy * L.w + xx] : 0;
        const o = (j * gw + i) * 4; d[o] = d[o + 1] = d[o + 2] = v; d[o + 3] = 255;
      }
      return { data: d };
    } };
}
function wordsFor(rec) {
  const full = fns.parseAzureWords(rec.full);
  if (!rec.tiles?.length) return full;
  const tw = [];
  for (const t of rec.tiles) for (const w of fns.parseAzureWords(t.raw)) { w.x += t.xoff; tw.push(w); }
  return fns.mergeAzureWordSets(full, tw, rec.w / rec.natW);
}

const GT = loadGT();
const out = {};
for (const [img, g] of Object.entries(GT)) {
  if (img.startsWith('_') || g.section !== 'nonfiction') continue;
  const p = join(HERE, 'azure_tiled', img + '.json');
  if (!existsSync(p)) continue;
  const rec = JSON.parse(readFileSync(p, 'utf-8'));
  const ctx = lumCtx(pgm(join(HERE, '_lum', img + '.pgm')));
  // keepUnpaired, exactly as runOcr does — anchors with no author tag are the
  // ones the closer look most needs to see.
  const labels = fns.reconstructNonfictionAzure(wordsFor(rec), ctx, true);
  const s = rec.natW / rec.w;                       // 2400-space -> native
  const cands = fns.zoomCandidates(labels).map(i => {
    const [x, y, w, h] = labels[i].numBox;
    return { i, text: labels[i].text,
             box: [Math.round((x - h * 2.2) * s), Math.round((y - h * 2.2) * s),
                   Math.round((x + w + h * 2.2) * s), Math.round((y + h * 4.0) * s)] };
  });
  out[img] = { natW: rec.natW, natH: rec.natH, cands };
  console.log(img.replace('PXL_2026060', ''), 'labels', labels.length, 'need a closer look', cands.length);
}
writeFileSync(join(HERE, '_cands.json'), JSON.stringify(out, null, 1));
