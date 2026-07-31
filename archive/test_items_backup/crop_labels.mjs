/**
 * Crop specific labels out of a photo, by the text the app read.
 *
 * The general audit (audit_extras) only shows labels ground truth does not
 * account for. When a specific label misbehaves — a false misshelving flag, say
 * — you need to see THAT sticker whether or not GT knows about it.
 *
 *   node crop_labels.mjs --fixture=live_step1.json --labels="635 ICON,635 FAS"
 *   node crop_labels.mjs --fixture=live_step1.json --flags     # every flagged book
 *
 * Writes _audit/_crops.json, then run  python audit_extras.py --plan=_crops.json
 */
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
const arg = (k, d = '') =>
  (process.argv.find(a => a.startsWith(`--${k}=`)) || '').split('=').slice(1).join('=') || d;

const live = JSON.parse(readFileSync(join(HERE, arg('fixture', 'live.json')), 'utf-8'));
const FLAGS = process.argv.includes('--flags');
const WANT = new Set(arg('labels').split(',').map(s => s.trim()).filter(Boolean));

const OUT = join(HERE, '_audit');
mkdirSync(OUT, { recursive: true });

const plan = {};
let n = 0;
for (const r of (live.results || [])) {
  if (!r.fixture) continue;
  // The suite records flags in `fp` (false positives) and flagEvidence.
  const flagTexts = new Set(FLAGS ? [...(r.fp || []), ...(r.flagEvidence || []).map(e => e.spine_label)] : []);
  const picks = r.fixture.filter(b =>
    b._bbox && b.spine_label && (WANT.has(b.spine_label) || flagTexts.has(b.spine_label)));
  if (!picks.length) continue;
  n += picks.length;
  plan[r.imgName] = {
    section: r.section, emitted: r.fixture.length, accountedFor: 0,
    extras: picks.map(b => ({
      text: b.spine_label, score: b._score ?? null, src: b._src ?? null,
      title: b.title ?? null, bbox: b._bbox,
    })),
  };
  console.log(`${r.imgName.replace('PXL_2026060', '')}  ${picks.map(p => `"${p.spine_label}"`).join(', ')}`);
}
writeFileSync(join(OUT, '_crops.json'), JSON.stringify(plan, null, 1));
console.log(`\n${n} crop(s) planned -> _audit/_crops.json`);
