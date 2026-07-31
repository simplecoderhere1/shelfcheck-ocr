/**
 * Emit the transcription plan for rebuilding ground truth.
 *
 * ground_truth_v2.json turned out to list roughly half the books physically on
 * these shelves (see audit_extras.mjs — 17 of 17 "unaccounted" labels on
 * 195911082 were real stickers the key never recorded). Any recall or precision
 * figure measured against it is measuring the key, not the app.
 *
 * Rebuilding it from scratch by eye is ~400 labels. Rebuilding it by VERIFYING
 * the app's own reads is far cheaper and, per that same audit, starts from
 * mostly-correct text. The risk of bootstrapping is inheriting the app's blind
 * spots, so the plan carries two artifacts per image and the second exists
 * purely to defeat that:
 *
 *   <img>_crops.jpg    every emitted label, numbered, cropped from the native
 *                      photo — "is this what the sticker says?"
 *   <img>_cover_N.jpg  the whole shelf in strips with every emitted label boxed
 *                      — "which stickers have NO box on them?"
 *
 *   FIXTURE=1 node site_suite.mjs --json=live_step1.json
 *   node gt_plan.mjs --fixture=live_step1.json
 *   python gt_sheets.py
 */
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
const arg = (k, d = '') =>
  (process.argv.find(a => a.startsWith(`--${k}=`)) || '').split('=').slice(1).join('=') || d;

const live = JSON.parse(readFileSync(join(HERE, arg('fixture', 'live_step1.json')), 'utf-8'));
const OUT = join(HERE, '_gt');
mkdirSync(OUT, { recursive: true });

const plan = {};
let total = 0;
for (const r of (live.results || [])) {
  if (!r.fixture || !r.fixture.length) {
    console.log(`${r.imgName}: NO FIXTURE — re-run site_suite with FIXTURE=1`);
    continue;
  }
  // Physical reading order: row first, then left to right. The rebuilt key has
  // to preserve this — run_ordering_test replays the row as a sequence, so a
  // key sorted alphabetically would silently erase every real misshelving.
  const labels = r.fixture
    .filter(b => b.spine_label && b._bbox)
    .map(b => ({
      text: b.spine_label,
      row: b.shelfRow ?? 0,
      score: b._score ?? null,
      title: b.title ?? null,
      bbox: b._bbox,
    }))
    .sort((a, b) => a.row - b.row || a.bbox[0] - b.bbox[0]);

  labels.forEach((l, i) => { l.n = i + 1; });
  total += labels.length;
  plan[r.imgName] = { section: r.section, labels };
  const rows = [...new Set(labels.map(l => l.row))].length;
  console.log(`${r.imgName.replace('PXL_2026060', '').padEnd(20)} ${r.section.padEnd(11)} ` +
    `${String(labels.length).padStart(3)} label(s) across ${rows} row(s)`);
}

writeFileSync(join(OUT, '_plan.json'), JSON.stringify(plan, null, 1));
console.log(`\n${total} label(s) to verify. Wrote ${join(OUT, '_plan.json')}`);
console.log('Next:  python gt_sheets.py');
