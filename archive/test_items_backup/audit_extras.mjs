/**
 * Crop every label the app emitted that ground truth does NOT account for.
 *
 * Why this exists: the app emits ~388 labels across the 10 shelves and only
 * ~284 match an entry in ground_truth_v2.json. That gap is NOT 104 mistakes —
 * some of it is real books the answer key never recorded (635 GAR, 635 CAL,
 * 635 BAK and 635.0484 BEN were all found that way), and some is genuine junk
 * (spine titles and frame-edge fragments read as call numbers: "635 ICON",
 * "604 FOTO", "635 AK"). Precision cannot be quoted until the two are told
 * apart, and the only way to tell them apart is to look at the pixels.
 *
 * So: cut each unaccounted-for label out of the native photo, and lay them on
 * one numbered contact sheet per image. A human (or a vision model) reads the
 * sheet and says, for each crop, whether a real sticker is there.
 *
 *   node audit_extras.mjs --fixture=live.json      # -> _audit/<img>.jpg + .json
 *
 * The matcher is COPIED from site_suite.mjs on purpose: this tool has to ask
 * "what did the suite fail to match", so it must use the suite's own rule. If
 * that rule changes, change it here too — they are meant to agree.
 */
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execFileSync } from 'child_process';
import { loadGT } from './gt_load.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const arg = (k, d = '') =>
  (process.argv.find(a => a.startsWith(`--${k}=`)) || '').split('=').slice(1).join('=') || d;

const FIXTURE = arg('fixture', 'live.json');
const OUT = join(HERE, '_audit');
mkdirSync(OUT, { recursive: true });

// ── site_suite's matcher, verbatim ──────────────────────────────────────────
function parseDeweyStr(s) {
  const m = /^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]{1,4})?/.exec(String(s));
  return m ? { num: parseFloat(m[1]), cut: (m[2] || '').toUpperCase() } : null;
}
function numStr(s) { const m = /^\s*(\d+(?:\.\d+)?)/.exec(String(s)); return m ? m[1] : null; }
function surname(s) {
  s = String(s).trim().toUpperCase();
  const m = /^(VON\s+\w+|VAN\s+\w+|DE\s+\w+|LA\s+\w+)/.exec(s);
  if (m) return m[1].replace(/\s+/g, '');
  return s.split(/[,\s]+/)[0] || '';
}
function labelMatch(detected, gt, section) {
  if (section === 'nonfiction') {
    const dn = numStr(detected), gn = numStr(gt);
    const dm = parseDeweyStr(detected), gm = parseDeweyStr(gt);
    if (dn && gn && dm && gm) {
      const numOk = dn.startsWith(gn) || gn.startsWith(dn);
      const dc = (dm.cut || '').slice(0, 3), gc = (gm.cut || '').slice(0, 3);
      return numOk && !!dc && !!gc && dc === gc;
    }
    return false;
  }
  const ds = surname(detected), gs = surname(gt);
  const mn = Math.min(ds.length, gs.length);
  if (mn < 3) return ds === gs;
  if (mn >= 6) { let d = 0; for (let i = 0; i < mn; i++) if (ds[i] !== gs[i]) d++; return d <= 1; }
  return ds.slice(0, 4) === gs.slice(0, 4);
}

// ── Which emitted labels does GT not account for? ───────────────────────────
const GT = loadGT();
const live = JSON.parse(readFileSync(join(HERE, FIXTURE), 'utf-8'));

const plan = {};
let totalExtra = 0, totalEmitted = 0;
for (const r of (live.results || [])) {
  const g = GT[r.imgName];
  if (!g || !r.fixture) continue;
  const gtLabels = [];
  for (const row of (g.rows || [])) gtLabels.push(...row.labels);

  // Greedy one-to-one, same as matchToGt: a GT entry can only absorb one read.
  const used = new Set();
  gtLabels.forEach(gl => {
    for (let di = 0; di < r.fixture.length; di++) {
      if (used.has(di)) continue;
      if (labelMatch(r.fixture[di].spine_label, gl, r.section)) { used.add(di); break; }
    }
  });

  const extras = r.fixture
    .map((b, i) => ({ ...b, i }))
    .filter(b => !used.has(b.i) && b.spine_label && b._bbox);
  totalExtra += extras.length;
  totalEmitted += r.fixture.length;
  plan[r.imgName] = {
    section: r.section,
    emitted: r.fixture.length,
    accountedFor: used.size,
    extras: extras.map(b => ({
      text: b.spine_label, score: b._score ?? null, src: b._src ?? null,
      title: b.title ?? null, bbox: b._bbox,
    })),
  };
  console.log(`${r.imgName.replace('PXL_2026060', '').padEnd(20)} ` +
    `emitted ${String(r.fixture.length).padStart(3)}  ` +
    `matched ${String(used.size).padStart(3)}  ` +
    `unaccounted ${String(extras.length).padStart(3)}`);
}
console.log(`\n${totalExtra} of ${totalEmitted} emitted labels are not in ground truth.`);

writeFileSync(join(OUT, '_plan.json'), JSON.stringify(plan, null, 1));
console.log(`Wrote ${join(OUT, '_plan.json')}`);
console.log('Now run:  python audit_extras.py     # -> _audit/<img>.jpg contact sheets');
