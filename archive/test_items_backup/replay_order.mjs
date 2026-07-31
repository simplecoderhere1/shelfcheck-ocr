/**
 * Replay ordering against REAL degraded reads.
 *
 * Why this exists: run_ordering_test.mjs feeds checkOrder clean ground-truth
 * labels, where it already scores 5 TP / 0 FP. Every live detection failure
 * happens for a different reason — a neighbour read that is single-engine,
 * truncated, or simply wrong — so the offline suite is blind to them. This
 * harness replays the ACTUAL fused labels a live run produced, which makes the
 * failures reproducible in ~50ms instead of a ~15-minute live run.
 *
 * Capture a fixture first:
 *   FIXTURE=1 node site_suite.mjs --json=live.json
 *   node replay_order.mjs --make=live.json          # -> order_fixture.json
 *
 * Then iterate:
 *   node replay_order.mjs                            # score current app code
 *   node replay_order.mjs --dump=195909571           # show one shelf's sequence
 *
 * Like the other suites, the ordering code is EXTRACTED from index.html — never
 * copied — so it can't drift from what ships.
 */
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
const arg = (k, d = '') =>
  (process.argv.find(a => a.startsWith(`--${k}=`)) || '').split('=').slice(1).join('=') || d;
const FIXTURE = join(HERE, 'order_fixture.json');

// ── Build a fixture from a site_suite --json run ────────────────────────────
const MAKE = arg('make');
if (MAKE) {
  const src = JSON.parse(readFileSync(join(HERE, MAKE), 'utf-8'));
  const out = {};
  for (const r of src.results || []) {
    if (!r.fixture) continue;
    out[r.imgName] = { section: r.section, books: r.fixture };
  }
  if (!Object.keys(out).length) {
    console.error('No fixture data in that file — re-run site_suite with FIXTURE=1');
    process.exit(1);
  }
  writeFileSync(FIXTURE, JSON.stringify(out, null, 2));
  console.log(`Wrote ${FIXTURE} — ${Object.keys(out).length} image(s), ` +
              `${Object.values(out).reduce((n, v) => n + v.books.length, 0)} books`);
  process.exit(0);
}

if (!existsSync(FIXTURE)) {
  console.error(`No fixture yet. Run:\n  FIXTURE=1 node site_suite.mjs --json=live.json\n` +
                `  node replay_order.mjs --make=live.json`);
  process.exit(1);
}

// ── Extract the app's ordering code (identical markers to run_ordering_test) ─
const html = readFileSync(join(HERE, '..', '..', 'shelfcheck-ocr', 'index.html'), 'utf-8');
const start = html.indexOf('const SURNAME_PREFIXES');
const end   = html.indexOf('// ── Results display');
if (start < 0 || end < 0) { console.error('extraction markers not found'); process.exit(1); }
const app = new Function(html.slice(start, end) + `
  return { checkOrder };
`)();

// Authoritative expectation, same list run_ordering_test.mjs uses.
const EXPECTED_MISORDERS = {
  'PXL_20260607_195903035.jpg':    ['635 GAR'],
  'PXL_20260607_195909571.MP.jpg': ['635.04 HEM', '635.0483 BAK', '635.0484 BEN'],
  'PXL_20260607_195911082.jpg':    ['635.04 HEM', '635.0483 BAK'],
  'PXL_20260607_200006014.jpg':    ['WALLACE, David Foster'],
};

const norm = s => String(s || '').toUpperCase().replace(/\s+/g, ' ').trim();
const numOf = s => (/^\s*(\d+(?:\.\d+)?)/.exec(String(s)) || [])[1] || null;
const cutOf = s => ((/^\s*\d+(?:\.\d+)?\s*([A-Za-z]{2,})/.exec(String(s)) || [])[1] || '').toUpperCase();
// Trailing punctuation is an OCR artifact, not part of the surname: Azure reads
// "WALLACE." for "WALLACE,". Leaving it in made this harness score the ONE real
// fiction detection as an FP *and* an FN, disagreeing with site_suite (whose
// fiction matcher compares letters and so matched it).
const surOf = s => (norm(s).split(/[,\s]+/)[0] || '').replace(/[^A-Z0-9]/g, '');
function flagMatches(flagged, expected, section) {
  if (section === 'fiction') return surOf(flagged) === surOf(expected);
  // Prefix, not equality — the same rule site_suite's matcher uses. A number
  // whose trailing digits run off the curve of the spine is read short but
  // correctly: the app flags "635.048 BEN" for a sticker whose full number is
  // 635.0484. Demanding equality scored that ONE real detection as an FP *and*
  // an FN, and put this harness in disagreement with site_suite over a book
  // both of them had actually got right.
  const a = numOf(flagged), b = numOf(expected);
  if (!a || !b || !(a === b || a.startsWith(b) || b.startsWith(a))) return false;
  return cutOf(flagged) === cutOf(expected);
}

const fx = JSON.parse(readFileSync(FIXTURE, 'utf-8'));
const DUMP = arg('dump');
let TP = 0, FP = 0, FN = 0;

console.log('REPLAY — app checkOrder over real live reads');
console.log('='.repeat(92));
for (const [img, { section, books }] of Object.entries(fx)) {
  const checked = app.checkOrder(books.map(b => ({ ...b })), section);
  const flagged = checked.filter(b => b.outOfOrder).map(b => b.spine_label);
  const expected = [...(EXPECTED_MISORDERS[img] || [])];

  const pool = [...expected];
  let tp = 0;
  const fps = [];
  for (const f of flagged) {
    const hit = pool.findIndex(e => flagMatches(f, e, section));
    if (hit >= 0) { pool.splice(hit, 1); tp++; } else fps.push(f);
  }
  TP += tp; FP += fps.length; FN += pool.length;

  const short = img.replace('PXL_2026060', '').replace('.MP', '').replace('.jpg', '');
  const ok = !fps.length && !pool.length;
  console.log(`${ok ? 'OK  ' : 'FAIL'} ${short.padEnd(14)} ${section.padEnd(11)} ` +
              `books=${String(books.length).padStart(3)}  TP${tp} FP${fps.length} FN${pool.length}`);
  for (const f of fps)  console.log(`       FP: "${f}"`);
  for (const m of pool) console.log(`       MISSED: "${m}"`);

  if (DUMP && img.includes(DUMP)) {
    console.log('  --- sequence ---');
    for (const b of checked) {
      const state = b.outOfOrder ? 'RED ' : b.unverifiable ? 'unv ' :
                    b.possiblyOutOfOrder ? 'poss' : 'ok  ';
      console.log(`    ${state} sc=${b._score ?? '-'} src=${b._src ?? '-'} row=${b.shelfRow} "${b.spine_label}"`);
    }
  }
}
console.log('='.repeat(92));
const rate = (TP + FN) ? (TP / (TP + FN) * 100).toFixed(0) : '—';
console.log(`TP ${TP}   FP ${FP}   FN ${FN}   detection ${TP}/${TP + FN} (${rate}%)`);
