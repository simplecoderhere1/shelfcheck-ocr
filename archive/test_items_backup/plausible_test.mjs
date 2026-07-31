/**
 * Unit test for labelPlausible — the shape gate that decides whether a read is
 * well-formed enough to be allowed to accuse a book of being misshelved.
 *
 * Two hard requirements, and they pull against each other:
 *   1. It must reject NOTHING that is really on the shelves. Every label in the
 *      ground-truth key must pass, or the gate is silencing real books.
 *   2. It must reject the malformed reads that caused false accusations —
 *      spine text promoted to a label, and two neighbouring stickers welded
 *      into one string.
 *
 * The KEEP list below is drawn from reads the app actually produced (see
 * `node replay_order.mjs --dump=`), not invented. The first version of this gate
 * demanded a comma in fiction labels, which every GT entry has — and which
 * almost no real Azure read has, because the assembler joins the surname and
 * given-name lines with a space. It demoted 90% of fiction, including the
 * genuinely misfiled WALLACE this app exists to catch. Hence these cases.
 *
 *   node plausible_test.mjs
 */
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { loadGT } from './gt_load.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(HERE, '..', '..', 'shelfcheck-ocr', 'index.html'), 'utf-8');
const src = html.slice(html.indexOf('// Fraction of pixels in a region'),
                       html.indexOf('// ── Gemini fusion (pure;'));
const { labelPlausible } = new Function(src + '; return { labelPlausible };')();

let fail = 0;

// ── 1. Every real ground-truth label must survive ───────────────────────────
const GT = loadGT(true);
let gtN = 0;
for (const [k, v] of Object.entries(GT)) {
  if (k.startsWith('_')) continue;
  for (const row of (v.rows || [])) {
    for (const l of row.labels) {
      gtN++;
      if (!labelPlausible(l, v.section)) { console.log(`  REJECTED a real label: [${v.section}] "${l}"`); fail++; }
    }
  }
}
console.log(`ground truth: ${gtN} label(s) checked, ${fail} wrongly rejected`);

// ── 2. Reads that must pass (real app output) ───────────────────────────────
const KEEP = [
  ['nonfiction', '635 WEA'], ['nonfiction', '641.5676 ZUS'], ['nonfiction', '635.0483 BAK'],
  ['nonfiction', '641.568 SOU 2019'], ['nonfiction', '635 FAS'],
  // Fiction as Azure actually returns it: no comma, or a full stop for one.
  ['fiction', 'WAGGONER Tim'], ['fiction', 'WAGNER Bruce'], ['fiction', 'WAKE Jules'],
  ['fiction', 'WALLACE. David Foster'], ['fiction', 'WALDMAN. Adelle'],
  ['fiction', 'WAITES, Martyn'], ['fiction', 'VUONG, Ocean'],
  ['fiction', 'VON ARNIM, Elizabeth'], ['fiction', 'WACLAWSKA, Karolina'],
  ['fiction', 'WAIT. Rebecca'], ['fiction', 'WAKEFIEL vikki'],
];
// ── 3. Reads that must be demoted ───────────────────────────────────────────
const DROP = [
  // Spine text and frame-edge fragments read as call numbers.
  ['nonfiction', '635 ICON'], ['nonfiction', '635 AK'], ['nonfiction', '604 FOTO'],
  ['nonfiction', '635 MCMA'], ['nonfiction', 'WACCIONES'],
  // Fiction fragments with no given name at all.
  ['fiction', 'VED'], ['fiction', 'WAIDN'], ['fiction', 'WAGGONER T'],
  // Two stickers welded into one.
  ['fiction', 'WACCIONES WAGGONS'], ['fiction', 'BOO WAL WALKER, Boo'],
  ['fiction', 'WALBER Kate ALDE'], ['fiction', 'WAGA WAGAMES RichardRichard'],
  ['fiction', 'IPEINHOLNZIEGESAR, REINHOLDRAIN Shola'], ['fiction', 'WAID WAID'],
  // Surname read so short it files under the wrong letter — "VON ARNIM,
  // Elizabeth" with the surname mostly missing. It accused a correctly
  // shelved book on 200004299.
  ['fiction', 'VON AI Elizabe'],
];

for (const [sec, t] of KEEP) {
  if (!labelPlausible(t, sec)) { console.log(`  should KEEP but demoted: [${sec}] "${t}"`); fail++; }
}
for (const [sec, t] of DROP) {
  if (labelPlausible(t, sec)) { console.log(`  should DEMOTE but kept:  [${sec}] "${t}"`); fail++; }
}
console.log(`keep list: ${KEEP.length} · drop list: ${DROP.length}`);

console.log(fail ? `\nFAIL — ${fail} case(s) wrong` : '\nPASS — gate agrees on every case');
process.exit(fail ? 1 : 0);
