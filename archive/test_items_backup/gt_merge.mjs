/**
 * Merge the per-image verification files into ground_truth_v3.json.
 *
 * Input:  _gt/_plan.json            what the automation read, in physical order
 *         _gt/verified/<img>.json   a human/vision pass over every one of those
 *                                   reads, plus the stickers it missed
 * Output: ground_truth_v3.json      the same shape as v2, so every existing
 *                                   harness keeps working unchanged
 *
 * Why v3 exists: v2 listed roughly half the books physically on these shelves.
 * On 195911082, 17 of 17 labels it did not account for turned out to be real,
 * legible stickers. Recall measured against v2 was therefore flattering (the
 * denominator was too small) and precision was meaningless (reads were counted
 * wrong for matching nothing, when the book was simply absent from the key).
 *
 * Ordering is load-bearing. run_ordering_test replays each row as a physical
 * left-to-right sequence and expects the app to flag exactly the books that are
 * genuinely out of place, so a row sorted into filing order would silently
 * erase every misshelving the test is meant to catch. Reads keep their x-order
 * from the plan; a found sticker is spliced in between the two labels the
 * verifier said it sits between.
 *
 *   node gt_merge.mjs            # -> ground_truth_v3.json
 *   node gt_merge.mjs --check    # report only, write nothing
 */
import { readFileSync, writeFileSync, existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
const GT_DIR = join(HERE, '_gt');
const VER_DIR = join(GT_DIR, 'verified');
const CHECK = process.argv.includes('--check');

const plan = JSON.parse(readFileSync(join(GT_DIR, '_plan.json'), 'utf-8'));
const v2 = existsSync(join(HERE, 'ground_truth_v2.json'))
  ? JSON.parse(readFileSync(join(HERE, 'ground_truth_v2.json'), 'utf-8')) : {};

// The shape gate the app itself applies, restated here so the key can be
// checked against it. A ground-truth label that fails this would mean either a
// bad transcription or a genuine exception to the collection's convention —
// either way it needs a human look before it becomes the answer key.
const PLAUSIBLE_NF = /^\s*\d{1,3}(?:\.\d+)?\s+[A-Za-z]{3}\s*$/;
function plausible(text, section) {
  const t = String(text || '').trim();
  if (!t) return false;
  if (section !== 'fiction') return PLAUSIBLE_NF.test(t.replace(/\s+(?:v\.?\s*)?\d{1,4}\s*$/i, ''));
  const p = t.split(',');
  if (p.length !== 2) return false;
  const sur = p[0].trim().split(/\s+/);
  return !!p[1].trim() && sur[sur.length - 1].length <= 14;
}

// Two clean-ups the verified files need before they can be an answer key.
//
// 1. A verdict of `ok` means "the sticker says what the caption says", and the
//    caption is the AUTOMATION's raw read — which for fiction routinely drops
//    the comma ("WALKER Boo") or reads it as a full stop ("WALKER. Megan").
//    The key must record what is on the sticker, so put the comma back: the
//    surname is the leading ALL-CAPS run, the given name is the rest.
// 2. A transcriber marking an unreadable character sometimes writes a
//    placeholder ("635.? ROS", "635.? CO_"). A placeholder is not a reading.
//    Keep only the part that was actually legible and mark the label uncertain,
//    which is the same treatment any half-legible sticker gets.
function normalise(text, section) {
  let t = String(text || '').trim();
  let uncertain = false;
  if (/[?_]/.test(t)) {
    uncertain = true;
    t = t.replace(/[?_]/g, '').replace(/\.\s/g, ' ').replace(/\s+/g, ' ').trim();
  }
  if (section === 'fiction' && t && !t.includes(',')) {
    const toks = t.replace(/\./g, ' ').split(/\s+/).filter(Boolean);
    const isUpper = w => w === w.toUpperCase() && /[A-Z]/.test(w);
    let i = toks.length;
    while (i > 0 && !isUpper(toks[i - 1])) i--;
    if (i > 0 && i < toks.length) t = toks.slice(0, i).join(' ') + ', ' + toks.slice(i).join(' ');
  }
  return { text: t, uncertain };
}

const out = {
  _notes: 'Human-verified ground truth v3 (2026-07-29). Rebuilt because v2 listed ' +
    'only about half the books physically present: a crop-by-crop audit of the ' +
    'labels v2 did not account for found them to be real, legible stickers ' +
    '(17 of 17 on 195911082, 17 of 17 on 195849222). Every label here was ' +
    'checked against the pixels of the native photo, and each shelf was swept ' +
    'again for stickers no automated read had boxed. Labels are in physical ' +
    'left-to-right order per row — do NOT sort them, the ordering tests replay ' +
    'the row as a sequence. "uncertain" lists labels whose sticker is real but ' +
    'not confidently readable (blurred, wrapped round the spine, cropped by the ' +
    'frame); they are excluded from the recall denominator but kept for ordering.',
};

let totV3 = 0, totV2 = 0, totJunk = 0, totCorrected = 0, totFound = 0, totUncertain = 0;
const problems = [];
const suspicions = [];

for (const [img, rec] of Object.entries(plan)) {
  const vf = join(VER_DIR, img.replace(/\.jpg$/, '') + '.json');
  if (!existsSync(vf)) { problems.push(`${img}: NO verification file yet`); continue; }
  let ver;
  try { ver = JSON.parse(readFileSync(vf, 'utf-8')); }
  catch (e) { problems.push(`${img}: verification file is not valid JSON — ${e.message}`); continue; }

  const byN = new Map((ver.labels || []).map(l => [l.n, l]));
  const seq = [];                       // {row, text, uncertain}
  for (const l of rec.labels) {
    const v = byN.get(l.n);
    if (!v) { problems.push(`${img}: label ${l.n} ("${l.text}") not verified`); continue; }
    if (v.verdict === 'junk') { totJunk++; continue; }
    if (v.verdict === 'corrected') totCorrected++;
    if (v.verdict === 'illegible') totUncertain++;
    const raw = (v.verdict === 'corrected' || v.verdict === 'illegible')
      ? (v.text || l.text) : l.text;
    const norm = normalise(raw, rec.section);
    // A verdict of `ok` confirms the SURNAME the automation read, not that its
    // caption is the whole label — several reads are right as far as they go
    // and simply stop early ("VONNEG" for VONNEGUT, Kurt). Where the recorded
    // text is not a well-formed label, keep it but mark it uncertain: the book
    // holds its place in the row without the key asserting a reading it does
    // not have. Completing it from the neighbours would be a guess, and this
    // key does not guess.
    const uncertain = v.verdict === 'illegible' || norm.uncertain ||
                      !plausible(norm.text, rec.section);
    seq.push({ n: l.n, row: l.row, text: norm.text, uncertain });
  }

  // Splice found stickers in between the two reads the verifier placed them by.
  for (const m of (ver.missing || [])) {
    totFound++;
    // betweenN is [leftLabel, rightLabel]. A null LEFT means the sticker sits
    // before the first read on its row — which is a real case: on 195858362 the
    // very first book on the shelf had no box at all. Falling back to "append at
    // the end" would file it after everything and invert the row, and the
    // ordering tests replay the row as a physical sequence.
    const [after, before] = Array.isArray(m.betweenN) ? m.betweenN : [null, null];
    const anchorN = after ?? before;
    const row = m.row ?? (seq.find(s => s.n === anchorN)?.row ?? 0);
    let at = seq.length;
    if (after != null) {
      const i = seq.findIndex(s => s.n === after);
      if (i >= 0) at = i + 1;
    } else if (before != null) {
      const i = seq.findIndex(s => s.n === before);
      if (i >= 0) at = i;
    }
    // A found sticker can be certainly PRESENT but not certainly READABLE — the
    // last book on 195901698 row 1 is shot at a steep angle and its middle tag
    // letter is a guess. Recording it as uncertain keeps it in the physical
    // sequence (so ordering still sees a book there) while keeping it out of the
    // recall denominator, which is the same treatment a half-legible sticker
    // gets anywhere else in this key.
    const norm = normalise(m.text, rec.section);
    // A sticker that is definitely THERE but completely unreadable still has to
    // occupy its place in the row: the ordering tests replay the row as a
    // physical sequence, and silently omitting a book would close a gap that
    // really exists. Give it a placeholder and mark it uncertain, so it holds
    // its position without ever counting as a label the app failed to read.
    seq.splice(at, 0, { n: null, row, text: norm.text || '(unreadable)',
                        uncertain: !norm.text || !!m.uncertain || norm.uncertain });
  }

  const rows = [];
  for (const row of [...new Set(seq.map(s => s.row))].sort((a, b) => a - b)) {
    const items = seq.filter(s => s.row === row);
    rows.push({
      row,
      labels: items.map(s => s.text),
      uncertain: items.filter(s => s.uncertain).map(s => s.text),
    });
  }

  for (const r of rows) {
    // Only well-formed READINGS are held to the shape rules. A label marked
    // uncertain is, by definition, not a reading — it records that a sticker is
    // physically there and could not be fully made out. Demanding it look like
    // a valid label would force exactly the guessing this key refuses to do.
    const unc = new Set(r.uncertain);
    for (const t of r.labels) {
      if (unc.has(t)) continue;
      if (!plausible(t, rec.section)) problems.push(`${img} row ${r.row}: implausible label "${t}"`);
    }
    // Truncation smell. A transcriber reading off a crop can copy the caption's
    // dropped digit instead of the sticker's real one — "641.5 RIG" in a row of
    // 641.568s. The tell is that the number is a strict PREFIX of several
    // neighbours: real shelves do carry both 635.04 and 635.048, but a number
    // that prefixes three or more of its row-mates is far more likely a short
    // read than a genuinely shorter call number. Flagged for a human look, not
    // auto-corrected — guessing the hidden digit is exactly what this project
    // forbids the app to do, and the key must hold itself to the same rule.
    // Only the IMMEDIATE neighbours count. A Dewey shelf legitimately runs 635,
    // 635.01, 635.02, so "prefixes many labels on this row" is normal and says
    // nothing. What is odd is a number sandwiched between two LONGER numbers
    // that both extend it — "641.5 RIG" between two 641.568s — because the
    // shelf is sorted, so a genuinely shorter call number would sort ahead of
    // both, not sit between them.
    const num = t => (/^\s*(\d+(?:\.\d+)?)/.exec(t) || [])[1];
    r.labels.forEach((t, i) => {
      const me = num(t), prev = num(r.labels[i - 1] || ''), next = num(r.labels[i + 1] || '');
      if (!me || !prev || !next) return;
      const extends_ = n => n !== me && n.startsWith(me);
      if (!extends_(prev) || !extends_(next)) return;
      suspicions.push(`${img} row ${r.row}: "${t}" between "${r.labels[i - 1]}" and "${r.labels[i + 1]}"`);
      // Truncation-shaped, so the key must not assert an order for it. The
      // trailing digits are not legible on the sticker — that is WHY the
      // recorded number is short — and the sorted position of "635.04 COF"
      // between two 635.048s depends entirely on the digit nobody can read.
      // Left certain, the key demands the app flag a book as misshelved on
      // evidence the photograph does not contain. The app already declines to
      // judge its own truncated reads; the answer key is held to the same rule.
      if (!r.uncertain.includes(t)) r.uncertain.push(t);
    });
  }

  out[img] = { section: rec.section, rows };
  const n = rows.reduce((a, r) => a + r.labels.length, 0);
  const nUnc = rows.reduce((a, r) => a + r.uncertain.length, 0);
  const old = (v2[img]?.rows || []).reduce((a, r) => a + r.labels.length, 0);
  totV3 += n; totV2 += old;
  console.log(`${img.replace('PXL_2026060', '').padEnd(20)} v2 ${String(old).padStart(3)} -> ` +
    `v3 ${String(n).padStart(3)}  (${nUnc} uncertain)`);
}

console.log('\n' + '='.repeat(64));
console.log(`labels        v2 ${totV2}  ->  v3 ${totV3}   (+${totV3 - totV2})`);
console.log(`reads corrected ${totCorrected} · dropped as junk ${totJunk} · illegible ${totUncertain}`);
console.log(`stickers found that no read had boxed: ${totFound}`);
if (problems.length) {
  console.log(`\n${problems.length} problem(s) — resolve before trusting this key:`);
  for (const p of problems) console.log('  ' + p);
}
if (suspicions.length) {
  console.log(`\n${suspicions.length} label(s) worth a second look (not blocking):`);
  for (const s of suspicions) console.log('  ' + s);
}
if (CHECK) { console.log('\n--check: nothing written'); process.exit(problems.length ? 1 : 0); }

writeFileSync(join(HERE, 'ground_truth_v3.json'), JSON.stringify(out, null, 1));
console.log(`\nWrote ground_truth_v3.json`);
