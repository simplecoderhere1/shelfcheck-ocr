/**
 * Ordering logic test — runs the APP'S OWN checkOrder (extracted live from
 * index.html, no copy drift) against ground_truth_v2.json.
 *
 * GT rows record the exact physical left→right order on the shelf, which
 * includes a few genuinely misshelved books.  The app SHOULD flag exactly
 * those (true positives) and nothing else (no false positives).
 *
 * Usage:  node run_ordering_test.mjs
 */

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { loadGT } from './gt_load.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const GT = loadGT();

// ── Extract ordering code from the app ─────────────────────────────────────
const html = readFileSync(join(HERE, '..', '..', 'shelfcheck-ocr', 'index.html'), 'utf-8');
const start = html.indexOf('const SURNAME_PREFIXES');
const end   = html.indexOf('// ── Results display');
if (start < 0 || end < 0) { console.error('extraction markers not found'); process.exit(1); }
const app = new Function(html.slice(start, end) + `
  return { checkOrder, compareKeys, sortKey, parseSurnameAndFirst, parseDewey };
`)();

// ── Books the GT itself shows physically misshelved (true positives) ───────
// 195909571/195911082: "635.04 HEM" filed after SMI; "635.0483 BAK" after VIA.
// 200006014: "WALLACE, David Foster" filed before WALDMAN/WALDON/WALKER.
// 195903035: "635 GAR" filed between CHI and COL.
// 195909571 also has "635.0484 BEN" filed after BIG, and the app flags it live.
// It is deliberately NOT expected here: this harness feeds the GT labels
// complete with their uncertainty marking, and GT marks BEN uncertain because
// its trailing digit runs off the spine. Declining to judge a label flagged
// uncertain is the correct behaviour, so expecting a flag here would be
// expecting the app to guess. site_suite and replay_order DO expect it, because
// they run on real reads where the closer-look pass reads the sticker cleanly.
const EXPECTED_MISORDERS = {
  'PXL_20260607_195903035.jpg':    ['635 GAR'],
  'PXL_20260607_195909571.MP.jpg': ['635.04 HEM', '635.0483 BAK'],
  'PXL_20260607_195911082.jpg':    ['635.04 HEM', '635.0483 BAK'],
  'PXL_20260607_200006014.jpg':    ['WALLACE, David Foster'],
};

// ── Convert ground_truth_v2 rows → book objects ───────────────────────────
function gtToBooks(data) {
  const books = [];
  for (const row of (data.rows || [])) {
    const uncertainCounts = {};
    for (const l of (row.uncertain || [])) uncertainCounts[l] = (uncertainCounts[l] || 0) + 1;
    const usedUncertain = {};
    for (const label of row.labels) {
      const remaining = (uncertainCounts[label] || 0) - (usedUncertain[label] || 0);
      const isUncertain = remaining > 0;
      if (isUncertain) usedUncertain[label] = (usedUncertain[label] || 0) + 1;
      books.push({
        spine_label: label,
        shelfRow: row.row,
        ...(isUncertain ? { confidence: 'low' } : {}),
      });
    }
  }
  return books;
}

// ── Run tests ──────────────────────────────────────────────────────────────
const entries = Object.entries(GT).filter(([k]) => !k.startsWith('_'));
let totalBooks = 0, totalUnv = 0, totalFP = 0, totalTP = 0, totalFN = 0;
let sectionCorrect = 0, failImages = 0;

console.log('ShelfCheck ordering test v3 — app code extracted from index.html');
console.log('='.repeat(66));

for (const [imgName, data] of entries) {
  const section = data.section;
  const gtBooks = gtToBooks(data);
  const checked = app.checkOrder(gtBooks, section);

  const flagged  = checked.filter(b => b.outOfOrder).map(b => b.spine_label);
  const expected = [...(EXPECTED_MISORDERS[imgName] || [])];

  // multiset compare: every expected flag present, no extra flags
  const expPool = [...expected];
  const fpList = [];
  for (const f of flagged) {
    const i = expPool.indexOf(f);
    if (i >= 0) expPool.splice(i, 1);   // true positive
    else fpList.push(f);                 // unexpected flag = false positive
  }
  const fnList = expPool;                // expected but not flagged = miss

  const unv = checked.filter(b => b.unverifiable).length;
  totalBooks += gtBooks.length; totalUnv += unv;
  totalFP += fpList.length; totalFN += fnList.length;
  totalTP += expected.length - fnList.length;

  const deweyish = gtBooks.filter(b => /^\s*\d{1,3}(\.\d+)?(\s|$)/.test(String(b.spine_label))).length;
  const ratio = deweyish / gtBooks.length;
  const detected = ratio > 0.6 ? 'nonfiction' : (ratio < 0.15 ? 'fiction' : 'ambiguous');
  if (detected === section) sectionCorrect++;

  const ok = fpList.length === 0 && fnList.length === 0;
  if (!ok) failImages++;
  const short = imgName.replace('PXL_2026060', '').replace('.jpg', '').replace('.MP', '');
  console.log(`[${ok ? 'OK  ' : 'FAIL'}] ${short.padEnd(16)} section=${section.padEnd(10)} books=${String(gtBooks.length).padEnd(3)} flags=${flagged.length} (expected ${expected.length})  unv=${unv}`);
  for (const f of fpList) console.log(`       FALSE POSITIVE: "${f}"`);
  for (const f of fnList) console.log(`       MISSED MISORDER: "${f}"`);
}

console.log('='.repeat(66));
console.log(`Total books        : ${totalBooks}`);
console.log(`Unverifiable       : ${totalUnv}`);
console.log(`True positives     : ${totalTP} (known misshelved books correctly flagged)`);
console.log(`False positives    : ${totalFP}  (rate ${(totalFP / totalBooks * 100).toFixed(1)}%)`);
console.log(`Missed misorders   : ${totalFN}`);
console.log(`Section detection  : ${sectionCorrect}/${entries.length}`);
process.exit(failImages ? 1 : 0);
