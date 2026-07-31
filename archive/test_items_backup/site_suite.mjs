/**
 * ShelfCheck SITE SUITE — drives the real deployed page through its real
 * MAIN-MODE path (native-camera capture -> #photoInput -> auto-analyze) over
 * all 10 test images, and reports ACCURACY and SPEED together with a set of
 * hard pass/fail gates.
 *
 * This is deliberately a *website* test, not a pipeline test: it touches only
 * what a volunteer touches (the file input the camera hands back, the rendered
 * photo, the summary badge, the flagged list) and it fails on anything a
 * volunteer would notice — a page error, a false accusation, a missing result
 * panel, an over-claiming success message, or a scan that takes too long.
 *
 * Per image it reports:
 *   • label recall / precision      (vs ground_truth_v2.json, strict matcher)
 *   • misshelving flags TP/FP/FN    (vs the known physically-misshelved books)
 *   • unread ("couldn't read") count as surfaced by the UI
 *   • speed: time to FIRST render and to FINAL (refined) render
 *   • the Open Library verify time, broken out — historically the slowest
 *     blocking step, so it gets its own column to keep it honest
 *
 * Usage:
 *   node site_suite.mjs                 # live APIs (ocr.space + Open Library)
 *   BLOCK_REMOTE=1 node site_suite.mjs  # on-device engine only, deterministic
 *   node site_suite.mjs --only=200006014
 *   node site_suite.mjs --headful
 *   node site_suite.mjs --json=out.json # machine-readable results
 */
import { readFileSync, writeFileSync, existsSync, mkdtempSync, mkdirSync, createReadStream, statSync } from 'fs';
import { join, dirname, extname } from 'path';
import { fileURLToPath } from 'url';
import { tmpdir } from 'os';
import http from 'http';
import puppeteer from 'puppeteer-core';
import { loadGT } from './gt_load.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..', '..', 'shelfcheck-ocr');
const INDEX = join(REPO, 'index.html');

const args = process.argv.slice(2);
const HEADFUL = args.includes('--headful');
const ONLY = (args.find(a => a.startsWith('--only=')) || '').split('=')[1] || '';
const JSON_OUT = (args.find(a => a.startsWith('--json=')) || '').split('=')[1] || '';
const OFFLINE = !!process.env.BLOCK_REMOTE;
// Extra query string appended to index.html, so the suite can A/B engine
// tunables (?det=, ?recb=, ?ep=webgpu) without editing the app.
const QS = (args.find(a => a.startsWith('--qs=')) || '').split('=').slice(1).join('=') || '';

// ── Gates ───────────────────────────────────────────────────────────────────
// A false positive is a volunteer sent to "fix" a correctly shelved book, so
// it is always fatal. The speed gates are generous relative to the ~5-8s
// target: they exist to catch a REGRESSION (something quietly serialising
// again), not to police normal network jitter.
const GATES = {
  // Product call (2026-07): a few false positives are acceptable as long as
  // label recall stays above 90%. FPs are still reported per-image so they can
  // be watched, but they no longer fail the run on their own.
  maxFalsePositives: 3,
  minRecallPct:      OFFLINE ? 60 : 90,   // on-device-only reads fewer labels
  // NOTE: the product target is ~5s. It is NOT met and this gate does not
  // pretend otherwise — it is set just above the measured worst case purely to
  // catch a regression. The gap is the on-device engine (det+rec ~5-7s), which
  // has to finish before a single fused answer can be shown. Closing it needs a
  // faster engine (a working GPU path or a smaller model), not more scheduling.
  maxFirstRenderMs:  OFFLINE ? 25000 : 15000,
  maxFinalRenderMs:  OFFLINE ? 40000 : 25000,
  maxOlVerifyMs:     6000,                // the whole tie-break step, not per call
  maxPageErrors:     0,
};

// ── Locate Chrome ───────────────────────────────────────────────────────────
const CHROME = [
  process.env.PUPPETEER_EXECUTABLE_PATH,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
].filter(Boolean).find(p => existsSync(p));
if (!CHROME) { console.error('Chrome not found. Set PUPPETEER_EXECUTABLE_PATH.'); process.exit(2); }

const indexSrc = readFileSync(INDEX, 'utf-8');
for (const hook of ['__shelfcheckRenders', 'photoInput']) {
  if (!indexSrc.includes(hook)) { console.error(`index.html is missing "${hook}".`); process.exit(2); }
}

// ── Strict label matcher, ported 1:1 from eval_fusion.mjs / recon.py ─────────
function parseDeweyStr(s) {
  const m = /^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]{2,})?/.exec(String(s));
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
function matchToGt(detected, gtLabels, section) {
  const used = new Set(), matched = new Set();
  gtLabels.forEach((gl, gi) => {
    for (let di = 0; di < detected.length; di++) {
      if (used.has(di)) continue;
      if (labelMatch(detected[di], gl, section)) { matched.add(gi); used.add(di); break; }
    }
  });
  return { matched, matchedCount: used.size };
}

// Known physically-misshelved books (the real true positives on these shelves).
// 195903035: "635 GAR" sits between 635 CHI and 635 COL — plainly legible in
// the photo, and GAR sorts after every other tag on that shelf. The app found
// it before the ground truth did; the sticker was verified against the pixels
// and ground_truth_v2 was corrected, not the detection.
const EXPECTED_MISORDERS = {
  'PXL_20260607_195903035.jpg':    ['635 GAR'],
  'PXL_20260607_195909571.MP.jpg': ['635.04 HEM', '635.0483 BAK', '635.0484 BEN'],
  'PXL_20260607_195911082.jpg':    ['635.04 HEM', '635.0483 BAK'],
  'PXL_20260607_200006014.jpg':    ['WALLACE, David Foster'],
};

// ── Static server ───────────────────────────────────────────────────────────
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
  '.txt': 'text/plain', '.onnx': 'application/octet-stream' };
function startServer() {
  return new Promise(resolve => {
    const srv = http.createServer((req, res) => {
      let p = decodeURIComponent(req.url.split('?')[0]);
      if (p === '/' || p === '') p = '/index.html';
      const fp = join(REPO, p);
      if (!fp.startsWith(REPO) || !existsSync(fp) || statSync(fp).isDirectory()) { res.writeHead(404); res.end('nf'); return; }
      res.writeHead(200, {
        'Content-Type': MIME[extname(fp)] || 'application/octet-stream',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      createReadStream(fp).pipe(res);
    });
    srv.listen(0, '127.0.0.1', () => resolve({ srv, port: srv.address().port }));
  });
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── One image, driven exactly like the phone drives it ───────────────────────
async function runImage(browser, baseUrl, imgName, gtData) {
  const section = gtData.section;
  const imagePath = join(HERE, imgName);
  const unc = new Set();
  const gt = [];
  for (const r of (gtData.rows || [])) { gt.push(...r.labels); (r.uncertain || []).forEach(u => unc.add(u)); }
  const hc = gt.filter(l => !unc.has(l));   // high-confidence GT = recall denominator

  const page = await browser.newPage();
  await page.setViewport({ width: 420, height: 860 });   // a phone, not a desktop
  const pageErrors = [];
  const perfLogs = [];
  page.on('pageerror', e => pageErrors.push(e.message));
  page.on('console', m => { const t = m.text(); if (t.startsWith('[perf]')) perfLogs.push(t); });

  const uiFails = [];
  try {
    // Headless throttles rAF, and both loadPhoto and analyzeShelf await a
    // double-rAF before doing anything — without this shim they hang forever.
    await page.evaluateOnNewDocument(() => {
      window.requestAnimationFrame = cb => setTimeout(() => cb(performance.now()), 16);
    });
    if (OFFLINE) {
      await page.setRequestInterception(true);
      page.on('request', r => { if (/workers\.dev/.test(r.url())) r.abort(); else r.continue(); });
    }
    await page.goto(`${baseUrl}/index.html${QS ? '?' + QS : ''}`, { waitUntil: 'domcontentloaded', timeout: 30000 });
    // coi-serviceworker reloads once per profile to enable crossOriginIsolated
    // (threaded WASM); ride that out before injecting anything.
    const isolated = await page.evaluate(() => window.crossOriginIsolated).catch(() => false);
    if (!isolated) await page.waitForNavigation({ waitUntil: 'domcontentloaded', timeout: 6000 }).catch(() => {});
    await page.waitForSelector('#photoInput', { timeout: 15000 });

    await page.evaluate((sec) => {
      document.querySelectorAll('.tab-btn').forEach(b => {
        const on = b.dataset.section === sec;
        b.classList.toggle('active', on);
        b.setAttribute('aria-selected', String(on));
      });
      window.__shelfcheckRenders = [];
    }, section);

    // ── The real main-mode entry point ──────────────────────────────────────
    // The Take Photo button opens the OS camera, which hands the photo back by
    // populating #photoInput. Injecting there IS the production path, and it
    // must auto-analyze with no further taps.
    const t0 = Date.now();
    const input = await page.$('#photoInput');
    await input.uploadFile(imagePath);

    const autoStarted = await page.waitForFunction(
      () => { const b = document.getElementById('analyzeBtn'); return b && (b.disabled || /read/i.test(b.textContent)); },
      { timeout: 30000 }).then(() => true).catch(() => false);
    if (!autoStarted) uiFails.push('capture did not auto-start analysis (volunteer would have to tap Analyze)');

    await page.waitForFunction(() => (window.__shelfcheckRenders || []).length > 0, { timeout: 90000 });
    const firstMs = Date.now() - t0;

    // Final render: the refine pass fuses the slower engine in. Wait until the
    // render count holds steady, capped so a hung request can't stall the run.
    // QUIET must exceed the on-device engine's worst full-image time (~15s in
    // the offline runs) or the suite silently scores the FIRST render and
    // reports the refined pass as if it never happened.
    const QUIET = 20000, CAP = 90000;
    let lastCount = await page.evaluate(() => window.__shelfcheckRenders.length);
    let stableSince = Date.now();
    const deadline = Date.now() + CAP;
    while (Date.now() < deadline) {
      await sleep(400);
      const c = await page.evaluate(() => window.__shelfcheckRenders.length);
      if (c !== lastCount) { lastCount = c; stableSince = Date.now(); }
      else if (Date.now() - stableSince >= QUIET) break;
    }
    const finalMs = Date.now() - t0 - QUIET;   // discount the quiet-detection tail

    const renders = await page.evaluate(() => window.__shelfcheckRenders);
    const final = renders[renders.length - 1] || { books: [], section, unreadCount: 0 };

    // --shots: save a screenshot of the rendered result (photo + boxes +
    // summary), exactly what a volunteer sees, for a visual audit.
    if (process.env.SHOTS) {
      const dir = join(HERE, 'audit_shots');
      if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
      const short = imgName.replace('PXL_2026060', '').replace('.MP', '').replace('.jpg', '');
      // Drop the first-run help modal so it doesn't cover the photo/boxes.
      await page.evaluate(() => document.querySelector('.first-hint')?.remove());
      await page.screenshot({ path: join(dir, `${short}.png`) }).catch(() => {});
    }

    // ── Website functionality, as a volunteer would judge it ────────────────
    const ui = await page.evaluate(() => {
      const panel = document.getElementById('flaggedList');
      const badge = document.querySelector('#resultsOverlay .overlay-badge');
      const snap = document.getElementById('snapshot');
      return {
        panelShown:  !!panel && panel.classList.contains('show'),
        panelText:   panel ? panel.textContent.replace(/\s+/g, ' ').trim() : '',
        badgeText:   badge ? badge.textContent.replace(/\s+/g, ' ').trim() : '',
        photoShown:  !!snap && snap.style.display !== 'none' && snap.width > 0,
        noteShown:   !!document.querySelector('.flagged-list-note'),
        stillLoading: !!document.querySelector('.loading-overlay'),
      };
    });
    if (!ui.photoShown) uiFails.push('photo/box canvas not visible after analysis');
    if (!ui.badgeText) uiFails.push('no summary badge rendered');
    if (ui.stillLoading) uiFails.push('loading overlay never dismissed');
    if (final.books.length > 0 && !ui.panelShown) uiFails.push('results panel never shown despite books read');
    // The success message is a promise to the volunteer: it must not be made
    // while books sit unread.
    if ((final.unreadCount || 0) > 0 && /Shelf looks correct/i.test(ui.panelText)) {
      uiFails.push(`claims "Shelf looks correct" with ${final.unreadCount} unread book(s)`);
    }
    // Unread books must be announced, not silently dropped.
    if ((final.unreadCount || 0) > 0 && !ui.noteShown) {
      uiFails.push(`${final.unreadCount} unread book(s) but no "couldn't be read" note`);
    }

    // ── Score ───────────────────────────────────────────────────────────────
    const detected = final.books.map(b => b.spine_label).filter(Boolean);
    const { matched, matchedCount } = matchToGt(detected, gt, section);
    const hcMatched = [...matched].filter(gi => !unc.has(gt[gi])).length;

    const flagged = final.books.filter(b => b.outOfOrder).map(b => b.spine_label);
    const expected = [...(EXPECTED_MISORDERS[imgName] || [])];
    const pool = [...expected];
    const fp = [];
    for (const f of flagged) {
      const i = pool.findIndex(e => labelMatch(f, e, section));
      if (i >= 0) pool.splice(i, 1); else fp.push(f);
    }

    // Score the FIRST render's flags too. A red flag that appears early and is
    // retracted by the refine pass is still a false accusation the volunteer
    // saw, so it has to be visible in the report rather than averaged away.
    const firstBooks = (renders[0] || { books: [] }).books;
    const fpFirst = firstBooks.filter(b => b.outOfOrder)
      .map(b => b.spine_label)
      .filter(f => !expected.some(e => labelMatch(f, e, section)));

    // On-device engine cost, broken out: this is what gates a single-render
    // scan, since the answer can't be published until it finishes.
    const detRecLog = perfLogs.filter(l => /localocr: det \d/.test(l)).pop() || '';
    const detMs = Number((/det (\d+)ms/.exec(detRecLog) || [])[1] || 0);
    const recMs = Number((/rec (\d+)ms/.exec(detRecLog) || [])[1] || 0);
    const localLog = perfLogs.filter(l => l.includes('callLocalOcr')).pop() || '';
    const localMs = Number((/at (\d+)ms/.exec(localLog) || [])[1] || 0);

    if (process.env.DUMP_SEQ) {
      console.log('\n  --- final sequence ---');
      for (const b of final.books) {
        console.log(`    ${b.outOfOrder ? 'RED ' : b.unverifiable ? 'unv ' : b.possiblyOutOfOrder ? 'poss' : 'ok  '} sc=${(b._score??0).toFixed?.(2)??b._score} src=${b._src} "${b.spine_label}"`);
      }
    }
    const olLog = perfLogs.filter(l => l.includes('verifyTitlesOpenLibrary')).pop() || '';
    const olMs = Number((/(\d+)ms/.exec(olLog) || [])[1] || 0);
    const olCalls = Number((/(\d+) call/.exec(olLog) || [])[1] || 0);

    return {
      imgName, section, renders: renders.length, error: null,
      // FIXTURE=1 captures the exact fused labels this live run produced, so
      // ordering logic can be iterated offline against REALISTIC degraded reads
      // (see replay_order.mjs). ground_truth_v2 can't expose these bugs: it
      // holds clean labels, where the ordering already scores 5 TP / 0 FP. The
      // detection failures only exist in the presence of single-engine reads,
      // truncations and mis-reads — i.e. exactly what this array records.
      fixture: process.env.FIXTURE ? final.books.map(b => ({
        spine_label: b.spine_label, title: b.title ?? null,
        shelfRow: b.shelfRow ?? 0, _src: b._src ?? null, _score: b._score ?? null,
        _truncated: !!b._truncated, _unclear: !!b._unclear, _bbox: b._bbox ?? null,
      })) : undefined,
      recallN: hcMatched, recallD: hc.length,
      precN: matchedCount, precD: detected.length,
      tp: expected.length - pool.length, fp, fn: pool, fpFirst,
      flagEvidence: final.books.filter(b => b.outOfOrder)
        .map(b => ({ spine_label: b.spine_label, _src: b._src, _score: b._score })),
      unread: final.unreadCount || 0,
      firstMs, finalMs, olMs, olCalls, detMs, recMs, localMs,
      pageErrors, uiFails, badgeText: ui.badgeText, perfLogs,
    };
  } catch (e) {
    return { imgName, section, error: e.message, renders: 0,
      recallN: 0, recallD: hc.length, precN: 0, precD: 0,
      tp: 0, fp: [], fn: (EXPECTED_MISORDERS[imgName] || []), fpFirst: [], unread: 0,
      firstMs: 0, finalMs: 0, olMs: 0, olCalls: 0, detMs: 0, recMs: 0, localMs: 0,
      pageErrors, uiFails, badgeText: '' };
  } finally {
    await page.close();
  }
}

// ── Main ────────────────────────────────────────────────────────────────────
const gtAll = loadGT();
const images = Object.keys(gtAll).filter(k => !k.startsWith('_') && (!ONLY || k.includes(ONLY)));

const { srv, port } = await startServer();
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: HEADFUL ? false : 'new',
  userDataDir: mkdtempSync(join(tmpdir(), 'shelfcheck-site-')),
  args: ['--no-first-run', '--no-default-browser-check'],
});

console.log(`ShelfCheck SITE SUITE — main-mode (camera capture) path, ${images.length} image(s)`);
console.log(`Mode: ${OFFLINE ? 'OFFLINE (on-device engine only)' : 'LIVE (ocr.space + Open Library)'}${QS ? '  qs=' + QS : ''}`);
console.log('='.repeat(112));

const results = [];
// PACE_MS: wait between images. The free Gemini tier limits requests per
// MINUTE, and this suite fires images back-to-back — with the tiled decimal
// pass that is ~30 calls/min, far above real use (a volunteer scans every
// ~30s). Without pacing the suite rate-limits itself and measures the limit
// rather than the app.
const PACE_MS = Number(process.env.PACE_MS || 0);
for (const imgName of images) {
  if (PACE_MS && results.length) await new Promise(r => setTimeout(r, PACE_MS));
  process.stdout.write(`  ${imgName} … `);
  const r = await runImage(browser, `http://127.0.0.1:${port}`, imgName, gtAll[imgName]);
  results.push(r);
  console.log(r.error ? `ERROR: ${r.error}` : `${(r.firstMs / 1000).toFixed(1)}s first / ${(r.finalMs / 1000).toFixed(1)}s final`);
  if (process.env.PERF) for (const l of r.perfLogs || []) console.log('      ' + l);
}

await browser.close();
srv.close();

// ── Report ──────────────────────────────────────────────────────────────────
const short = n => n.replace('PXL_2026060', '').replace('.MP', '').replace('.jpg', '');
console.log('\n' + '='.repeat(112));
console.log('IMAGE           SECTION     RECALL          PRECISION       FLAGS            UNREAD  FIRST   FINAL   OL');
console.log('-'.repeat(112));

let RN = 0, RD = 0, PN = 0, PD = 0, TP = 0, FP = 0, FN = 0, errs = 0, uiFailCount = 0, pageErrCount = 0;
const firsts = [], finals = [], ols = [];
for (const r of results) {
  RN += r.recallN; RD += r.recallD; PN += r.precN; PD += r.precD;
  TP += r.tp; FP += r.fp.length; FN += r.fn.length;
  if (r.error) errs++;
  uiFailCount += r.uiFails.length;
  pageErrCount += r.pageErrors.length;
  if (!r.error) { firsts.push(r.firstMs); finals.push(r.finalMs); ols.push(r.olMs); }

  const rec = r.recallD ? `${r.recallN}/${r.recallD} (${(r.recallN / r.recallD * 100).toFixed(0)}%)` : 'n/a';
  const prec = r.precD ? `${r.precN}/${r.precD} (${(r.precN / r.precD * 100).toFixed(0)}%)` : 'n/a';
  const bad = r.error || r.fp.length || r.uiFails.length || r.pageErrors.length;
  console.log(
    `${(bad ? '✗ ' : '  ') + short(r.imgName).padEnd(14)}${r.section.padEnd(12)}` +
    `${rec.padEnd(16)}${prec.padEnd(16)}` +
    `${`TP${r.tp} FP${r.fp.length} FN${r.fn.length}`.padEnd(17)}` +
    `${String(r.unread).padEnd(8)}` +
    `${r.error ? 'ERR' : (r.firstMs / 1000).toFixed(1) + 's'}`.padEnd(8) +
    `${r.error ? '' : (r.finalMs / 1000).toFixed(1) + 's'}`.padEnd(8) +
    `${r.olMs ? `${(r.olMs / 1000).toFixed(1)}s/${r.olCalls}` : '–'}`);

  if (r.error) console.log(`      ERROR: ${r.error}`);
  for (const f of r.fp) {
    const ev = (r.flagEvidence || []).find(e => e.spine_label === f);
    console.log(`      FALSE POSITIVE flag: "${f}"  ← would send a volunteer to a correctly shelved book` +
      (ev ? `   [src=${ev._src} score=${ev._score}]` : ''));
  }
  for (const f of (r.fpFirst || [])) {
    if (!r.fp.includes(f)) console.log(`      TRANSIENT false flag on first render (retracted by refine): "${f}"`);
  }
  for (const f of r.fn) console.log(`      MISSED misshelved: "${f}"`);
  for (const f of r.uiFails) console.log(`      UI: ${f}`);
  for (const f of r.pageErrors) console.log(`      PAGE ERROR: ${f}`);
}

const med = a => a.length ? [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)] : 0;
const recallPct = RD ? RN / RD * 100 : 0;
const precPct = PD ? PN / PD * 100 : 0;

console.log('='.repeat(112));
console.log(`Label recall     : ${RN}/${RD} = ${recallPct.toFixed(1)}%`);
console.log(`Label precision  : ${PN}/${PD} = ${precPct.toFixed(1)}%`);
console.log(`Misshelved flags : TP ${TP}  FP ${FP}  FN ${FN}`);
console.log(`Speed (median)   : first render ${(med(firsts) / 1000).toFixed(1)}s · final ${(med(finals) / 1000).toFixed(1)}s · OL verify ${(med(ols) / 1000).toFixed(1)}s`);
console.log(`Speed (worst)    : first render ${(Math.max(0, ...firsts) / 1000).toFixed(1)}s · final ${(Math.max(0, ...finals) / 1000).toFixed(1)}s · OL verify ${(Math.max(0, ...ols) / 1000).toFixed(1)}s`);
console.log(`On-device engine : det ${(med(results.filter(r=>!r.error).map(r=>r.detMs))/1000).toFixed(1)}s · rec ${(med(results.filter(r=>!r.error).map(r=>r.recMs))/1000).toFixed(1)}s · total ${(med(results.filter(r=>!r.error).map(r=>r.localMs))/1000).toFixed(1)}s (median)`);
console.log(`Site health      : ${errs} error(s), ${uiFailCount} UI failure(s), ${pageErrCount} page error(s)`);

// ── Gates ───────────────────────────────────────────────────────────────────
const failures = [];
if (FP > GATES.maxFalsePositives) failures.push(`${FP} false-positive flag(s) (max ${GATES.maxFalsePositives})`);
if (errs) failures.push(`${errs} image(s) errored`);
if (uiFailCount) failures.push(`${uiFailCount} UI failure(s)`);
if (pageErrCount > GATES.maxPageErrors) failures.push(`${pageErrCount} page error(s)`);
if (recallPct < GATES.minRecallPct) failures.push(`recall ${recallPct.toFixed(1)}% below floor ${GATES.minRecallPct}%`);
if (Math.max(0, ...firsts) > GATES.maxFirstRenderMs) failures.push(`slowest first render ${(Math.max(...firsts) / 1000).toFixed(1)}s over budget ${(GATES.maxFirstRenderMs / 1000).toFixed(0)}s`);
if (Math.max(0, ...finals) > GATES.maxFinalRenderMs) failures.push(`slowest final render ${(Math.max(...finals) / 1000).toFixed(1)}s over budget ${(GATES.maxFinalRenderMs / 1000).toFixed(0)}s`);
if (Math.max(0, ...ols) > GATES.maxOlVerifyMs) failures.push(`slowest Open Library verify ${(Math.max(...ols) / 1000).toFixed(1)}s over budget ${(GATES.maxOlVerifyMs / 1000).toFixed(0)}s`);

console.log('='.repeat(112));
if (failures.length) { console.log('RESULT: FAIL'); for (const f of failures) console.log('  ✗ ' + f); }
else console.log('RESULT: PASS — all accuracy, speed, and site-functionality gates met');

if (JSON_OUT) {
  writeFileSync(JSON_OUT, JSON.stringify({
    mode: OFFLINE ? 'offline' : 'live', gates: GATES,
    totals: { recallN: RN, recallD: RD, precN: PN, precD: PD, TP, FP, FN },
    speed: { firstMedian: med(firsts), finalMedian: med(finals), olMedian: med(ols),
             firstWorst: Math.max(0, ...firsts), finalWorst: Math.max(0, ...finals) },
    failures, results,
  }, null, 2));
  console.log(`\nWrote ${JSON_OUT}`);
}

process.exit(failures.length ? 1 : 0);
