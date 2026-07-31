/**
 * Syntax-check the app's inline <script> without a browser.
 *
 * index.html is one 6,000-line inline script; a bad edit anywhere in it takes
 * the whole page down with a blank screen and no error the user would ever see.
 * This pulls the script out and hands it to the JS parser, so a broken edit is
 * caught in ~100ms instead of on a phone in a library.
 *
 * Also reports identifiers that are REFERENCED but never declared, which is how
 * a half-finished deletion shows up: the parse still succeeds, and the page dies
 * at runtime on the first call to a function that is no longer there.
 *
 *   node syntax_check.mjs
 */
import { readFileSync, writeFileSync, unlinkSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execFileSync } from 'child_process';

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = join(HERE, '..', '..', 'shelfcheck-ocr', 'index.html');
const html = readFileSync(APP, 'utf-8');

// The app script is the last, longest <script> block — the only one without src.
const blocks = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)];
if (!blocks.length) { console.error('no inline <script> found'); process.exit(1); }
const block = blocks.sort((a, b) => b[1].length - a[1].length)[0];
const code = block[1];
const lineOffset = html.slice(0, block.index).split('\n').length;

const tmp = join(HERE, '_syntax_check.tmp.mjs');
writeFileSync(tmp, code);
try {
  execFileSync(process.execPath, ['--check', tmp], { stdio: 'pipe' });
  console.log(`PARSE OK — ${code.split('\n').length} lines (index.html:${lineOffset}+)`);
} catch (e) {
  const msg = String(e.stderr || e.message);
  // Re-point the parser's line numbers at index.html so the error is clickable.
  console.error(msg.replace(/_syntax_check\.tmp\.mjs:(\d+)/g,
    (_, n) => `index.html:${+n + lineOffset - 1}`));
  unlinkSync(tmp);
  process.exit(1);
}
unlinkSync(tmp);

// ── Dangling references ────────────────────────────────────────────────────
// Cheap and deliberately approximate: collect every declared name, then flag
// call sites `foo(` whose name was never declared anywhere and isn't a known
// global. Catches the "deleted the function, kept the caller" failure mode.
const declared = new Set();
for (const re of [/\bfunction\s+([A-Za-z_$][\w$]*)/g,
                  /\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)/g,
                  /\bclass\s+([A-Za-z_$][\w$]*)/g]) {
  for (const m of code.matchAll(re)) declared.add(m[1]);
}
// Destructured and parameter names, coarsely.
for (const m of code.matchAll(/\b(?:const|let|var)\s*[{[]([^}\]]*)[}\]]/g)) {
  for (const p of m[1].split(',')) {
    const n = p.split(':').pop().split('=')[0].trim().replace(/^\.\.\./, '');
    if (/^[A-Za-z_$][\w$]*$/.test(n)) declared.add(n);
  }
}
for (const m of code.matchAll(/(?:function\s*[\w$]*|=>\s*)?\(([^()]*)\)\s*(?:=>|\{)/g)) {
  for (const p of m[1].split(',')) {
    const n = p.split('=')[0].trim().replace(/^\.\.\./, '');
    if (/^[A-Za-z_$][\w$]*$/.test(n)) declared.add(n);
  }
}

const GLOBALS = new Set(['if','for','while','switch','catch','return','typeof','await',
  'new','function','with','console','Math','JSON','Object','Array','String','Number',
  'Boolean','Promise','Set','Map','Date','Error','RegExp','parseInt','parseFloat',
  'isNaN','isFinite','encodeURIComponent','decodeURIComponent','setTimeout','fetch',
  'clearTimeout','setInterval','clearInterval','requestAnimationFrame','alert','atob','btoa',
  'cancelAnimationFrame','document','window','navigator','localStorage','location','structuredClone',
  'URL','URLSearchParams','Blob','FileReader','Image','ImageData','ImageCapture','Uint8Array',
  'Uint8ClampedArray','Float32Array','Int32Array','AbortController','FormData','Headers',
  'performance','crypto','queueMicrotask','OffscreenCanvas','createImageBitmap','Intl',
  'DOMParser','TextDecoder','TextEncoder','Symbol','Proxy','Reflect','BigInt','WeakMap','WeakSet']);

// Comments must go first, or English prose reads as calls: "OCR proxies during
// page load" scans as a call to `proxies(`. Strings too — CSS and HTML
// fragments are full of parentheses. Newlines are preserved so the reported
// line numbers still point at the right place.
const blank = s => s.replace(/[^\n]/g, ' ');
const stripped = code
  .replace(/\/\*[\s\S]*?\*\//g, blank)
  .replace(/(^|[^:])\/\/[^\n]*/g, (m, p) => p + blank(m.slice(p.length)))
  .replace(/`(?:\\.|\$\{[^}]*\}|[^`\\])*`/g, blank)
  .replace(/'(?:\\.|[^'\\\n])*'/g, blank)
  .replace(/"(?:\\.|[^"\\\n])*"/g, blank);

const missing = new Map();
for (const m of stripped.matchAll(/(^|[^.\w$])([A-Za-z_$][\w$]*)\s*\(/gm)) {
  const n = m[2];
  if (declared.has(n) || GLOBALS.has(n)) continue;
  if (!missing.has(n)) missing.set(n, code.slice(0, m.index).split('\n').length + lineOffset - 1);
}
if (missing.size) {
  console.log(`\n${missing.size} call(s) to undeclared name(s):`);
  for (const [n, ln] of [...missing].sort((a, b) => a[1] - b[1])) {
    console.log(`  index.html:${ln}  ${n}(...)`);
  }
} else {
  console.log('No dangling calls.');
}
