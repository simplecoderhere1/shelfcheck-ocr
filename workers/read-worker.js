// One upload, both readers. The phone posts its shelf JPEG here ONCE, as raw
// bytes; this Worker hands the same bytes to the Gemini proxy and the Azure OCR
// worker in parallel over service bindings (no extra network hop) and returns
// both answers together.
//
// Why it exists: measured on a throttled phone profile (4x CPU, 6 Mbps up), the
// page uploading the photo twice -- ~1.4 MB of base64 JSON to Gemini plus ~1 MB
// to Azure over the same uplink -- took 5.7-12.6s end to end, against 2.6-4.3s
// on a desktop line. The uplink was the bottleneck, so the fix is to cross it
// once, with binary instead of base64.
//
// The Gemini retry/hedge policy lives here too (it used to live in the page):
// a hedge request after HEDGE_MS no longer costs the phone a second upload.
// The prompt text is authored in fuse.js and sent in a header, so there is one
// source of truth for it.
//
// Keys stay where they were: the Gemini key in the selfcheck-proxy Worker, the
// Azure key (and its monthly quota counter) in shelfcheck-azure-ocr.

const SITE_ORIGIN = 'https://simplecoderhere1.github.io';
const LOOPBACK = /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/;
const MODELS = ['gemini-3.1-flash-lite', 'gemini-2.5-flash'];
const HEDGE_MS = 0;   // race from the start (see gemini() below)
const GEMINI_TIMEOUT_MS = 20000;
const AZURE_TIMEOUT_MS = 9000;
const MAX_OUTPUT_TOKENS = 1500;
const MAX_BYTES = 6 * 1024 * 1024;

function cors(origin) {
  const ok = origin === SITE_ORIGIN || LOOPBACK.test(origin);
  return {
    'Access-Control-Allow-Origin': ok ? origin : SITE_ORIGIN,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, X-Prompt',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
  };
}
const json = (obj, status, h) => new Response(JSON.stringify(obj), { status, headers: { ...h, 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } });

function b64(bytes) {
  // nodejs_compat provides Buffer; its base64 is native (a JS loop over 1 MB
  // would eat the free plan's CPU budget)
  return Buffer.from(bytes).toString('base64');
}

async function withTimeout(ms, fn) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ms);
  try { return await fn(ctrl.signal); } finally { clearTimeout(t); }
}

async function azure(env, bytes) {
  const t0 = Date.now();
  try {
    return await withTimeout(AZURE_TIMEOUT_MS, async signal => {
      const r = await env.AZURE.fetch('https://azure/ocr', {
        method: 'POST', body: bytes, signal,
        headers: { 'Content-Type': 'application/octet-stream', Origin: SITE_ORIGIN },
      });
      if (!r.ok) return { error: `azure ${r.status}`, ms: Date.now() - t0 };
      const j = await r.json();
      return { result: j, ms: Date.now() - t0 };
    });
  } catch (e) { return { error: String(e && e.name || e), ms: Date.now() - t0 }; }
}

// Body built by concatenation around the base64 so the 1.4 MB string is not
// re-serialised for every attempt.
function geminiBody(model, temperature, prompt, data) {
  return `{"model":${JSON.stringify(model)},"generationConfig":{"temperature":${temperature},"maxOutputTokens":${MAX_OUTPUT_TOKENS},"thinkingConfig":{"thinkingBudget":0}},` +
    `"contents":[{"parts":[{"text":${JSON.stringify(prompt)}},{"inline_data":{"mime_type":"image/jpeg","data":"${data}"}}]}]}`;
}

class GemErr extends Error { constructor(kind, msg) { super(msg); this.kind = kind; } }

async function geminiOnce(env, prompt, data, model, temperature, signal) {
  const ctrl = new AbortController();
  const onAbort = () => ctrl.abort();
  signal.addEventListener('abort', onAbort);
  const t = setTimeout(() => ctrl.abort(), GEMINI_TIMEOUT_MS);
  try {
    const r = await env.GEMINI.fetch('https://gemini/', {
      method: 'POST', signal: ctrl.signal,
      headers: { 'Content-Type': 'application/json', Origin: SITE_ORIGIN },
      body: geminiBody(model, temperature, prompt, data),
    });
    if (!r.ok) throw new GemErr(r.status === 429 ? 'QUOTA' : 'HTTP', `gemini ${r.status}`);
    const j = await r.json();
    const c = j?.candidates?.[0];
    const text = (c?.content?.parts || []).map(p => p.text || '').join('');
    const finish = c?.finishReason || '';
    if (!text.trim()) throw new GemErr('EMPTY', `no text (${finish})`);
    return { text, finish, model };
  } catch (e) {
    if (e instanceof GemErr) throw e;
    throw new GemErr(signal.aborted ? 'ABORTED' : 'TIMEOUT', String(e && e.name || e));
  } finally { clearTimeout(t); signal.removeEventListener('abort', onAbort); }
}

// retries: a 429 switches model (each has its own free-tier quota); an empty
// answer (RECITATION) or a runaway loop that hit the output cap retries warmer
async function geminiChain(env, prompt, data, signal, log, t0 = 0.2) {
  let model = MODELS[0], temperature = t0, last, capped = null;
  for (let i = 0; i < 3 && !signal.aborted; i++) {
    const t0 = Date.now();
    try {
      const r = await geminiOnce(env, prompt, data, model, temperature, signal);
      log.push(`${model}@${temperature}:${r.finish}:${Date.now() - t0}`);
      if (r.finish === 'MAX_TOKENS' && i < 2) { capped = r; temperature = 0.5; continue; }
      return r;
    } catch (e) {
      log.push(`${model}@${temperature}:${e.kind}:${Date.now() - t0}`);
      last = e;
      if (e.kind === 'ABORTED') break;
      if (e.kind === 'QUOTA') model = MODELS[(MODELS.indexOf(model) + 1) % MODELS.length];
      if (e.kind === 'EMPTY') temperature = 0.5;
      await new Promise(r => setTimeout(r, 300 * (i + 1)));
    }
  }
  if (capped) return capped;
  throw last || new GemErr('ABORTED', 'aborted');
}

// Two chains race from the start and the first answer wins. Measured on the
// phone profile, Gemini's latency is bimodal and random per request (the same
// photo was slow in both of two runs only 2 times in 36): ~65% of calls answer
// in ~2.2s, the rest in 4-8s. A hedge launched late still needs ~2.3s after it
// starts, so it barely moves the mean; two requests from t=0 are both slow only
// ~12% of the time. Costs two calls per photo (free tier: 15/min per model).
function gemini(env, prompt, data, log) {
  return new Promise((resolve, reject) => {
    const a = new AbortController(), b = new AbortController();
    let settled = false, running = 1, hedged = false, lastErr;
    const ok = (r, loser, tag) => { if (settled) return; settled = true; loser.abort(); resolve({ ...r, via: tag }); };
    const bad = e => { lastErr = e; if (--running === 0 && !settled) { settled = true; reject(lastErr); } };
    const hedge = () => {
      if (settled || hedged) return;
      hedged = true; running++;
      log.push('hedge');
      geminiChain(env, prompt, data, b.signal, log, 0.3).then(r => ok(r, a, 'hedge'), bad);
    };
    geminiChain(env, prompt, data, a.signal, log).then(r => ok(r, b, 'first'), e => { hedge(); bad(e); });
    setTimeout(hedge, HEDGE_MS);
  });
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';
    const h = cors(origin);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: h });
    const url = new URL(request.url);
    if (request.method !== 'POST' || url.pathname !== '/read') return json({ error: 'POST /read' }, 404, h);
    let prompt;
    try { prompt = decodeURIComponent(request.headers.get('X-Prompt') || ''); } catch { prompt = ''; }
    if (!prompt) return json({ error: 'missing X-Prompt' }, 400, h);
    const bytes = new Uint8Array(await request.arrayBuffer());
    if (!bytes.byteLength || bytes.byteLength > MAX_BYTES) return json({ error: 'bad image size' }, 400, h);
    const t0 = Date.now();
    const data = b64(bytes);
    const log = [];
    const azP = url.searchParams.get('azure') === '0' ? Promise.resolve({ error: 'skipped' }) : azure(env, bytes);
    let gem;
    try { gem = await gemini(env, prompt, data, log); }
    catch (e) {
      const az = await azP;
      return json({ error: 'gemini', kind: e.kind || 'ERROR', message: String(e.message || e), azure: az.result || null, log }, 502, h);
    }
    const tg = Date.now() - t0;
    const az = await azP;
    return json({ gemini: gem, azure: az.result || null, azureError: az.error || null, ms: { gemini: tg, azure: az.ms, total: Date.now() - t0 }, log }, 200, h);
  },
};
