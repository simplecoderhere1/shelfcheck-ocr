// Key-holding proxy for Azure AI Vision Read, on Cloudflare Workers.
//
// The page is served from GitHub Pages, which can only serve static files — a
// subscription key shipped to the browser is a key handed to anyone who opens
// devtools. The browser posts its JPEG here instead, and this Worker adds the
// key server-side.
//
// Unlike the same-origin Azure Functions version, Pages and workers.dev are
// DIFFERENT ORIGINS, so this must answer the CORS preflight and echo an allowed
// origin on every response — including errors, since a response without the
// header is unreadable by the page even when the status is 200.
//
// Configure (Worker -> Settings -> Variables and Secrets):
//   AZURE_VISION_ENDPOINT   https://<resource>.cognitiveservices.azure.com  (plaintext)
//   AZURE_VISION_KEY        <the key>                                       (SECRET)

const VISION_PATH =
  '/computervision/imageanalysis:analyze?api-version=2024-02-01&features=read';

const ALLOWED_ORIGINS = new Set([
  'https://simplecoderhere1.github.io',
  'http://localhost:8080',
  'http://127.0.0.1:8080',
]);

function corsHeaders(origin) {
  const allow = ALLOWED_ORIGINS.has(origin) ? origin : 'https://simplecoderhere1.github.io';
  return {
    'Access-Control-Allow-Origin': allow,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
  };
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';
    const cors = corsHeaders(origin);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: cors });
    }
    if (request.method !== 'POST') {
      return json({ error: 'POST only' }, 405, cors);
    }

    const url = new URL(request.url);
    if (!url.pathname.endsWith('/ocr')) {
      return json({ error: 'not found' }, 404, cors);
    }

    const endpoint = (env.AZURE_VISION_ENDPOINT || '').replace(/\/+$/, '');
    const key = env.AZURE_VISION_KEY || '';
    if (!endpoint || !key) {
      // Name the missing setting, never echo its value.
      return json({ error: 'vision credentials not configured' }, 500, cors);
    }

    const body = await request.arrayBuffer();
    if (!body.byteLength) return json({ error: 'empty body' }, 400, cors);

    // A volunteer is holding a phone waiting on this, and the app's whole budget
    // is 5s. Fail fast rather than hang; the client falls back to its other
    // engines when this errors.
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 20000);
    try {
      const res = await fetch(endpoint + VISION_PATH, {
        method: 'POST',
        headers: {
          'Ocp-Apim-Subscription-Key': key,
          'Content-Type': 'application/octet-stream',
        },
        body,
        signal: ctrl.signal,
      });
      const text = await res.text();
      if (!res.ok) {
        return json({ error: 'vision request failed', status: res.status }, res.status, cors);
      }
      return new Response(text, {
        status: 200,
        headers: { ...cors, 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
      });
    } catch (err) {
      return json({ error: 'vision timeout' }, 504, cors);
    } finally {
      clearTimeout(timer);
    }
  },
};

function json(obj, status, cors) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { ...cors, 'Content-Type': 'application/json' },
  });
}
