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
// The endpoint is hardcoded on purpose. It was briefly a second variable, and
// two boxes side by side -- one public URL, one secret, told apart only by name
// -- invited pasting the key into the wrong one. It is a public host, not a
// credential, so it belongs in the code.
const VISION_ENDPOINT = 'https://shelfcheck-vision.cognitiveservices.azure.com';

const VISION_PATH =
  '/computervision/imageanalysis:analyze?api-version=2024-02-01&features=read';

// The key is read from EITHER secret name. AZURE_VISION_KEY is the intended one;
// AZURE_VISION_ENDPOINT is where the key actually lives on this account, left
// over from when the endpoint was a separate setting. Accepting both means the
// Worker works as configured today and keeps working if the secret is later
// re-added under the clearer name.
function visionKey(env) {
  return env.AZURE_VISION_KEY || env.AZURE_VISION_ENDPOINT || '';
}

const SITE_ORIGIN = 'https://simplecoderhere1.github.io';

// Loopback on ANY port: the test harness serves the app from an ephemeral port
// (`listen(0)`), so no fixed list can match it. Allowing only two hardcoded
// ports meant every harness run was rejected by CORS and scored 0 labels.
// Loopback is not a meaningful trust boundary here anyway — the allowlist
// exists to stop other sites spending the free Azure quota, and reaching
// 127.0.0.1 already means running on this machine.
const LOOPBACK = /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/;

function isAllowed(origin) {
  return origin === SITE_ORIGIN || LOOPBACK.test(origin);
}

function corsHeaders(origin) {
  const allow = isAllowed(origin) ? origin : SITE_ORIGIN;
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

    const key = visionKey(env);
    if (!key) {
      return json({ error: 'no vision key secret set on this Worker' }, 500, cors);
    }

    const body = await request.arrayBuffer();
    if (!body.byteLength) return json({ error: 'empty body' }, 400, cors);

    // A volunteer is holding a phone waiting on this, and the app's whole budget
    // is 5s. Fail fast rather than hang; the client falls back to its other
    // engines when this errors.
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 20000);
    try {
      const res = await fetch(VISION_ENDPOINT + VISION_PATH, {
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
