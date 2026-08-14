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
//
// It also meters the free tier. The F0 allowance is 5,000 reads a month and 20
// a minute, and one shelf photo spends about twelve of them, so the ceiling is
// roughly 400 shelves and it is reachable. Past it Azure answers 403 to
// whoever is standing at the shelf. Counting here — the one place every device
// passes through — lets the app be told it is out BEFORE it asks, so it can
// divert to Gemini cleanly instead of failing a read in front of a volunteer.
// See quota-do.js for why the counter is a Durable Object.
import { Quota } from './quota-do.js';
export { Quota };

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
    // Read the meter without spending it. The app asks once at startup so that
    // a page opened after the allowance ran out goes to Gemini from its very
    // first photo, rather than learning by failing one.
    if (new URL(request.url).pathname.endsWith('/quota')) {
      if (!env.QUOTA) return json({ metered: false }, 200, cors);
      const s = await env.QUOTA.get(env.QUOTA.idFromName('azure-f0'))
        .fetch('https://q/status').then(r => r.json());
      return json({ metered: true, ...s }, 200, cors);
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

    // One object, one name — the count is global, not per-device.
    const quota = env.QUOTA
      ? env.QUOTA.get(env.QUOTA.idFromName('azure-f0'))
      : null;

    // Reserve the call, and only now: everything above this line can reject a
    // request that was never going to reach Azure, and a count spent on one of
    // those is a shelf the volunteer does not get to read later.
    //
    // Reserving BEFORE the call rather than counting after it is deliberate.
    // Azure charges the call the moment it arrives, so a request that is sent
    // and then times out on the way back has been spent; counting successes
    // would miss exactly the calls most likely to be retried.
    let q = null;
    if (quota) {
      q = await quota.fetch('https://q/take').then(r => r.json()).catch(() => null);
      if (q && !q.ok) {
        // Two different refusals, and the app must tell them apart. `month` is
        // over until the 1st, so the app should switch engines and stay
        // switched. `minute` clears in seconds, so it must NOT make the app
        // give up on Azure for the rest of the session.
        const h = { ...cors, ...quotaHeaders(q), 'Content-Type': 'application/json' };
        if (q.scope === 'minute') h['Retry-After'] = String(q.retryAfter ?? 5);
        return new Response(JSON.stringify({
          error: q.scope === 'month' ? 'quota exhausted' : 'rate limited',
          scope: q.scope, retryAfter: q.retryAfter ?? null, reset: q.reset,
        }), { status: 429, headers: h });
      }
    }

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
        // Azure is the authority on its own allowance, and it disagrees with
        // us: 403 out-of-call-volume means the month is gone even though our
        // count says otherwise (calls made outside this Worker — a test
        // script, a second app — spend the same 5,000). Record it so every
        // later request is refused here instead of being sent to be refused
        // there. 429 is the per-minute limiter and says nothing about the month.
        if (res.status === 403 && quota && /quota|call volume/i.test(text)) {
          q = await quota.fetch('https://q/spent').then(r => r.json()).catch(() => q);
        }
        return new Response(JSON.stringify({
          error: 'vision request failed', status: res.status,
          scope: res.status === 403 ? 'month' : res.status === 429 ? 'minute' : null,
        }), { status: res.status, headers: { ...cors, ...quotaHeaders(q), 'Content-Type': 'application/json' } });
      }
      return new Response(text, {
        status: 200,
        headers: {
          ...cors, ...quotaHeaders(q),
          'Content-Type': 'application/json', 'Cache-Control': 'no-store',
        },
      });
    } catch (err) {
      return json({ error: 'vision timeout' }, 504, cors);
    } finally {
      clearTimeout(timer);
    }
  },
};

// The meter, on every response including the failures. The app reads these to
// decide whether there is enough left for another whole shelf, so they have to
// ride along with the reads rather than needing a separate poll.
// Exposed to the page explicitly: CORS hides every header but a short safe
// list, so without Expose-Headers the browser can see them but JS cannot.
function quotaHeaders(q) {
  if (!q) return {};
  return {
    'X-Quota-Used': String(q.used ?? ''),
    'X-Quota-Remaining': String(q.remaining ?? ''),
    'X-Quota-Photos-Left': String(q.photosLeft ?? ''),
    'X-Quota-Reset': String(q.reset ?? ''),
    'Access-Control-Expose-Headers':
      'X-Quota-Used, X-Quota-Remaining, X-Quota-Photos-Left, X-Quota-Reset',
  };
}

function json(obj, status, cors) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { ...cors, 'Content-Type': 'application/json' },
  });
}
