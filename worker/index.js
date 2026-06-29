/**
 * ShelfCheck OCR Proxy — Cloudflare Worker
 *
 * Accepts POST with a raw JPEG body, forwards it to ocr.space Engine 2
 * with the API key injected from a Worker secret, returns the JSON response.
 *
 * Set the secret before deploying:
 *   wrangler secret put OCRSPACE_API_KEY
 */

const OCRSPACE_URL = 'https://api.ocr.space/parse/image';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin':  '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age':       '86400',
};

export default {
  async fetch(request, env) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405, headers: CORS_HEADERS });
    }

    const imageBytes = await request.arrayBuffer();
    if (!imageBytes.byteLength) {
      return new Response('Empty body', { status: 400, headers: CORS_HEADERS });
    }

    const form = new FormData();
    form.append('file', new Blob([imageBytes], { type: 'image/jpeg' }), 'shelf.jpg');
    form.append('apikey',            env.OCRSPACE_API_KEY);
    form.append('language',          'eng');
    form.append('OCREngine',         '2');
    form.append('isOverlayRequired', 'true');
    form.append('detectOrientation', 'false');
    form.append('scale',             'true');

    let resp;
    try {
      resp = await fetch(OCRSPACE_URL, { method: 'POST', body: form });
    } catch (e) {
      return new Response(`Upstream fetch failed: ${e.message}`, {
        status: 502, headers: CORS_HEADERS,
      });
    }

    if (!resp.ok) {
      const body = await resp.text().catch(() => '');
      return new Response(`ocr.space ${resp.status}: ${body.slice(0, 300)}`, {
        status: 502, headers: CORS_HEADERS,
      });
    }

    const json = await resp.text();
    return new Response(json, {
      status: 200,
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
    });
  },
};
