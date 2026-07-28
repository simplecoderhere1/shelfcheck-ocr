const { app } = require('@azure/functions');

// Key-holding proxy for Azure AI Vision Read.
//
// The browser posts the JPEG here and gets Azure's raw response back. The
// subscription key is read from application settings and never leaves the
// server, so the static page can stay public without exposing it. This is the
// whole reason the site is hosted on Static Web Apps rather than GitHub Pages:
// Pages can only serve files, and a key shipped to the browser is a key handed
// to anyone who opens devtools.
//
// Configure in the SWA portal (Settings -> Environment variables):
//   AZURE_VISION_ENDPOINT   https://<resource>.cognitiveservices.azure.com
//   AZURE_VISION_KEY        <the key>

const API_PATH = '/computervision/imageanalysis:analyze?api-version=2024-02-01&features=read';

app.http('ocr', {
  methods: ['POST'],
  authLevel: 'anonymous',
  route: 'ocr',
  handler: async (request, context) => {
    const endpoint = (process.env.AZURE_VISION_ENDPOINT || '').replace(/\/+$/, '');
    const key = process.env.AZURE_VISION_KEY || '';
    if (!endpoint || !key) {
      // Say which setting is missing, but never echo the value.
      context.error('missing AZURE_VISION_ENDPOINT or AZURE_VISION_KEY');
      return { status: 500, jsonBody: { error: 'vision credentials not configured' } };
    }

    const body = Buffer.from(await request.arrayBuffer());
    if (!body.length) return { status: 400, jsonBody: { error: 'empty body' } };

    // The phone is waiting on this call, and the app's own budget is 5s end to
    // end. Fail fast rather than leave a volunteer staring at a spinner: the
    // client falls back to its other engines when this errors.
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 20000);
    try {
      const res = await fetch(endpoint + API_PATH, {
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
        context.error(`vision ${res.status}`);
        return { status: res.status, jsonBody: { error: 'vision request failed', status: res.status } };
      }
      return {
        status: 200,
        headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
        body: text,
      };
    } catch (err) {
      context.error('vision call failed: ' + err.name);
      return { status: 504, jsonBody: { error: 'vision timeout' } };
    } finally {
      clearTimeout(timer);
    }
  },
});
