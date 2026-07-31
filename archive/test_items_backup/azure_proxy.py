"""
Local stand-in for the production Azure Vision proxy — TEST ONLY.

The browser must never hold the subscription key, so calls go through a proxy
that injects it. In production that proxy is a Cloudflare Worker or an Azure
Static Web Apps managed Function with the key as a secret/app setting. This is
the same contract, running on localhost, so the real page can be driven
end-to-end without deploying anything.

The key is read from the gitignored file at startup and never logged.

Run:  python azure_proxy.py [--port=8788]
Then: site_suite.mjs --qs=azure=http://127.0.0.1:8788
"""
import json, os, sys, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ENDPOINT = 'https://shelfcheck-vision.cognitiveservices.azure.com'
API = '/computervision/imageanalysis:analyze?api-version=2024-02-01&features=read'
PORT = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--port=')), '8788'))

KEY = ''
for p in [r'C:\Users\krish\shelfcheck-ocr\.azure_key.txt',
          r'C:\Users\krish\shelfcheck-ocr\.azure_key.txt.txt']:
    if os.path.exists(p):
        KEY = open(p, encoding='utf-8-sig').read().strip()
        break
if not KEY:
    raise SystemExit('no key file found')


class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        self.send_response(200); self._cors()
        self.send_header('Content-Type', 'application/json'); self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        req = urllib.request.Request(ENDPOINT + API, data=body, method='POST', headers={
            'Ocp-Apim-Subscription-Key': KEY, 'Content-Type': 'application/octet-stream'})
        t0 = time.time()
        try:
            out = urllib.request.urlopen(req, timeout=120).read()
            code = 200
        except urllib.error.HTTPError as e:
            out = e.read(); code = e.code
        except Exception as e:
            out = json.dumps({'error': str(e)}).encode(); code = 502
        print(f'  /ocr {len(body)//1024}KB -> {code} in {time.time()-t0:.2f}s', flush=True)
        self.send_response(code); self._cors()
        self.send_header('Content-Type', 'application/json'); self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


print(f'azure proxy on 127.0.0.1:{PORT} -> {ENDPOINT}', flush=True)
ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
