"""Build the "look closer" cache: azure_zoom/<img>.json

Azure reads a lone sticker far better than the same sticker inside a photo of a
packed shelf — measured, "635 BUC" comes back 655/535 on the full frame and
'635' at 0.99 once cropped out, with NO upscaling. So the second pass is about
isolating the sticker, not about resolution.

Every sticker that needs a closer look is cropped from the native photo and
pasted into ONE composite image with white gutters, so the whole pass costs a
single Azure call per photo. The cell rectangles are cached alongside the
response so the reader can map each word back to the sticker it came from.

Run dump_zoom_candidates.mjs first to produce _cands.json.
"""
import io, json, os, time, urllib.request
from PIL import Image

KEY = open(r'C:\Users\krish\shelfcheck-ocr\.azure_key.txt.txt').read().strip()
URL = ('https://shelfcheck-vision.cognitiveservices.azure.com/computervision/'
       'imageanalysis:analyze?api-version=2024-02-01&features=read')
OUT = os.environ.get('ZOUT', 'azure_zoom')
import os
ZOOM = int(os.environ.get('ZOOM', 2))
GUTTER = int(os.environ.get('GUT', 40))

os.makedirs(OUT, exist_ok=True)


def call(im, q=90):
    b = io.BytesIO(); im.save(b, 'JPEG', quality=q)
    r = urllib.request.Request(URL, data=b.getvalue(),
        headers={'Ocp-Apim-Subscription-Key': KEY,
                 'Content-Type': 'application/octet-stream'})
    for a in range(10):
        try:
            return json.load(urllib.request.urlopen(r))
        except urllib.error.HTTPError as e:
            if e.code != 429 or a == 9:
                raise
            time.sleep(8)


def compose(im, cands):
    """Grid-pack the crops. Returns (composite, [cell rect per candidate])."""
    crops = []
    for c in cands:
        x0, y0, x1, y1 = c['box']
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(im.width, x1), min(im.height, y1)
        cr = im.crop((x0, y0, x1, y1))
        crops.append(cr.resize((cr.width * ZOOM, cr.height * ZOOM), Image.LANCZOS))
    cols = max(1, int(len(crops) ** 0.5 + 0.999))
    cw = max(c.width for c in crops) + GUTTER
    ch = max(c.height for c in crops) + GUTTER
    rows = (len(crops) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cw + GUTTER, rows * ch + GUTTER), 'white')
    rects = []
    for i, cr in enumerate(crops):
        x = GUTTER + (i % cols) * cw
        y = GUTTER + (i // cols) * ch
        sheet.paste(cr, (x, y))
        rects.append([x, y, cr.width, cr.height])
    return sheet, rects


cands = json.load(open('_cands.json'))
for fn, rec in cands.items():
    dst = os.path.join(OUT, fn + '.json')
    if os.path.exists(dst):
        print('have', fn); continue
    if not rec['cands']:
        json.dump({'cands': [], 'rects': [], 'raw': None}, open(dst, 'w')); continue
    im = Image.open(fn)
    sheet, rects = compose(im, rec['cands'])
    raw = call(sheet)
    json.dump({'cands': rec['cands'], 'rects': rects, 'zoom': ZOOM,
               'sheet': sheet.size, 'raw': raw}, open(dst, 'w'))
    print('built', fn, len(rec['cands']), 'crops, sheet', sheet.size)
