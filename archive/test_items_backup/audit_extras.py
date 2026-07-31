"""Build the contact sheets for audit_extras.mjs.

Each unaccounted-for label becomes one numbered tile: the sticker cropped out of
the NATIVE photo with enough margin to show its neighbours, captioned with what
the app read. Looking at the tile answers the only question that matters — is
there really a sticker there saying that, or did the app invent a book?

    node audit_extras.mjs        # -> _audit/_plan.json
    python audit_extras.py       # -> _audit/<img>.jpg

The bbox in the plan is in the coordinate space of the image the app UPLOADED,
which encodeForOCR caps at maxDim 2400. Every test photo is 4080 on its long
edge, so the scale back to native is 4080/2400 = 1.70 — but it is computed per
image rather than assumed, because assuming it is exactly the bug that produced
two useless crops last time (the portrait photo 195849222 is 3072x4080, so
width/2400 is wrong for it).
"""
import json, os, sys
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '_audit')
PLAN = '_plan.json'
SUFFIX = ''
for a in sys.argv[1:]:
    if a.startswith('--plan='):
        PLAN = a.split('=', 1)[1]
        SUFFIX = '_' + os.path.splitext(PLAN)[0].strip('_')
SENT_MAX_DIM = 2400          # encodeForOCR's cap
PAD_X, PAD_ABOVE, PAD_BELOW = 2.6, 2.2, 3.2   # in sticker-heights
COLS = 6
TILE_W, TILE_H = 300, 260
CAPTION_H = 34

plan = json.load(open(os.path.join(OUT, PLAN)))

for img_name, rec in plan.items():
    extras = rec['extras']
    if not extras:
        print(f'{img_name}: nothing to audit')
        continue
    im = Image.open(os.path.join(HERE, img_name))
    # maxDim, NOT width — the portrait photo would scale wrong otherwise.
    scale = max(im.width, im.height) / SENT_MAX_DIM

    tiles = []
    for e in extras:
        x, y, w, h = e['bbox']
        x0 = int((x - h * PAD_X) * scale)
        y0 = int((y - h * PAD_ABOVE) * scale)
        x1 = int((x + w + h * PAD_X) * scale)
        y1 = int((y + h * (1 + PAD_BELOW)) * scale)
        # Where the sticker itself sits inside the crop, so the tile can mark it.
        box = (int(x * scale) - x0, int(y * scale) - y0,
               int((x + w) * scale) - x0, int((y + h) * scale) - y0)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(im.width, x1), min(im.height, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            tiles.append((None, e))
            continue
        crop = im.crop((x0, y0, x1, y1))
        d = ImageDraw.Draw(crop)
        d.rectangle(box, outline=(255, 40, 40), width=3)
        crop.thumbnail((TILE_W, TILE_H - CAPTION_H), Image.LANCZOS)
        tiles.append((crop, e))

    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new('RGB', (COLS * TILE_W, rows * TILE_H), 'white')
    dr = ImageDraw.Draw(sheet)
    for i, (crop, e) in enumerate(tiles):
        cx, cy = (i % COLS) * TILE_W, (i // COLS) * TILE_H
        if crop:
            sheet.paste(crop, (cx + (TILE_W - crop.width) // 2, cy + CAPTION_H))
        cap = f"{i + 1}. {e['text']}"
        if e.get('score') is not None:
            cap += f"  ({e['score']:.2f})"
        dr.text((cx + 6, cy + 6), cap, fill=(0, 0, 0))
        dr.rectangle([cx, cy, cx + TILE_W - 1, cy + TILE_H - 1], outline=(200, 200, 200))

    dst = os.path.join(OUT, img_name.replace('.jpg', '') + SUFFIX + '.jpg')
    sheet.save(dst, quality=88)
    print(f'{img_name}: {len(tiles)} crop(s) -> {dst}  ({sheet.width}x{sheet.height})')
