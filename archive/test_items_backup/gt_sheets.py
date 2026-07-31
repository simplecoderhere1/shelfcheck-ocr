"""Render the two artifacts gt_plan.mjs describes.

  python gt_sheets.py            # all images
  python gt_sheets.py 195911082  # one image

For each photo:

  _gt/<img>_crops.jpg      Numbered tile per emitted label, cropped from the
                           native photo with the sticker ringed in red and
                           enough margin to show its neighbours. Answers "is
                           this what the sticker actually says?".

  _gt/<img>_cover_<k>.jpg  The photo in horizontal strips at a readable scale,
                           every emitted label boxed and numbered. Answers
                           "which stickers have no box on them at all?" — the
                           question a bootstrap from the app's own reads cannot
                           answer any other way.

Coordinates in the plan are in the space of the image the app UPLOADED, which
encodeForOCR caps at maxDim 2400. Scale back to native is computed per image
from max(w, h), never from width: the portrait photo 195849222 is 3072x4080, so
width/2400 is wrong for it and produces crops of the wrong part of the shelf.
"""
import json, os, sys
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '_gt')
SENT_MAX_DIM = 2400

# Crop sheet geometry
PAD_X, PAD_ABOVE, PAD_BELOW = 2.6, 2.0, 3.0    # in sticker-heights
COLS, TILE_W, TILE_H, CAPTION_H = 6, 300, 250, 26

# Coverage strip geometry.
#
# A whole 4080px-wide shelf scaled to fit one image puts each sticker at ~20px —
# unreadable, which defeats the point. So strips are cut per ROW (the stickers
# sit in a narrow band near the bottom of the spines; the top two thirds of the
# frame is bare book) and then split horizontally into segments narrow enough
# that a sticker survives at output scale.
SEG_NATIVE_W = 1350     # native px per segment
SEG_OUT_W = 1650        # output px per segment -> ~1.2x upscale
SEG_OVERLAP = 0.10      # so a sticker on a seam is whole in one of them
BAND_PAD = 3.0          # sticker-heights of margin above/below the label band

only = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else ''
plan = json.load(open(os.path.join(OUT, '_plan.json')))


def crop_sheet(im, labels, scale, dst):
    tiles = []
    for e in labels:
        x, y, w, h = e['bbox']
        x0, y0 = int((x - h * PAD_X) * scale), int((y - h * PAD_ABOVE) * scale)
        x1, y1 = int((x + w + h * PAD_X) * scale), int((y + h * (1 + PAD_BELOW)) * scale)
        box = (int(x * scale) - x0, int(y * scale) - y0,
               int((x + w) * scale) - x0, int((y + h) * scale) - y0)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(im.width, x1), min(im.height, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            tiles.append((None, e)); continue
        crop = im.crop((x0, y0, x1, y1))
        ImageDraw.Draw(crop).rectangle(box, outline=(255, 40, 40), width=3)
        crop.thumbnail((TILE_W, TILE_H - CAPTION_H), Image.LANCZOS)
        tiles.append((crop, e))

    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new('RGB', (COLS * TILE_W, rows * TILE_H), 'white')
    dr = ImageDraw.Draw(sheet)
    for i, (crop, e) in enumerate(tiles):
        cx, cy = (i % COLS) * TILE_W, (i // COLS) * TILE_H
        if crop:
            sheet.paste(crop, (cx + (TILE_W - crop.width) // 2, cy + CAPTION_H))
        dr.text((cx + 6, cy + 6), f"{e['n']}. {e['text']}", fill=(0, 0, 0))
        dr.rectangle([cx, cy, cx + TILE_W - 1, cy + TILE_H - 1], outline=(200, 200, 200))
    sheet.save(dst, quality=90)
    return len(tiles)


def coverage_strips(im, labels, scale, base):
    """One readable segment per chunk of each shelf row, labels boxed."""
    marked = im.copy()
    dr = ImageDraw.Draw(marked)
    for e in labels:
        x, y, w, h = e['bbox']
        r = [int(x * scale), int(y * scale), int((x + w) * scale), int((y + h) * scale)]
        dr.rectangle(r, outline=(0, 230, 60), width=4)
        dr.text((r[0] + 2, max(0, r[1] - 30)), str(e['n']), fill=(0, 230, 60))

    made = []
    for row in sorted({e['row'] for e in labels}):
        rl = [e for e in labels if e['row'] == row]
        hs = [e['bbox'][3] for e in rl]
        pad = (sum(hs) / len(hs)) * BAND_PAD * scale
        y0 = int(min(e['bbox'][1] for e in rl) * scale - pad)
        y1 = int(max(e['bbox'][1] + e['bbox'][3] for e in rl) * scale + pad)
        y0, y1 = max(0, y0), min(im.height, y1)

        step = int(SEG_NATIVE_W * (1 - SEG_OVERLAP))
        x = 0
        seg = 0
        while x < im.width:
            x1 = min(im.width, x + SEG_NATIVE_W)
            s = marked.crop((x, y0, x1, y1))
            s = s.resize((SEG_OUT_W, max(1, int(s.height * SEG_OUT_W / s.width))), Image.LANCZOS)
            seg += 1
            dst = f'{base}_row{row}_seg{seg}.jpg'
            s.save(dst, quality=90)
            made.append(dst)
            if x1 >= im.width:
                break
            x += step
    return made


for img_name, rec in plan.items():
    if only and only not in img_name:
        continue
    labels = rec['labels']
    im = Image.open(os.path.join(HERE, img_name))
    scale = max(im.width, im.height) / SENT_MAX_DIM
    base = os.path.join(OUT, img_name.replace('.jpg', ''))
    n = crop_sheet(im, labels, scale, base + '_crops.jpg')
    strips = coverage_strips(im, labels, scale, base)
    print(f'{img_name}: {n} crop(s), {len(strips)} coverage strip(s)  (scale {scale:.3f})')
