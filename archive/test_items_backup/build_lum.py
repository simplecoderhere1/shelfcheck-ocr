"""Luminance rasters at the SHIPPING full-frame scale (encodeForOCR: maxDim 2400).
Needed by the truncation-geometry bench, which samples pixels beside a number.
Regenerate if encodeForOCR's maxDim changes."""
import json, os
from PIL import Image
GT=json.load(open('ground_truth_v2.json'))
os.makedirs('_lum',exist_ok=True)
for f,g in GT.items():
    if f.startswith('_') or g.get('section')!='nonfiction' or not os.path.exists(f): continue
    im=Image.open(f); W,H=im.size
    s=min(1,2400/max(W,H)); sw,sh=round(W*s),round(H*s)
    im.resize((sw,sh),Image.LANCZOS).convert('L').save(f'_lum/{f}.pgm')
    print(f,sw,sh)
