"""Is the ground-truth call number physically PRESENT in Azure's raw words?

Separates "Azure cannot see it" from "our assembler failed to build it".
Words are joined only when they sit in the same spine column, the way a
human eye follows a label down a spine.
"""
import json, os, re, glob

CACHE = 'azure_cache'
gt = json.load(open('ground_truth.json'))


def words_for(path):
    d = json.load(open(path))
    out = []
    for blk in d['raw']['readResult']['blocks']:
        for ln in blk['lines']:
            for w in ln['words']:
                xs = [p['x'] for p in w['boundingPolygon']]
                ys = [p['y'] for p in w['boundingPolygon']]
                out.append({
                    't': w['text'], 'x': min(xs), 'y': min(ys),
                    'w': max(xs) - min(xs), 'h': max(ys) - min(ys),
                    'c': w['confidence'],
                })
    return out


def digits(s):
    return re.sub(r'\D', '', s)


def colov(a, b):
    """Horizontal overlap as a fraction of the narrower word."""
    px = min(a['x'] + a['w'], b['x'] + b['w']) - max(a['x'], b['x'])
    return max(0, px) / max(1, min(a['w'], b['w']))


def stacked(a, b):
    """b sits directly below a in the same column."""
    if colov(a, b) < 0.35:
        return False
    dy = b['y'] - (a['y'] + a['h'])
    return -a['h'] * 0.6 <= dy <= a['h'] * 3.0


def find_number(target, words):
    """'exact' | 'joined' | 'partial' | 'absent' for a digit string."""
    if not target:
        return 'absent'
    for w in words:
        if digits(w['t']) == target:
            return 'exact'
    # up to three vertically stacked fragments in one column
    for a in words:
        da = digits(a['t'])
        if not da or not target.startswith(da) or da == target:
            continue
        for b in words:
            if b is a or not stacked(a, b):
                continue
            db = da + digits(b['t'])
            if db == target:
                return 'joined'
            if not target.startswith(db):
                continue
            for c in words:
                if c is a or c is b or not stacked(b, c):
                    continue
                if db + digits(c['t']) == target:
                    return 'joined'
    for w in words:
        d = digits(w['t'])
        if len(d) >= 3 and target.startswith(d):
            return 'partial'
    return 'absent'


tot = {'exact': 0, 'joined': 0, 'partial': 0, 'absent': 0}
cut_ok = cut_tot = 0
fiction = 0
absent_examples = []
partial_examples = []

for img, rec in gt.items():
    p = os.path.join(CACHE, img + '.json')
    if not os.path.exists(p):
        continue
    words = words_for(p)
    upper = [w for w in words if re.fullmatch(r'[A-Za-z]{2,4}', w['t'])]
    for bk in rec['books']:
        lab = (bk.get('spine_label') or '').strip()
        if not lab:
            continue
        m_num = re.search(r'\d[\d.,]*', lab)
        if not m_num:
            fiction += 1          # surname-only spine, no call number to find
            continue
        num = digits(m_num.group(0))
        verdict = find_number(num, words)
        tot[verdict] += 1
        if verdict == 'absent' and len(absent_examples) < 12:
            absent_examples.append((img, lab))
        if verdict == 'partial' and len(partial_examples) < 12:
            partial_examples.append((img, lab))
        m = re.search(r'\b([A-Za-z]{3})\b', lab)
        if m:
            cut_tot += 1
            if any(w['t'].upper().startswith(m.group(1).upper()) for w in upper):
                cut_ok += 1

n = sum(tot.values())
print(f'Numbered labels checked      : {n}   (+{fiction} fiction surname-only spines)')
for k in ('exact', 'joined', 'partial', 'absent'):
    print(f'  {k:8} : {tot[k]:4}  ({tot[k]/n*100:5.1f}%)')
print(f'\nAzure CEILING (exact+joined) : {(tot["exact"]+tot["joined"])/n*100:.1f}%')
print(f'Cutter present               : {cut_ok}/{cut_tot} = {cut_ok/max(1,cut_tot)*100:.1f}%')
print('\nPARTIAL (Azure saw a prefix, rest not in image):')
for img, lab in partial_examples:
    print(f'  {img[-12:]}  {lab}')
print('\nABSENT (no trace at all):')
for img, lab in absent_examples:
    print(f'  {img[-12:]}  {lab}')
