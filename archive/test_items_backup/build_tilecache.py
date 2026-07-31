import io, json, urllib.request, os, time
from PIL import Image
KEY=open(r'C:\Users\krish\shelfcheck-ocr\.azure_key.txt.txt').read().strip()
URL='https://shelfcheck-vision.cognitiveservices.azure.com/computervision/imageanalysis:analyze?api-version=2024-02-01&features=read'
OUT='azure_tiled'; os.makedirs(OUT,exist_ok=True)
def call(im,q):
    b=io.BytesIO(); im.save(b,'JPEG',quality=q)
    r=urllib.request.Request(URL,data=b.getvalue(),
        headers={'Ocp-Apim-Subscription-Key':KEY,'Content-Type':'application/octet-stream'})
    for a in range(10):
        try: return json.load(urllib.request.urlopen(r))
        except urllib.error.HTTPError as e:
            if e.code!=429 or a==9: raise
            time.sleep(8)
GT=json.load(open('ground_truth_v2.json'))
for f,g in GT.items():
    if f.startswith('_') or not os.path.exists(f): continue
    dst=os.path.join(OUT,f+'.json')
    if os.path.exists(dst): print('have',f); continue
    im=Image.open(f); W,H=im.size
    # full frame exactly as encodeForOCR: maxDim 2400, q0.88
    s=min(1,2400/max(W,H)); sw,sh=round(W*s),round(H*s)
    full=call(im.resize((sw,sh),Image.LANCZOS),88)
    rec={'w':sw,'h':sh,'natW':W,'natH':H,'full':full,'tiles':[]}
    if g.get('section')=='nonfiction':
        ov=round(W*0.05); n=2
        for i in range(n):
            x0=W*i//n; x1=W*(i+1)//n
            lo=max(0,x0-(ov if i else 0)); hi=min(W,x1+(ov if i<n-1 else 0))
            rec['tiles'].append({'xoff':lo,'raw':call(im.crop((lo,0,hi,H)),85)})
    json.dump(rec,open(dst,'w'))
    print('built',f,'tiles',len(rec['tiles']))
