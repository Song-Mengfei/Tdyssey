"""Compare immutable predictions at300/1000/3000. No model or simulator calls."""
from pathlib import Path
import collections
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
indices = read(BASE / "r2c0/evaluation/indices.json")["cohorts"]
manifest_path = BASE / "r2c0/data/manifest.json"
manifest = read(manifest_path)
entries = {e['trajectory_id']: e for e in manifest['entries']}
raw, metrics, hashes = {}, {}, {}
for update in (300, 1000, 3000):
    root = (BASE / "r2c1r1/run/evaluation/final" if update == 300 else
            BASE / f"r2c3/run/evaluation/update{update}/steps3")
    for path in sorted((root/'steps').glob('*.npz')):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        cohort = path.name.split('-batch')[0]
        horizon = int(path.stem.rsplit('step', 1)[1])+1
        with np.load(path, allow_pickle=False) as z:
            for wid, image in zip(z['window_ids'].tolist(), z['raw']):
                assert image.shape == (3,96,96) and image.dtype == np.float32
                assert np.isfinite(image).all() and 0 <= image.min() <= image.max() <= 1
                raw[update,cohort,wid,horizon] = image.copy()
    for cohort in indices:
        for row in read(root/f'cohort-{cohort}.metrics.json')['rows']:
            metrics[update,cohort,row['window_id'],row['horizon']] = row
frames = {}
for windows in indices.values():
    for w in windows:
        tid = w['trajectory_id']
        if tid not in frames:
            path = (manifest_path.parent/manifest['source_root']/entries[tid]['file']).resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == entries[tid]['sha256']
            hashes[str(path)] = digest
            with np.load(path, allow_pickle=False) as z:
                frames[tid] = z['rgb'].astype(np.float32)/255

def font(n):
    return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', n)
TITLE, SUB, ROW, SMALL = font(28), font(18), font(19), font(16)
GROUPS = {'train_visual':'训练样本', 'val_motion':'运动验证样本', 'val_natural':'常规锚点验证样本'}
STATUS = {'unknown':'未知：推杆相邻可能遮挡', 'failure':'检测到形状失败',
          'assessable_no_detected_failure':'未触发形状失败阈值（不等于严格保形）'}
generated = []
def figure(cohort, windows, horizon, name, title):
    width, header, rh = 1120, 138, 284
    canvas = Image.new('RGB',(width,header+rh*len(windows)+35),'white')
    d=ImageDraw.Draw(canvas)
    d.text((24,14),title,font=TITLE,fill='#172234')
    d.text((24,55),f'{GROUPS[cohort]} · H{horizon}：预测 {horizon/10:.1f} 秒后 · 全部固定为 3 次去噪',font=SUB,fill='#394554')
    d.text((24,83),'每行同一输入/动作/配对种子；相同坐标裁切，整数倍像素放大，无平滑或修形。',font=SMALL,fill='#586575')
    xs=[24,300,576,852]
    for x,label in zip(xs,['真实未来','训练 300 步','训练 1,000 步','训练 3,000 步']):
        d.text((x+22,110),label,font=ROW,fill='#172234')
    for j,w in enumerate(windows):
        y=header+j*rh; wid=w['window_id']; r=metrics[3000,cohort,wid,horizon]
        d.line((24,y,width-24,y),fill='#dbe2e9',width=1)
        d.text((24,y+8),f"{w['letter']}   {wid}   |   3,000步：{STATUS[r['shape_outcome']]}",font=ROW,fill='#263746')
        images=[frames[w['trajectory_id']][w['anchor']+horizon]]+[raw[u,cohort,wid,horizon] for u in (300,1000,3000)]
        mask=np.logical_or.reduce([(a<.98).any(axis=0) for a in images]); yy,xx=np.nonzero(mask)
        size=min(96,max(int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))+12)
        left=max(0,min(96-size,int((xx.min()+xx.max()+1-size)//2)))
        top=max(0,min(96-size,int((yy.min()+yy.max()+1-size)//2)))
        scale=max(1,218//size)
        for k,(x,a) in enumerate(zip(xs,images)):
            pixels=np.rint(a.transpose(1,2,0)*255).astype(np.uint8)
            tile=Image.fromarray(pixels).crop((left,top,left+size,top+size)).resize((size*scale,size*scale),Image.Resampling.NEAREST)
            canvas.paste(tile,(x+(244-tile.width)//2,y+36+(218-tile.height)//2))
            if k:
                m=metrics[(300,1000,3000)[k-1],cohort,wid,horizon]
                d.text((x+22,y+254),f"物体 IoU {m['raw_mask_iou']:.3f} · 孔洞 {m['predicted_holes']}/{m['truth_holes']}",font=SMALL,fill='#465768')
    d.text((24,canvas.height-27),'孔洞：预测 / 真值。IoU 同时受位置与形状影响，未触发失败阈值并非棱角完全一致。',font=SMALL,fill='#586575')
    canvas.save(OUT/name,optimize=True);generated.append(name)

focus_ids=['R2B-new-25-O-left-bottom:12','R2B-new-48-B-top:5','R2B-new-28-C-left:6','R2B-new-38-X-right:13']
lookup={w['window_id']:w for w in indices['val_motion']}
focus=[lookup[x] for x in focus_ids]
for h in (1,4):
    figure('val_motion',focus,h,f'corners-H{h}.png','棱角与孔洞：300 → 1,000 → 3,000 步')
gallery=[]
for c,ws in indices.items():
    for offset in range(0,len(ws),4):
        name=f'all-{c}-H4-{offset//4+1:02d}.png'
        figure(c,ws[offset:offset+4],4,name,'3,000 步：全部 36 个固定窗口的 H4 对照')
        gallery.append((GROUPS[c],name))
rows=[metrics[3000,c,w['window_id'],4] for c,ws in indices.items() for w in ws]
unknown=[r for r in rows if r['shape_outcome']=='unknown']
summary=dict(total=36,outcomes=dict(collections.Counter(r['shape_outcome'] for r in rows)),
             unknown_with_possible_occlusion=sum(bool(r['occlusion_ambiguous']) for r in unknown),
             unknown_with_object_border=sum(bool(r['prediction_object_border_touch'] or r['truth_object_border_touch']) for r in unknown))
parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>3,000步预测对照</title>',
       '<style>body{font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1180px;margin:24px auto;padding:0 20px;color:#172234}img{max-width:100%;height:auto}summary{cursor:pointer;font-weight:600}a{color:#175f9c}</style>',
       '<h1>3,000步：相同验证窗口的棱角和孔洞</h1>',
       '<p>四列依次为真值、300步、1,000步、3,000步。只用已保存的原始输出，统一3次去噪，不重新推理。四个重点窗口与上次1,000步展示完全相同；完整36个H4窗口在下方。</p>',
       '<p>图像为相同坐标的前景联合范围裁切、整数倍最近邻放大；没有位置对齐、平滑、修形或锐化。未知不算通过；未触发形状失败阈值也不代表每个棱角都准确。</p>']
for h in (1,4):
    parts += [f'<h2>H{h}：{h/10:.1f}秒后</h2>',f'<a href="corners-H{h}.png"><img src="corners-H{h}.png"></a>']
parts += ['<h2>完整36个H4窗口</h2>','<p>TRAIN与两个验证组分列。两组验证来自相同12条轨迹的不同窗口，不是互相独立的测试集。</p>']
for title,name in gallery:
    parts.append(f'<details><summary>{title} · {name}</summary><a href="{name}"><img loading="lazy" src="{name}"></a></details>')
parts.append('</html>');(OUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')
provenance=dict(kind='parent-saved-rgb-3000-figure.v1',model_calls=0,physics_controls=0,checkpoint_loads=0,new_scores=0,
                training_updates=[300,1000,3000],sampler_steps=3,focus_ids=focus_ids,
                selection='Same four O/B/C/X windows as1000 view; all36H4 included, no claim of representative statistics',
                rendering='RGB*255 rounded uint8; union of any channel<.98 plus6px margin; integer nearest neighbor, no alignment or shape repair',
                H4_summary=summary,source_hashes=hashes,figures=generated)
(OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(path=str(OUT),figure_count=len(generated),bytes=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()),H4=summary),ensure_ascii=False))
