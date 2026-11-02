"""CPU-only figures from immutable saved RGB; no model or simulator imports."""
from pathlib import Path
import collections
import hashlib
import html
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
def read(p):
    return json.loads(p.read_text(encoding="utf-8"))
indices = read(BASE / "r2c0/evaluation/indices.json")["cohorts"]
manifest_path = BASE / "r2c0/data/manifest.json"
manifest = read(manifest_path)
entries = {x["trajectory_id"]: x for x in manifest["entries"]}
raw = {}
source_hashes = {}
for update, root in [(300, BASE / "r2c1r1/run/evaluation/final/steps"),
                     (1000, BASE / "r2c3/run/evaluation/update1000/steps3/steps")]:
    for path in sorted(root.glob("*.npz")):
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        cohort = path.name.split("-batch")[0]
        step = int(path.stem.rsplit("step", 1)[1]) + 1
        with np.load(path, allow_pickle=False) as z:
            for wid, image in zip(z["window_ids"].tolist(), z["raw"]):
                assert image.shape == (3, 96, 96) and image.dtype == np.float32
                assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
                raw[update, cohort, wid, step] = image.copy()
metrics = {}
for update, root in [(300, BASE / "r2c1r1/run/evaluation/final"),
                     (1000, BASE / "r2c3/run/evaluation/update1000/steps3")]:
    for cohort in indices:
        for row in read(root / f"cohort-{cohort}.metrics.json")["rows"]:
            metrics[update, cohort, row["window_id"], row["horizon"]] = row
trajectories = {}
for cohort in indices:
    for w in indices[cohort]:
        tid = w["trajectory_id"]
        if tid not in trajectories:
            path = (manifest_path.parent / manifest["source_root"] / entries[tid]["file"]).resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == entries[tid]["sha256"]
            source_hashes[str(path)] = digest
            with np.load(path, allow_pickle=False) as z:
                trajectories[tid] = z["rgb"].astype(np.float32) / 255

FONT = "C:/Windows/Fonts/msyh.ttc"
def font(n):
    return ImageFont.truetype(FONT, n)
F_TITLE, F_SUB, F_ROW, F_SMALL = font(28), font(18), font(19), font(16)
COHORT_NAMES = {"train_visual": "训练样本", "val_motion": "运动验证样本", "val_natural": "常规锚点验证样本"}
OUTCOMES = {"unknown": "未知：可能被推杆遮挡", "failure": "检测到形状失败", "assessable_no_detected_failure": "未触发形状失败阈值（不等于严格保形）"}
files = []
def arrays(cohort, w, horizon):
    frames = trajectories[w["trajectory_id"]]
    return [frames[w["anchor"]], frames[w["anchor"] + horizon],
            raw[300, cohort, w["window_id"], horizon],
            raw[1000, cohort, w["window_id"], horizon]]

def common_box(images):
    foreground = np.logical_or.reduce([(x < .98).any(axis=0) for x in images])
    yy, xx = np.nonzero(foreground)
    assert len(xx)
    size = min(96, max(int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)) + 12)
    left = max(0, min(96-size, int((xx.min()+xx.max()+1-size)//2)))
    top = max(0, min(96-size, int((yy.min()+yy.max()+1-size)//2)))
    return left, top, left+size, top+size

def make_page(cohort, windows, horizon, name, title):
    width, header, row_h = 1120, 137, 284
    canvas = Image.new("RGB", (width, header + row_h*len(windows) + 35), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 14), title, font=F_TITLE, fill="#172234")
    draw.text((24, 55), f"{COHORT_NAMES[cohort]} · H{horizon}：预测 {horizon/10:.1f} 秒后 · 两模型均为 3 次去噪", font=F_SUB, fill="#394554")
    draw.text((24, 83), "每行采用相同坐标裁切；整数倍像素放大，无平滑、修形或位置对齐。", font=F_SMALL, fill="#586575")
    xpositions = [24, 300, 576, 852]
    for x, label in zip(xpositions, ["最后输入", "真实未来", "训练 300 步", "训练 1,000 步"]):
        draw.text((x+22, 110), label, font=F_ROW, fill="#172234")
    for j, w in enumerate(windows):
        y = header + j*row_h
        row = metrics[1000, cohort, w["window_id"], horizon]
        draw.line((24, y, width-24, y), fill="#dbe2e9", width=1)
        draw.text((24, y+8), f"{w['letter']}   {w['window_id']}   |   {OUTCOMES[row['shape_outcome']]}", font=F_ROW, fill="#263746")
        images = arrays(cohort, w, horizon)
        box = common_box(images)
        side = box[2]-box[0]
        scale = max(1, 218//side)
        for column, (x, arr) in enumerate(zip(xpositions, images)):
            pixels = np.rint(arr.transpose(1, 2, 0)*255).astype(np.uint8)
            tile = Image.fromarray(pixels).crop(box).resize((side*scale, side*scale), Image.Resampling.NEAREST)
            px, py = x+(244-tile.width)//2, y+36+(218-tile.height)//2
            canvas.paste(tile, (px, py))
            if column >= 2:
                m = metrics[[300, 1000][column-2], cohort, w["window_id"], horizon]
                draw.text((x+22, y+254), f"物体 IoU {m['raw_mask_iou']:.3f} · 孔洞 {m['predicted_holes']}/{m['truth_holes']}", font=F_SMALL, fill="#465768")
    draw.text((24, canvas.height-27), "孔洞显示为：预测数量 / 真值数量。IoU 同时受位置与形状影响，并非单独的保形分数。", font=F_SMALL, fill="#586575")
    path = OUT / name
    canvas.save(path, optimize=True)
    files.append(name)
    return name

focus_ids = ["R2B-new-25-O-left-bottom:12", "R2B-new-48-B-top:5", "R2B-new-28-C-left:6", "R2B-new-38-X-right:13"]
by_id = {w["window_id"]: w for w in indices["val_motion"]}
focus = [by_id[x] for x in focus_ids]
for h in [1, 4]:
    make_page("val_motion", focus, h, f"corners-H{h}.png", "棱角与孔洞：300 步和 1,000 步预测对照")
gallery = []
for cohort, windows in indices.items():
    for start in range(0, len(windows), 4):
        name = f"all-{cohort}-H4-{start//4+1:02d}.png"
        make_page(cohort, windows[start:start+4], 4, name, "1,000 步：全部 36 个固定样本的 H4 对照")
        gallery.append((COHORT_NAMES[cohort], name))

h4rows = [metrics[1000,c,w['window_id'],4] for c,ws in indices.items() for w in ws]
unknown = [r for r in h4rows if r['shape_outcome']=='unknown']
assert len(unknown)==24 and all(r['occlusion_ambiguous'] for r in unknown)
assert all(not r['prediction_object_border_touch'] and not r['truth_object_border_touch'] for r in unknown)
reason = dict(total_H4=36, unknown=24, unknown_due_possible_pusher_occlusion=24,
              unknown_with_object_viewport_border=0,
              outcome_counts=dict(collections.Counter(r['shape_outcome'] for r in h4rows)),
              unknown_by_cohort=dict(collections.Counter(r['cohort'] for r in unknown)))
page = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>1,000步预测对比</title>',
        '<style>body{font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1180px;margin:24px auto;padding:0 20px;color:#172234}img{max-width:100%;height:auto}summary{cursor:pointer;font-weight:600}p{max-width:1050px}a{color:#175f9c}</style>',
        '<h1>1,000 步预测：棱角与孔洞</h1>',
        '<p>来源为已保存的原始 RGB，不重新运行模型。固定 3 次去噪，输入、动作和配对随机种子一致；300 和 1,000 指训练更新数。主图选四个有棱角/孔洞问题的验证窗口，完整 H4 样本在下方。</p>',
        '<p><b>24 个“未知”是 36 个 H4 窗口中的 24 个，不是字母数量。</b>这 24 个全部因为真值或预测中推杆与物体在像素上相邻，被保守地视为可能遮挡；没有物体画面边缘触碰。未知不算通过，也不隐藏可见的形状问题。另有 5 个检测到失败、7 个未触发失败阈值；后者同样不是严格刚性保证。</p>']
for h in [1,4]:
    page += [f'<h2>H{h}：{h/10:.1f} 秒后</h2>',f'<a href="corners-H{h}.png"><img src="corners-H{h}.png" alt="棱角对照H{h}"></a>']
page.append('<h2>全部 36 个 H4 样本</h2><p>保留原来的窗口顺序；三组各 12 个。TRAIN 与验证分开，两个验证组来自相同 12 条轨迹的不同窗口，不能当作互相独立的测试集。</p>')
for label,name in gallery:
    page.append(f'<details><summary>{html.escape(label)} · {html.escape(name)}</summary><a href="{name}"><img loading="lazy" src="{name}"></a></details>')
page.append('</html>')
(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
metadata = dict(kind='parent-cpu-saved-rgb-1000-visual-review.v1', model_calls=0, physics_controls=0,
                checkpoint_loads=0, score_recomputations=0, sampler_steps=3, focus_window_ids=focus_ids,
                focus_selection='O/B/C/X corner and hole inspection, not a statistical sample; complete36 H4 included',
                display='RGB*255 rounded to uint8; shared coordinates union of any channel below .98 plus6px margin; integer nearest-neighbor; no registration or shape repair',
                unknown_summary=reason, source_files_sha256=source_hashes, figures=files)
(OUT/'provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),figure_count=len(files),bytes=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()),unknown=reason),ensure_ascii=False))
