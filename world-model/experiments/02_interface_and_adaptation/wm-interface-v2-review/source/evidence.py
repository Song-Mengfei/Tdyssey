"""Annotate existing saved RGB; no decoder, environment or model is imported."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).resolve().parents[2]
source=root/'artifacts/wm-interface-v2-development'
rows={r['id']:r for r in map(json.loads,(source/'per-image.jsonl').read_text(encoding='utf-8').splitlines())}
ids=('L00-hold-noise-5478-H1','L06-hold-noise-5478-H1','T13-hold-noise-5479-H1',
     'L06-toward-noise-5480-H3','T13-toward-noise-5478-H2','T14-toward-noise-5478-H4')
font=ImageFont.truetype('arial.ttf',20)
small=ImageFont.truetype('arial.ttf',18)
canvas=Image.new('RGB',(1240,len(ids)*310),'white')
draw=ImageDraw.Draw(canvas)
def rgb(row):
    with np.load(row['path'],allow_pickle=False) as archive:
        image=archive[row['key']][row['index']]
    if image.shape==(3,96,96):
        image=np.moveaxis(image,0,-1)
    return Image.fromarray(np.clip(image*255,0,255).astype('uint8')).resize((256,256),Image.Resampling.NEAREST)
def number(value):
    return 'unavailable' if value is None else f'{value:.5f}'
for index,identity in enumerate(ids):
    row=rows[identity]
    physical=rows[f"{row['anchor']}-{row['candidate']}-future-H{row['horizon']}"]
    y=index*310
    draw.text((10,y+2),identity,font=font,fill='black')
    draw.text((10,y+28),'Saved actual RGB',font=small,fill='black')
    draw.text((280,y+28),'Saved forecast RGB',font=small,fill='black')
    canvas.paste(rgb(physical),(10,y+51))
    canvas.paste(rgb(row),(280,y+51))
    record=row['v2']['measurement']
    errors=row['v2']['errors']
    shape=record['shape']
    lines=[f"v1 object valid: {row['v1']['view']['object_valid']}",
           f"v2 object valid: {row['v2']['view']['object_valid']}",
           f"Reason: {shape['reason']}",
           f"Area ratio: {number(shape.get('area_ratio'))}",
           f"Full soft IoU: {number(shape.get('soft_iou'))}",
           f"Pusher valid: {row['v2']['view']['pusher_valid']}",
           f"Coverage estimate / truth: {number(row['v2']['view']['coverage'])} / {number(errors['truth_coverage'])}",
           f"Position error px: {number(errors['position_error_pixels'])}",
           f"Angle error deg: {number(errors['angle_error_degrees'])}",
           f"Actual rotation deg: {number(errors['actual_rotation_degrees'])}",
           f"Termination: {row['v2']['view']['termination']} (development interval)"]
    for line_index,line in enumerate(lines):
        draw.text((560,y+52+line_index*22),line,font=small,fill='black')
path=Path(__file__).resolve().parent/'saved-rgb-evidence.png'
if path.exists():
    raise SystemExit('Refusing to replace saved evidence')
canvas.save(path)
print(json.dumps(dict(output=str(path),bytes=path.stat().st_size,decoder_calls=0,real_render_calls=0)))
