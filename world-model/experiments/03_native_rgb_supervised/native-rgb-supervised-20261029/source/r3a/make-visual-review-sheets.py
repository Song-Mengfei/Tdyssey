"""Arrange existing exact PNG strips for human visual QA; no data changes."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

stage=Path(__file__).resolve().parent
source=stage/'report/videos'
records=json.loads((source/'video-provenance.json').read_text(encoding='utf-8'))['records']
out=stage/'report/qa';out.mkdir(exist_ok=True)
try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
except OSError:font=ImageFont.load_default()
for start in range(0,len(records),5):
    rows=records[start:start+5];canvas=Image.new('RGB',(960,len(rows)*220),'white');draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        label=row['trajectory_id']+'  '+', '.join(role.split(':',1)[1]for role in row['roles'])
        draw.text((5,i*220+3),label,fill='black',font=font)
        canvas.paste(Image.open(source/row['strip']).convert('RGB'),(0,i*220+26))
    canvas.save(out/f'overview-{start//5+1:02d}.png')
failure=stage/'report/failure-videos'
items=json.loads((failure/'failure-video-provenance.json').read_text(encoding='utf-8'))['records']
canvas=Image.new('RGB',(1152,2*430),'white');draw=ImageDraw.Draw(canvas)
for i,row in enumerate(items[:6]):
    x=(i%3)*384;y=(i//3)*430
    draw.text((x+4,y+3),row['trajectory_id'],fill='black',font=font)
    draw.text((x+4,y+22),f"{row['status']}; last actual t={row.get('actual_substep_chain_last_time_seconds')}s",fill='black',font=font)
    if row.get('actual_last_substep_image'):canvas.paste(Image.open(failure/row['actual_last_substep_image']).convert('RGB'),(x,y+44))
canvas.save(out/'failed-last-actual-states.png')
print(f'{(len(records)+4)//5} overview sheets plus failed actual states saved')
