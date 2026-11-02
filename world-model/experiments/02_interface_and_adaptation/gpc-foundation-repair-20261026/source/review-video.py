"""One bounded CPU decode of sixteen predeclared uniform video frames."""
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np

root = Path(__file__).resolve().parent
source = root.parent / 'manual-world-model/20261026-115956-905921/121126-971032.mp4'
out = root / 'video-review'
out.mkdir(exist_ok=False)
started = time.monotonic()
cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
try:
    if not cap.isOpened():
        raise RuntimeError('Existing video could not be opened')
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    indices = np.linspace(0, count-1, 16).round().astype(int).tolist()
    sheet = np.full((4*290, 4*500, 3), 235, np.uint8)
    rows = []
    for index, frame_number in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Decode failed at frame {frame_number}')
        name = f'frame-{frame_number:06d}.png'
        if not cv2.imwrite(str(out / name), frame):
            raise OSError('Video frame write failed')
        preview = cv2.resize(frame, (500, 269), interpolation=cv2.INTER_AREA)
        y, x = (index//4)*290, (index%4)*500
        sheet[y:y+269,x:x+500] = preview
        label = f'{frame_number/fps:.1f}s / frame {frame_number}'
        cv2.putText(sheet, label, (x+5,y+285), cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1)
        rows.append(dict(frame=frame_number, seconds=frame_number/fps, path=name))
    if not cv2.imwrite(str(out / 'contact-sheet.png'), sheet):
        raise OSError('Contact sheet write failed')
    result = dict(source=str(source), source_bytes=source.stat().st_size,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        fps=fps, frames=count, duration_seconds=count/fps, sampled=rows,
        seconds=time.monotonic()-started, real_models=0, physics=0, training=0)
    (out / 'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
finally:
    cap.release()
