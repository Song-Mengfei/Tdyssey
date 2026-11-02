"""Reconstruct visible inference pauses from recorded timings, without rerunning policy."""
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
run = ROOT / 'artifacts/td-baseline-300-20261021'
report = json.loads((run / 'report.json').read_text(encoding='utf-8'))
episode = next(e for e in report['episodes'] if e['shape'] == 'D')
source = run / 'D-100000.mp4'
output = run / 'D-100000-with-inference-pauses.mp4'
if output.exists():
    raise FileExistsError(output)
capture = cv2.VideoCapture(str(source))
fps = 20
ok, image = capture.read()
if not ok:
    raise ValueError('Cannot read the starting frame')
height, width = image.shape[:2]
writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
if not writer.isOpened():
    raise RuntimeError('Cannot open paused video writer')
clock = 0.0
written = 0
control = 0
schedule = []

def hold(frame, seconds, reason, at_control):
    global clock, written
    start = clock
    clock += seconds
    target_frames = round(clock * fps)
    count = target_frames - written
    for _ in range(count):
        writer.write(frame)
    written += count
    schedule.append({'control': at_control, 'kind': reason, 'seconds': seconds,
                     'start_seconds': start, 'video_frames': count})

try:
    for pause in episode['planning_seconds']:
        hold(image, pause, 'recorded_inference_wait', control)
        count = min(report['config']['execute_steps'], len(episode['steps']) - control)
        for within_chunk in range(count):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f'Missing source action frame {control+1}')
            control += 1
            # The viewer waits 0.1 s between controls inside a sampled chunk.
            # At a chunk boundary the next measured inference wait takes over.
            if within_chunk + 1 < count:
                hold(image, 0.1, 'control_display_interval', control)
    if control != len(episode['steps']) or capture.read()[0]:
        raise ValueError('Action/video frame count mismatch')
    hold(image, 0.1, 'final_frame', control)
finally:
    writer.release()
    capture.release()

capture = cv2.VideoCapture(str(output))
decoded = 0
while capture.read()[0]:
    decoded += 1
capture.release()
if decoded != written:
    raise ValueError(f'Failed encoded frame check: {decoded} != {written}')
result = {'kind': 'inference_timing_reconstruction', 'source_video': str(source.relative_to(ROOT)),
          'controls': control, 'inference_pauses': len(episode['planning_seconds']),
          'measured_inference_wait_seconds': sum(episode['planning_seconds']),
          'fps': fps, 'frames': decoded, 'duration_seconds': decoded/fps,
          'path': str(output), 'schedule': schedule,
          'timing_limit': 'Uses measured policy waits and nominal viewer control intervals; not a desktop capture.'}
output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'schedule'}, indent=2))
