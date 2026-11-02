"""CPU-only, bounded visible replay of the recorded A run; no policy loaded."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.pop('SDL_VIDEODRIVER', None)
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from tdyssey.official import load_official_source
from tdyssey.envs.official import make_official_shape_env_class
from tdyssey.simulator_planning import state_error
from tdyssey.state import Snapshot
from tdyssey.viewer import PushTViewer
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=int, default=180)
args = parser.parse_args()
report = json.loads((ROOT / 'artifacts/td-abc-20261021-A/report.json').read_text(encoding='utf-8'))
cls = make_official_shape_env_class(load_official_source(ROOT / 'third_party/diffusion_policy'))
viewer = PushTViewer(title='Tdyssey | RECORDED ACTUAL RUN | Esc to close', window_size=768)
frames = {}
try:
    for episode in report['episodes']:
        env = cls(shape=episode['shape'], legacy=True)
        try:
            env.seed(episode['scene_seed'])
            env.reset()
            env.restore_state(Snapshot.from_dict(episode['initial_state']))
            images = [env.render(mode='rgb_array')]
            for row in episode['steps']:
                _, reward, done, _ = env.step(np.asarray(row['action'], dtype=np.float64))
                error = state_error(env.capture_state(), Snapshot.from_dict(row['state']))
                if error > 1e-7 or abs(reward-row['reward']) > 1e-9 or bool(done) != row['upstream_done']:
                    raise ValueError(f'Replay diverged: {episode["shape"]}, error={error}')
                images.append(env.render(mode='rgb_array'))
            frames[episode['shape']] = images
            output = ROOT / f'artifacts/reranking-20261021/{episode["shape"]}-baseline-eight-replay.mp4'
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), 10, (512,512))
            for image in images:
                writer.write(cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (512,512)))
            writer.release()
            capture = cv2.VideoCapture(str(output))
            decoded = 0
            while capture.read()[0]:
                decoded += 1
            capture.release()
            if decoded != len(images):
                raise ValueError(f'Video read-back failed: {decoded} != {len(images)}')
            print(f'{episode["shape"]}: verified {len(episode["steps"])} controls, exported {output}', flush=True)
        finally:
            env.close()
    until = time.monotonic() + args.seconds
    while not viewer.closed and time.monotonic() < until:
        for episode in report['episodes']:
            shape = episode['shape']
            for step, image in enumerate(frames[shape]):
                if not viewer.draw(image, f'Tdyssey | REPLAY A: no ranking, execute 8 | {shape} | {step}/{len(frames[shape])-1} | Esc closes', 10):
                    break
            if viewer.closed:
                break
finally:
    viewer.close()
