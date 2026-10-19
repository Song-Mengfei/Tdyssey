"""Extract best-IoU screenshots from saved videos; no policy inference or physics."""

import argparse
import json
from pathlib import Path

from summarize_shapes import peak_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    import cv2

    report = json.loads(args.report.read_text())
    saved = []
    for episode in report['episodes']:
        peak = peak_diagnostics(episode)['best_step']
        if peak is None:
            continue
        stem = f'{episode["shape"]}-{episode["scene_seed"]}'
        source = args.report.parent / (stem + '.mp4')
        if not source.is_file():
            continue
        video = cv2.VideoCapture(str(source))
        try:
            # Sequential decode avoids backend-dependent keyframe seeking.
            frame = None
            for index in range(peak+1):
                ok, frame = video.read()
                if not ok:
                    raise ValueError(f'{source.name} ends before peak frame {peak}')
            output = args.report.parent / (stem + '-best.png')
            if not cv2.imwrite(str(output), frame):
                raise ValueError(f'Could not save {output.name}')
            saved.append({'file': output.name, 'control_step': peak})
        finally:
            video.release()
    print(json.dumps({'saved': saved}))


if __name__ == '__main__':
    main()
