"""Write a small SVG catalog of exact collision rectangles, without dependencies."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdyssey.envs.official import official_shape
from tdyssey.letters import ALL_SHAPES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="900" viewBox="0 0 900 900">',
             '<rect width="900" height="900" fill="white"/>',
             '<text x="20" y="25" font-family="sans-serif" font-size="18">Official PushT letter probes — exact rigid-body geometry</text>']
    for index, name in enumerate(ALL_SHAPES):
        shape = official_shape(name)
        left = min(r.x0 for r in shape.components)
        top = min(r.y0 for r in shape.components)
        width = max(r.x1 for r in shape.components) - left
        x, y = 20 + (index % 6) * 148, 55 + (index // 6) * 165
        parts.append(f'<g transform="translate({x + (120-width)/2-left},{y-top})" fill="#778899">')
        for r in shape.components:
            parts.append(f'<rect x="{r.x0:.6f}" y="{r.y0:.6f}" width="{r.x1-r.x0:.6f}" height="{r.y1-r.y0:.6f}"/>')
        parts.append('</g>')
        parts.append(f'<text x="{x+60}" y="{y+143}" text-anchor="middle" font-family="sans-serif" font-size="16">{name}</text>')
    parts.append('</svg>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(parts)+'\n')


if __name__ == '__main__':
    main()
