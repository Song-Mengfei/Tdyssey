"""Preview implemented geometry and scores without a simulator or dependencies."""

import argparse
from dataclasses import asdict
from html import escape
import json
import math
from pathlib import Path

from .evaluation import goal_metrics
from .geometry import Pose, get_shape


def shape_svg(scale=30.0) -> str:
    shapes = [get_shape(name, scale) for name in ("T", "I", "L")]
    extent = max(abs(coord) for shape in shapes for rect in shape.components
                 for vertex in rect.vertices for coord in vertex)
    factor = 85 / extent
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="660" height="260" viewBox="0 0 660 260">',
             '<rect width="660" height="260" fill="white"/>',
             '<text x="20" y="24" font-family="sans-serif" font-size="16">T / I / L geometry preview — no physics or policy rollout</text>']
    for index, shape in enumerate(shapes):
        pose = Pose((110 + 220 * index, 130), 0)
        for rect in shape.components:
            vertices = [pose.transform((x * factor, y * factor)) for x, y in rect.vertices]
            coords = " ".join(f"{x:.4f},{y:.4f}" for x, y in vertices)
            parts.append(f'<polygon points="{coords}" fill="#527aaf" stroke="#29445f"/>')
        parts.append(f'<circle cx="{pose.position[0]}" cy="130" r="3" fill="#e8a12b"/>')
        parts.append(f'<text x="{80 + 220 * index}" y="240" font-family="sans-serif" font-size="14">{escape(shape.name)}: {shape.symmetry_order}-fold symmetry</text>')
    return "\n".join([*parts, '</svg>']) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--svg", type=Path, help="Write a small optional geometry preview")
    args = parser.parse_args()
    report = {"kind": "geometry_checks_not_physical_performance", "shapes": {}}
    for name in ("T", "I", "L"):
        shape = get_shape(name, args.scale)
        report["shapes"][name] = {
            "area_pixels_squared": shape.area,
            "centroid": shape.centroid,
            "origin": shape.origin,
            "geometry_hash": shape.geometry_hash,
            "half_turn_metrics": asdict(goal_metrics(shape, Pose((256, 256), math.pi), Pose((256, 256), 0))),
        }
    if args.svg:
        args.svg.write_text(shape_svg(args.scale), encoding="utf-8")
        report["preview"] = str(args.svg)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
