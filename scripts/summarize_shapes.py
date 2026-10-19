"""Create a portable HTML/video gallery and CSV from a completed shape report."""

import argparse
import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


def write_summary(report_path, output):
    report_path, output = Path(report_path), Path(output)
    report = json.loads(report_path.read_text())
    fields = ("shape", "scene_seed", "success", "stop_reason", "control_steps", "max_iou", "final_iou", "max_reward", "contact_steps")
    episodes = report["episodes"]
    rows = [{name: episode.get(name) for name in fields} for episode in episodes]
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    escape = lambda value: html.escape(str(value), quote=True)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Frozen T-policy letter probes</title>',
             '<style>body{font:16px system-ui;margin:24px auto;padding:0 18px;max-width:1150px;color:#192838;background:#f7f9fc}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:9px;border-bottom:1px solid #dde3eb}a{color:#1757ab}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}article{padding:15px;background:white;border:1px solid #dde3eb;border-radius:10px}video{width:100%;aspect-ratio:1;background:#fff}small{color:#526278}h1{font-size:26px}h2{font-size:21px}</style>',
             '<h1>Frozen T-policy letter probes</h1>',
             '<p>The official T-trained visual CNN receives each new shape without training or adaptation. Policy inference uses CUDA; physics uses the original CPU simulator.</p>',
             f'<p>Report status: <b>{escape(report["status"])}</b>. Completed episode records: {len(episodes)}. Success uses the original coverage threshold; IoU measures overlap separately. A single scene per shape is an exploratory check, not a reliability estimate.</p>',
             '<p>Mass is held fixed. Letter outline, contact geometry, density and inertia can differ. Block-font results apply to these specific shapes. T_narrow changes only the top-bar width, retaining original mass, inertia and physical-center convention.</p>',
             '<table><tr><th>Shape / scene</th><th>Outcome</th><th>Controls</th><th>Best IoU</th><th>Final IoU</th><th>Contact steps</th></tr>']
    for i, row in enumerate(rows):
        outcome = 'Solved' if row['success'] else ('Not solved in budget' if row['stop_reason'] == 'control_budget' else row['stop_reason'])
        value = lambda key: '—' if row[key] is None else f'{row[key]:.3f}'
        parts.append(f'<tr><td><a href="#episode-{i}">{escape(row["shape"])} / {escape(row["scene_seed"])}</a></td><td>{escape(outcome)}</td><td>{escape(row["control_steps"])}</td><td>{value("max_iou")}</td><td>{value("final_iou")}</td><td>{escape(row["contact_steps"])}</td></tr>')
    parts.append('</table><h2>Recorded attempts</h2><p>Videos include the starting frame and one frame after each control. Playback speed is 10 frames per second, independent of inference time.</p><div class="gallery">')
    for i, row in enumerate(rows):
        parts.append(f'<article id="episode-{i}"><h2>{escape(row["shape"])}</h2>')
        path = report_path.parent / f'{row["shape"]}-{row["scene_seed"]}.mp4'
        if path.is_file():
            relative = quote(os.path.relpath(path, output.parent).replace(os.sep, '/'), safe='/')
            parts.append(f'<video controls muted playsinline preload="none" src="{escape(relative)}"></video>')
        else:
            parts.append('<p>No recorded video available.</p>')
        parts.append(f'<p>{"Solved" if row["success"] else "Not solved"} · best IoU {row["max_iou"]:.3f} · {escape(row["control_steps"])} controls</p></article>')
    parts.append('</div></html>')
    output.write_text('\n'.join(parts)+'\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = write_summary(args.report, args.output)
    print(json.dumps({'episodes': len(rows), 'solved': sum(row['success'] for row in rows), 'html': str(args.output), 'csv': str(args.output.with_suffix('.csv'))}))


if __name__ == '__main__':
    main()
