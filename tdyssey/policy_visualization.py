"""Portable offline views of measured policy samples and physical outcomes."""

import base64
import json
from pathlib import Path


def image_url(frame):
    import cv2

    ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Could not encode a diagnostic frame")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def draw_targets(ax, frame, samples, executed, final=False):
    import numpy as np

    ax.imshow(frame, extent=(0, 512, 512, 0), interpolation="nearest")
    ax.set(xlim=(0, 512), ylim=(512, 0), xticks=[], yticks=[])
    ax.scatter(samples[:, 0], samples[:, 1], s=19, color="#111111", zorder=3)
    if final:
        ax.plot(samples[:, 0], samples[:, 1], color="#777777", lw=.9)
        points = samples[executed]
        ax.plot(points[:, 0], points[:, 1], "o-", color="#157d42", lw=1.5, markersize=4)
        for index in executed:
            ax.annotate(str(index), samples[index], xytext=(3, 3), textcoords="offset points", fontsize=7)
    outside = int(np.any((samples < 0) | (samples > 512), axis=1).sum())
    if outside:
        ax.text(.03, .97, f"{outside}/16 targets outside view", transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})


def save_overview(output, view):
    import matplotlib.pyplot as plt

    trace, episode = view["trace"], view["episode"]
    figure, axes = plt.subplots(2, 4, figsize=(16, 8.8), constrained_layout=True)
    samples, executed = trace["pixel_samples"], trace["executed_indices"]
    stages = [0, 1, 10, 25, 50, len(samples) - 1]
    for axis, stage in zip(axes.flat, stages):
        draw_targets(axis, view["frames"][0], samples[stage], executed, final=stage == len(samples) - 1)
        axis.set_title("Initial Gaussian noise" if stage == 0 else f"After {stage} denoising steps", fontsize=11)
    axis = axes.flat[6]
    peak_offset = view["facts"]["best_control"] - view["start_control"]
    frame = view["frames"][min(max(peak_offset, 0), len(view["frames"]) - 1)]
    axis.imshow(frame, interpolation="nearest")
    axis.set(xticks=[], yticks=[], title=f"Physical scene at best control {view['facts']['best_control']}")
    axis = axes.flat[7]
    steps = list(range(1, len(episode["steps"]) + 1))
    axis.plot(steps, [r["metrics"]["coverage"] for r in episode["steps"]], label="Coverage")
    axis.plot(steps, [r["metrics"]["iou"] for r in episode["steps"]], label="IoU", color="#b66a1c")
    axis.axhline(.95, color="#555555", ls="--", label="Success coverage > 0.95")
    axis.axvline(view["facts"]["best_control"], color="#a32424", lw=1)
    axis.axvspan(view["start_control"], view["start_control"] + len(view["frames"]) - 1, color="#157d42", alpha=.15)
    axis.set(xlabel="Physical control step", ylabel="Overlap", ylim=(0, 1.03), title="Saved full physical rollout")
    axis.legend(fontsize=7, loc="lower left")
    figure.suptitle(f"{view['shape']}: actual frozen-policy denoising before controls "
                   f"{view['start_control'] + 1}-{view['start_control'] + len(view['frames']) - 1}\n"
                   "Dots are pusher action targets, not object predictions. Green = executed prefix; other points are unused.",
                   fontsize=14)
    figure.savefig(output / f"{view['shape']}-denoising.png", dpi=140)
    plt.close(figure)


def save_video(output, view):
    """Small self-contained clip: real denoising followed by actual execution."""
    import cv2
    import numpy as np

    path = output / f"{view['shape']}-denoising.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1200, 704))
    if not writer.isOpened():
        raise ValueError("Installed OpenCV cannot encode diagnostic video")
    samples = view["trace"]["pixel_samples"]
    indices = view["trace"]["executed_indices"]

    def frame(stage, physical):
        canvas = np.full((704, 1200, 3), 255, dtype=np.uint8)
        def label(value, x, y, scale=.62):
            cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), 1, cv2.LINE_AA)
        label(f"{view['shape']} | original T-trained policy | recorded plan at control {view['start_control']}", 25, 30, .7)
        label(f"Denoising: {stage}/{len(samples)-1} (scene fixed)", 25, 66)
        label(f"Physical execution: {physical}/{len(view['frames'])-1}", 640, 66)
        for left, image in ((25, view["frames"][0]), (640, view["frames"][physical])):
            canvas[85:597, left:left+512] = cv2.cvtColor(cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
        for i, (x, y) in enumerate(samples[stage]):
            if 0 <= x <= 511 and 0 <= y <= 511:
                color = (45, 130, 35) if stage == len(samples)-1 and i in indices else (0, 0, 0)
                cv2.circle(canvas, (25+round(float(x)), 85+round(float(y))), 5, color, -1, cv2.LINE_AA)
        for a, b in zip(indices, indices[1:]):
            p, q = samples[-1, a], samples[-1, b]
            if np.all((p >= 0) & (p <= 511)) and np.all((q >= 0) & (q <= 511)):
                cv2.line(canvas, (640+round(float(p[0])), 85+round(float(p[1]))),
                         (640+round(float(q[0])), 85+round(float(q[1]))), (45, 130, 35), 2, cv2.LINE_AA)
        outside = np.any((samples[stage] < 0) | (samples[stage] > 512), axis=1).sum()
        label(f"Black dots: noisy pusher targets. Outside view: {outside}/16", 25, 628)
        label("Green path: final executable prefix (indices 1-8).", 25, 656)
        label("These are action samples, not predicted object positions or an explanation of internal reasoning.", 25, 684, .55)
        return canvas

    try:
        for stage in [0] * 12 + list(range(len(samples))) + [len(samples)-1] * 12:
            writer.write(frame(stage, 0))
        for physical in range(len(view["frames"])):
            for _ in range(6):
                writer.write(frame(len(samples)-1, physical))
    finally:
        writer.release()


def write_policy_views(output, views, report, video=False):
    import numpy as np

    cases = {case["shape"]: case for case in report["episodes"]}
    payload = {"views": [], "cases": cases}
    for view in views:
        samples = view["trace"]["pixel_samples"]
        item = {"shape": view["shape"], "start": view["start_control"],
                "samples": samples.round(4).tolist(), "timesteps": view["trace"]["timesteps"],
                "indices": view["trace"]["executed_indices"],
                "frames": [image_url(frame) for frame in view["frames"]],
                "positions": view["positions"], "input_positions": view["trace"]["conditioning_pusher_positions"].tolist(),
                "extent": [float(min(-32, samples.min() - 16)), float(max(544, samples.max() + 16))],
                "curve": [{"coverage": r["metrics"]["coverage"], "iou": r["metrics"]["iou"],
                           "contacts": r["n_contacts"]} for r in view["episode"]["steps"]]}
        payload["views"].append(item)
        if view["start_control"] == view["facts"]["plan_start_control"]:
            save_overview(output, view)
            if video:
                save_video(output, view)
    template = Path(__file__).with_name("policy_trace_template.html").read_text()
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__TRACE_DATA__", serialized))
