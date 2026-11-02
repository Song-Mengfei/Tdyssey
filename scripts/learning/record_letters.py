"""Record one video per letter for a policy checkpoint, every letter on the SAME starting scene and policy seed.

    .venv/bin/python -B scripts/learning/record_letters.py --student model.pt --out DIR [--seed 100000] [--policy-seed 0]

Scene: matched_reset (the repository's common-scene rule, also used for docs/media/official-alphabet-2026-10-19/):
the first seed >= --seed that is a valid start for every letter; the block pose is matched across letters. Every
letter's policy noise comes from its own generator seeded with --policy-seed, so letters differ only by their shape.
All letters run together (one batched denoiser call per decision); 8 of the 16 planned targets are executed.
Writes <LETTER>.mp4 (512x512, 10 fps, H.264), all_letters_grid.mp4, final_frames.png and summary.json.
"""
import argparse
import json
import string
import sys
from collections import deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import base_rollouts  # noqa: E402
from tdyssey.envs.official import matched_reset  # noqa: E402
from tdyssey.official import physical_metrics  # noqa: E402
from tdyssey.viewer import presentation_frame  # noqa: E402

TRAINED = {"T", "Y", "D", "L", "E", "F", "J", "P", "U"}          # (near-T is the tenth trained shape, not a letter)


def caption(img, top, bottom, ok=None, right=""):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(img, top, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    if right:
        w = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(img, right, (img.shape[1] - w - 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
    color = (20, 140, 60) if ok else (30, 30, 30)
    cv2.rectangle(img, (0, img.shape[0]-30), (img.shape[1], img.shape[0]), (255, 255, 255), -1)
    cv2.putText(img, bottom, (8, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1 + bool(ok), cv2.LINE_AA)
    return img


def writer(path, size, fps):
    w = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=8,
                                    macro_block_size=16)
    w.send(None)
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--label", default="ten-letter policy")
    p.add_argument("--letters", nargs="*", default=list(string.ascii_uppercase))
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    agent, env_class = base_rollouts.build(str(a.student))
    envs = {L: env_class(shape=L, legacy=True) for L in a.letters}
    obs, records = matched_reset(envs, a.seed, {"reset_physics_substeps": 0})
    scene = records[-1]["seed"]
    print(f"common starting scene: seed {scene} (tried {len(records)})", flush=True)
    hist = {L: deque([obs[L]] * 2, maxlen=2) for L in a.letters}
    gens = {L: torch.Generator(device=agent.device).manual_seed(a.policy_seed) for L in a.letters}
    S = a.size; tile = 256; cols = 6; rows = -(-len(a.letters) // cols)
    vids = {L: writer(a.out/f"{L}.mp4", (S, S), a.fps) for L in a.letters}
    grid = writer(a.out/"all_letters_grid.mp4", (cols*tile, rows*tile), a.fps)
    state = {L: {"controls": 0, "done": False, "coverage": 0.0, "max_coverage": 0.0} for L in a.letters}
    last, tiles = {}, {}

    def frame(L):
        s = state[L]; tag = "trained letter" if L in TRAINED else "never trained on"
        if s["done"]:
            bottom, ok = f"SUCCESS after {s['controls']} controls ({s['coverage']:.0%} covered)", True
        elif s["controls"] >= a.steps:
            bottom, ok = f"not solved in {a.steps} controls: {s['coverage']:.0%} covered", False
        else:
            bottom, ok = f"control {s['controls']}/{a.steps}   goal covered {s['coverage']:.0%}", None
        rawf = presentation_frame(envs[L], S)
        img = caption(rawf, f"{L}  ({tag})", bottom, ok, right=f"{a.label}, scene {scene}")
        last[L] = img
        t = cv2.resize(rawf, (tile, tile), interpolation=cv2.INTER_AREA)       # grid tile: big letter + short status
        cv2.putText(t, L, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 40) if L in TRAINED else (150, 150, 150), 2, cv2.LINE_AA)
        if L in TRAINED:
            cv2.putText(t, "trained", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        short = f"solved at {s['controls']}" if s["done"] else f"{s['coverage']:.0%} covered"
        cv2.putText(t, short, (8, tile - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 140, 60) if s["done"] else (60, 60, 60),
                    2 if s["done"] else 1, cv2.LINE_AA)
        cv2.rectangle(t, (0, 0), (tile - 1, tile - 1), (200, 200, 200), 1)
        tiles[L] = t
        return img

    def write_grid():
        g = np.full((rows*tile, cols*tile, 3), 255, np.uint8)
        for k, L in enumerate(a.letters):
            r, c = divmod(k, cols)
            g[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = tiles[L]
        grid.send(np.ascontiguousarray(g))

    for L in a.letters:
        state[L]["coverage"] = state[L]["max_coverage"] = float(physical_metrics(envs[L])["coverage"])
        vids[L].send(np.ascontiguousarray(frame(L)))
    write_grid()
    active = list(a.letters)
    while active:
        acts, _ = agent.sample_batch(agent.encode_batch([hist[L] for L in active]), [gens[L] for L in active])
        acts = acts.cpu().numpy() if torch.is_tensor(acts) else np.asarray(acts)
        for j in range(acts.shape[1]):                                     # execute the 8 planned targets in step
            for i, L in enumerate(active):
                s = state[L]
                if s["done"] or s["controls"] >= a.steps:
                    continue
                o, _, done, _ = envs[L].step(acts[i, j])
                hist[L].append(o); s["controls"] += 1; s["done"] = bool(done)
                s["coverage"] = float(physical_metrics(envs[L])["coverage"]); s["max_coverage"] = max(s["max_coverage"], s["coverage"])
                vids[L].send(np.ascontiguousarray(frame(L)))
            write_grid()
        active = [L for L in active if not state[L]["done"] and state[L]["controls"] < a.steps]
        print(f"  {sum(s['done'] for s in state.values())} solved, {len(active)} still running", flush=True)
    for L in a.letters:                                                    # hold the final frame for 1.5 s
        for _ in range(int(1.5*a.fps)):
            vids[L].send(np.ascontiguousarray(last[L]))
        vids[L].close()
    for _ in range(3*a.fps):
        write_grid()
    grid.close()
    g = np.full((rows*tile, cols*tile, 3), 255, np.uint8)
    for k, L in enumerate(a.letters):
        r, c = divmod(k, cols); g[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = tiles[L]
    cv2.imwrite(str(a.out/"final_frames.png"), cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
    summary = {"student": str(a.student.name), "scene_seed": scene, "policy_seed": a.policy_seed, "max_controls": a.steps,
               "success_rule": "upstream PushT: block covers > 95% of the goal",
               "letters": {L: {"trained": L in TRAINED, "success": s["done"], "controls": s["controls"],
                               "final_coverage": round(s["coverage"], 3), "max_coverage": round(s["max_coverage"], 3)}
                           for L, s in state.items()}}
    (a.out/"summary.json").write_text(json.dumps(summary, indent=1))
    print("solved:", " ".join(L for L, s in state.items() if s["done"]) or "none")
    print("trained letters solved:", sum(state[L]["done"] for L in a.letters if L in TRAINED), "/", sum(L in TRAINED for L in a.letters),
          "| untrained solved:", sum(state[L]["done"] for L in a.letters if L not in TRAINED), "/", sum(L not in TRAINED for L in a.letters))


if __name__ == "__main__":
    main()
