"""CPU checks of the goal overlay (no GPU; run from the repository root with env.sh sourced).

  identity   overlay off renders exactly what the code before the overlay rendered (frames replayed from teacher
             demonstrations against their stored frames/ files, and stepped episodes against a source tree of
             --baseline-rev); overlay on with nothing uncovered renders the official image.
  geometry   the overlay's fast goal-minus-block region equals the coverage metric's (pymunk_to_shapely) region.
  response   translated/rotated near-goal states of several letters: red pixels at 96 px whenever coverage < 0.95, red
             area against uncovered area, and the green the official camera shows of the same gap.
  radius     visibility of slivers for several radii R on random near-goal states (coverage 0.90-0.95).
  near-miss  teacher-demonstration states (a G pair around 94.6% / 95.1%, near misses of other letters): 96 px
             before/after images and pixel differences.
  cost       rendering time per frame with and without the overlay.
  loader     bc_train with --goal-overlay on a tiny copy of the data (overlay_frames.py demos/human, record tags).
  e2e        base_rollouts.py eval --device cpu on one scene with an overlay-tagged copy of a student checkpoint
             (--student; the copy is written to --out), checking that its cameras show the overlay.

  CUDA_VISIBLE_DEVICES= .venv/bin/python -B scripts/learning/overlay_check.py --out OUT [--radius 9]
      [--student $W/runs/dagger26m/round50/train/ema-step54000.pt] [--skip e2e ...]
Writes OUT/overlay_check.json and PNG images; exits non-zero if a pass/fail check fails.
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tdyssey.envs.official import goal_overlay_region, make_official_shape_env_class, uncovered_goal  # noqa: E402
from tdyssey.envs import official as official_env  # noqa: E402
from tdyssey.official import load_official_source, physical_metrics  # noqa: E402

W = Path(os.environ.get("W", "/tmp/Tdyssey"))
DEMOS = sorted(glob.glob(str(W/"runs/teacher-plan-demos*")))
HUMAN = W/"data/pusht-official/pusht/pusht_cchi_v7_replay.zarr"
CLS = make_official_shape_env_class(load_official_source(ROOT/"third_party/diffusion_policy"))
RESULTS, FAILED = {}, []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)
    return ok


def u8(obs):
    return (np.moveaxis(obs["image"], 0, -1) * 255).round().astype(np.uint8)


def red_px(img):
    """Pixels clearly tinted red (R exceeds both G and B by >= 100); the official image has none."""
    return int(((img[..., 0].astype(int) - np.maximum(img[..., 1], img[..., 2])) >= 100).sum())


def solid_red_px(img):
    """Pixels almost fully red (R exceeds G and B by >= 180): the band covers the camera pixel's whole sample."""
    return int(((img[..., 0].astype(int) - np.maximum(img[..., 1], img[..., 2])) >= 180).sum())


def green_px(img):
    """Pixels visibly tinted goal-green (G exceeds R and B by >= 40): the uncovered goal the official camera shows."""
    return int(((img[..., 1].astype(int) - np.maximum(img[..., 0], img[..., 2])) >= 40).sum())


def env_pair(shape, radius):
    off, on = CLS(shape=shape, legacy=True), CLS(shape=shape, legacy=True, goal_overlay=radius)
    for e in (off, on):
        e.seed(0); e.reset()
    return off, on


def place(env, x, y, angle, px, py):
    """Put the block's local origin at (x, y) with `angle` and the pusher at (px, py), at rest."""
    env.block.angle = angle; env.block.position = (x, y)
    env.agent.position = (px, py)
    env.block.velocity, env.block.angular_velocity, env.agent.velocity = (0.0, 0.0), 0.0, (0.0, 0.0)
    env.space.reindex_shapes_for_body(env.block); env.space.reindex_shapes_for_body(env.agent)


def goal_origin(env):
    return float(env.goal_pose[0]), float(env.goal_pose[1]), float(env.goal_pose[2])


def render_both(off, on, x, y, a, px=60.0, py=60.0):
    for e in (off, on):
        place(e, x, y, a, px, py)
    return u8(off._get_obs()), u8(on._get_obs()), physical_metrics(off)["coverage"]


# ------------------------------------------------------------------------------------------------- identity
def identity(a):
    sys.path.insert(0, str(ROOT/"scripts/learning/image_wm_data"))
    import replay_frames
    out = {}
    files = sorted(f for d in DEMOS for f in glob.glob(f"{d}/*.npz"))[:: max(1, len(DEMOS))][:3]
    for f in files:
        stored = Path(f).parent/"frames"/f"{Path(f).stem}_frames.npz"
        with np.load(f) as z:
            planner = "source" in z.files and "planner" in str(z["source"])
            mask = z["mask"][:4]
        eps, frames = replay_frames.replay((f, [0, 1, 2, 3], 96, planner))
        ref = np.load(stored)["frames"][:4]
        valid = mask[:, :, None, None, None]
        same = bool(np.array_equal(frames * valid, ref * valid))
        out[Path(f).name] = {"frames": int(mask.sum()), "identical": same}
        check(f"identity: overlay-off replay == stored frames/ ({Path(f).name}, {int(mask.sum())} frames)", same)
    # stepped episodes: this tree vs the source of --baseline-rev (both overlay off), same seeds and actions
    base = Path(a.out)/"baseline_src"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", a.baseline_rev, "tdyssey", "manifests"], capture_output=True,
                         check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(base)], input=tar, check=True)
    (base/"third_party").mkdir(exist_ok=True)
    os.symlink((ROOT/"third_party/diffusion_policy").resolve(), base/"third_party/diffusion_policy")
    shots = {}
    for name, root in (("baseline", base), ("this", ROOT)):
        target = Path(a.out)/f"stepped_{name}.npz"
        subprocess.run([sys.executable, "-B", "-c", STEPPED, str(root), str(target)], check=True,
                       env={**os.environ, "REPO": str(root), "CUDA_VISIBLE_DEVICES": ""})
        shots[name] = np.load(target)["frames"]
    same = shots["baseline"].shape == shots["this"].shape and bool(np.array_equal(shots["baseline"], shots["this"]))
    out["stepped"] = {"frames": int(len(shots["this"])), "identical": same, "baseline_rev": a.baseline_rev}
    check(f"identity: overlay-off stepped episodes == {a.baseline_rev} ({len(shots['this'])} frames, 6 letters)", same)
    # overlay on, block exactly on the goal (nothing uncovered): the official image
    eq = []
    for shape in ("T", "G", "E", "O"):
        off, on = env_pair(shape, a.radius)
        x, y, ang = goal_origin(off)
        i_off, i_on, cov = render_both(off, on, x, y, ang)
        eq.append(bool(np.array_equal(i_off, i_on)) and cov > 0.999)
    out["on_goal_identical"] = eq
    check("identity: overlay on with the block on the goal == official image (T, G, E, O)", all(eq))
    RESULTS["identity"] = out


STEPPED = r"""
import os, sys
from pathlib import Path
root, target = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root)); os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
from tdyssey.envs.official import make_official_shape_env_class
from tdyssey.official import load_official_source
cls = make_official_shape_env_class(load_official_source(root / "third_party/diffusion_policy"))
rng = np.random.default_rng(5); frames = []
for shape in ("T", "G", "E", "O", "L", "X"):
    env = cls(shape=shape, legacy=True); env.seed(int(rng.integers(1e6))); obs = env.reset()
    frames.append((np.moveaxis(obs["image"], 0, -1) * 255).round().astype(np.uint8))
    for t in range(60):
        goal = np.array(env.block.position) + rng.normal(0, 40, 2)          # pushes towards the block: contacts happen
        obs, _, done, _ = env.step(np.clip(goal, 0, 512))
        frames.append((np.moveaxis(obs["image"], 0, -1) * 255).round().astype(np.uint8))
np.savez_compressed(target, frames=np.stack(frames))
"""


# ------------------------------------------------------------------------------------------------- geometry
def geometry(a):
    from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
    rng = np.random.default_rng(1); worst = worst_cov = 0.0
    for shape in ("T", "G", "E", "O", "L", "X", "K"):
        env = CLS(shape=shape, legacy=True); env.seed(0); env.reset()
        x, y, ang = goal_origin(env)
        goal_area = pymunk_to_shapely(env._get_goal_pose_body(env.goal_pose), env.block.shapes).area
        for _ in range(20):
            place(env, x + rng.normal(0, 15), y + rng.normal(0, 15), ang + rng.normal(0, 0.2), 60, 60)
            ref, fast = uncovered_goal(env), official_env._uncovered_goal_fast(env)
            worst = max(worst, ref.symmetric_difference(fast).area)
            worst_cov = max(worst_cov, abs(fast.area / goal_area - (1 - physical_metrics(env)["coverage"])))
    RESULTS["geometry"] = {"max_symmetric_difference_px2": worst, "max_uncovered_share_vs_coverage": worst_cov}
    check("geometry: fast uncovered region == coverage metric's region", worst < 1e-6 and worst_cov < 1e-9,
          f"(max {worst:.2e} px^2; uncovered share vs 1 - coverage {worst_cov:.1e})")


# ------------------------------------------------------------------------------------------------- response
def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def response(a):
    rows, missing = [], []
    for shape in ("T", "G", "E", "O", "L", "X", "S", "K"):
        off, on = env_pair(shape, a.radius)
        x, y, ang = goal_origin(off)
        for direction in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            dx, dy = math.cos(direction), math.sin(direction)
            for w in (0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24):
                i_off, i_on, cov = render_both(off, on, x + w * dx, y + w * dy, ang)
                rows.append({"shape": shape, "dir": round(float(direction), 3), "shift_px": w, "coverage": cov,
                             "uncovered_px2": float(uncovered_goal(on).area), "red_px": red_px(i_on),
                             "solid_red_px": solid_red_px(i_on),
                             "green_px_official": green_px(i_off)})
                if cov < 0.95 and rows[-1]["red_px"] == 0:
                    missing.append(rows[-1])
    below = [r for r in rows if r["coverage"] < 0.95]
    check(f"response: red pixels in every 96 px frame with coverage < 0.95 ({len(below)} states)", not missing,
          f"({len(missing)} without red)")
    rho = spearman([r["uncovered_px2"] for r in rows], [r["red_px"] for r in rows])
    band = [r for r in rows if 0.90 <= r["coverage"] < 0.95]
    rho_band = spearman([r["uncovered_px2"] for r in band], [r["red_px"] for r in band]) if len(band) > 5 else None
    # per letter and direction (one sliver geometry growing), red area against the shift while slivers are thinner
    # than R; pooled correlations mix letters with different edge lengths and the switch at width R (see GOAL_OVERLAY.md)
    drops, per_seq = 0, []
    for shape in sorted({r["shape"] for r in rows}):
        for d in sorted({r["dir"] for r in rows}):
            seq = [r for r in rows if r["shape"] == shape and r["dir"] == d and r["shift_px"] <= a.radius]
            per_seq.append(spearman([r["shift_px"] for r in seq], [r["red_px"] for r in seq]))
            drops += sum(1 for p, q in zip(seq, seq[1:]) if q["red_px"] < 0.85 * p["red_px"])
    near = [r for r in rows if 0.90 <= r["coverage"] < 0.95]
    summary = {"states": len(rows), "below_095": len(below), "without_red": len(missing),
               "spearman_red_vs_uncovered_all": rho, "spearman_red_vs_uncovered_090_095": rho_band,
               "drops_over_15pct_in_thin_regime": drops, "steps_in_thin_regime": len(per_seq) * (sum(1 for w in (0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8) if w <= a.radius) - 1),
               "median_spearman_red_vs_shift_thin_regime": float(np.median(per_seq)),
               "share_sequences_spearman_ge_0.8": float(np.mean(np.array(per_seq) >= 0.8)),
               "median_red_px_090_095": float(np.median([r["red_px"] for r in near])) if near else None,
               "median_green_px_official_090_095": float(np.median([r["green_px_official"] for r in near])) if near else None,
               "share_official_green_le_2px_090_095": float(np.mean([r["green_px_official"] <= 2 for r in near])) if near else None}
    RESULTS["response"] = {"summary": summary, "rows": rows}
    check("response: red area rises with the sliver while it is thinner than R (median per-sequence Spearman >= 0.8)",
          summary["median_spearman_red_vs_shift_thin_regime"] >= 0.8,
          f"(median {summary['median_spearman_red_vs_shift_thin_regime']:.2f}; {drops} drops > 15% in "
          f"{summary['steps_in_thin_regime']} steps; pooled rho all {rho:.2f}, 0.90-0.95 {rho_band:.2f})")
    print(f"  0.90-0.95: median red px {summary['median_red_px_090_095']}, official green px "
          f"{summary['median_green_px_official_090_095']} (<= 2 green px in {summary['share_official_green_le_2px_090_095']:.2f})",
          flush=True)


# ------------------------------------------------------------------------------------------------- radius
def radius_sweep(a):
    rng = np.random.default_rng(7); out = {}
    sets = {"0.90-0.95": (0.90, 0.95, 6, 3), "0.95-0.99": (0.95, 0.99, 2, 1)}   # coverage range, max shift px, max turn deg
    configs = {k: [] for k in sets}
    for shape in ("T", "G", "E", "O", "L", "X", "S", "K", "A", "M"):
        env = CLS(shape=shape, legacy=True); env.seed(0); env.reset()
        x, y, ang = goal_origin(env)
        for name, (lo, hi, dmax, tmax) in sets.items():
            while sum(c[0] == shape for c in configs[name]) < 25:
                pose = (x + rng.uniform(-dmax, dmax), y + rng.uniform(-dmax, dmax), ang + math.radians(rng.uniform(-tmax, tmax)))
                place(env, *pose, 60, 60)
                if lo <= physical_metrics(env)["coverage"] < hi:
                    configs[name].append((shape, pose))
    for radius in (5, 7, 9, 11, 13):
        out[radius] = {}
        envs = {}
        for name, cfgs in configs.items():
            counts, solid, greens = [], [], []
            for shape, pose in cfgs:
                if shape not in envs:
                    envs[shape] = env_pair(shape, radius)
                off, on = envs[shape]
                i_off, i_on, _ = render_both(off, on, *pose)
                counts.append(red_px(i_on)); solid.append(solid_red_px(i_on)); greens.append(green_px(i_off))
            r = out[radius][name] = {
                "states": len(counts), "min_red_px": int(min(counts)), "median_red_px": float(np.median(counts)),
                "min_solid_red_px": int(min(solid)), "median_solid_red_px": float(np.median(solid)),
                "share_ge_4_solid_red_px": float(np.mean(np.array(solid) >= 4)),
                "median_official_green_px": float(np.median(greens)), "share_official_green_le_2px": float(np.mean(np.array(greens) <= 2))}
            print(f"  R={radius:2d} {name}: red min {r['min_red_px']} median {r['median_red_px']:.0f}; solid red min "
                  f"{r['min_solid_red_px']} median {r['median_solid_red_px']:.0f}, >=4 solid {r['share_ge_4_solid_red_px']:.2f}; "
                  f"official green median {r['median_official_green_px']:.0f}, <=2 green px {r['share_official_green_le_2px']:.2f}", flush=True)
    RESULTS["radius"] = out
    rows = []                                                          # the same near-goal states for each radius
    for shape, pose in [c for c in configs["0.90-0.95"] if c[0] in ("G", "L", "S")][::9][:3]:
        off = CLS(shape=shape, legacy=True); off.seed(0); off.reset(); place(off, *pose, 60, 60)
        ims = [u8(off._get_obs())]
        for radius in (5, 7, 9, 11, 13):
            on = CLS(shape=shape, legacy=True, goal_overlay=radius); on.seed(0); on.reset(); place(on, *pose, 60, 60)
            ims.append(u8(on._get_obs()))
        rows.append(np.hstack([np.pad(cv2.resize(im[24:72, 24:72], (192, 192), interpolation=cv2.INTER_NEAREST),
                                      ((0, 4), (0, 4), (0, 0)), constant_values=255) for im in ims]))
    cv2.imwrite(str(Path(a.out)/"radius_compare.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print(f"  wrote {Path(a.out)/'radius_compare.png'} (close-ups: official, R = 5, 7, 9, 11, 13)", flush=True)
    check(f"radius: R={a.radius} shows >= 4 solid red px in every random state of both sets",
          all(out[a.radius][k]["share_ge_4_solid_red_px"] == 1.0 for k in sets))


# ------------------------------------------------------------------------------------------------- near-miss
def demo_state(f, i, t):
    with np.load(f) as z:
        s = z["state"][i, t]
        return str(z["shape"][i]), float(z["coverage"][i, t]), (float(s[0]), float(s[1]), float(s[2]), float(s[6]), float(s[7]))


def near_miss(a):
    picks = []
    best = None                                                        # a G pair: last state below 95% and the success
    for f in sorted(glob.glob(str(W/"runs/teacher-plan-demos26/G-*.npz"))):
        with np.load(f) as z:
            for i in range(len(z["seed"])):
                n = int(z["mask"][i].sum()); cov = z["coverage"][i, :n]
                if not z["success"][i] or n < 3 or not 0.94 <= cov[-2] < 0.95:
                    continue
                score = abs(cov[-2] - 0.946) + abs(cov[-1] - 0.951)
                if best is None or score < best[0]:
                    best = (score, f, i, n)
    if best:
        _, f, i, n = best
        picks += [("G pair, before success", f, i, n - 2), ("G pair, success", f, i, n - 1)]
    fails = []                                                         # teacher near misses (never above 95%)
    for d in DEMOS:
        for f in sorted(glob.glob(f"{d}/*.npz")):
            with np.load(f) as z:
                for i in range(len(z["seed"])):
                    n = int(z["mask"][i].sum()); cov = z["coverage"][i, :n]
                    if not z["success"][i] and 0.90 <= cov.max() < 0.95 and str(z["shape"][i]) not in {s for s, *_ in fails}:
                        fails.append((str(z["shape"][i]), f, i, int(cov.argmax())))
    for shape, f, i, t in fails[:6]:
        picks.append((f"{shape} near miss (peak)", f, i, t))
    tiles, stats = [], []
    envs = {}
    for label, f, i, t in picks:
        shape, cov_rec, (x, y, ang, px, py) = demo_state(f, i, t)
        if shape not in envs:
            envs[shape] = env_pair(shape, a.radius)
        off, on = envs[shape]
        i_off, i_on, cov = render_both(off, on, x, y, ang, px, py)
        stats.append({"label": label, "file": Path(f).name, "episode": i, "control": t, "coverage": cov,
                      "coverage_recorded": cov_rec, "red_px": red_px(i_on), "green_px_official": green_px(i_off)})
        tiles.append((f"{label}: {100 * cov:.1f}%", i_off, i_on))
    pair = [s for s in stats if s["label"].startswith("G pair")]
    if len(pair) == 2:
        a0, a1 = tiles[0], tiles[1]
        diff_before = int((np.abs(a0[1].astype(int) - a1[1]).max(-1) > 30).sum())
        diff_after = int((np.abs(a0[2].astype(int) - a1[2]).max(-1) > 30).sum())
        RESULTS["near_miss_pair"] = {"coverage": [pair[0]["coverage"], pair[1]["coverage"]],
                                     "pixels_differing_gt30_before": diff_before, "pixels_differing_gt30_after": diff_after,
                                     "red_px": [pair[0]["red_px"], pair[1]["red_px"]],
                                     "green_px_official": [pair[0]["green_px_official"], pair[1]["green_px_official"]]}
        print(f"  G pair {100 * pair[0]['coverage']:.1f}% vs {100 * pair[1]['coverage']:.1f}%: pixels differing (>30) "
              f"official {diff_before}, overlay {diff_after}; red px {pair[0]['red_px']} / {pair[1]['red_px']}", flush=True)
    RESULTS["near_miss"] = stats
    check("near-miss: recorded coverage reproduced by the placed states", all(abs(s["coverage"] - s["coverage_recorded"]) < 1e-4 for s in stats))
    check("near-miss: every state below 95% shows red", all(s["red_px"] > 0 for s in stats if s["coverage"] < 0.95))
    save_grid(tiles, Path(a.out)/"near_miss_before_after.png", Path(a.out)/"near_miss_96px_raw.png")


def save_grid(tiles, path, raw_path, scale=4):
    """Rows: official 96 px frame, overlay 96 px frame (nearest-neighbour x4), and x8 close-ups of the goal area."""
    rows, raw = [], []
    for label, i_off, i_on in tiles:
        big = [cv2.resize(im, (96 * scale, 96 * scale), interpolation=cv2.INTER_NEAREST) for im in (i_off, i_on)]
        zoom = [cv2.resize(im[24:72, 24:72], (96 * scale, 96 * scale), interpolation=cv2.INTER_NEAREST) for im in (i_off, i_on)]
        for z in zoom:                                                 # grid: one 96 px camera pixel per square
            z[:, ::scale * 2] = (z[:, ::scale * 2] * 0.85).astype(np.uint8); z[::scale * 2, :] = (z[::scale * 2, :] * 0.85).astype(np.uint8)
        bar = np.full((28, 4 * 96 * scale + 3 * 6, 3), 255, np.uint8)
        cv2.putText(bar, f"{label}   (official | overlay | official close-up | overlay close-up)", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        gap = np.full((96 * scale, 6, 3), 255, np.uint8)
        rows.append(np.vstack([bar, np.hstack([big[0], gap, big[1], gap, zoom[0], gap, zoom[1]])]))
        raw.append(np.hstack([i_off, np.full((96, 2, 3), 255, np.uint8), i_on]))
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(raw_path), cv2.cvtColor(np.vstack([np.vstack([r, np.full((2, r.shape[1], 3), 255, np.uint8)]) for r in raw]), cv2.COLOR_RGB2BGR))
    print(f"  wrote {path} and {raw_path}", flush=True)


# ------------------------------------------------------------------------------------------------- cost
def cost(a):
    out = {}
    for shape in ("T", "G", "E", "X"):
        off, on = env_pair(shape, a.radius)
        x, y, ang = goal_origin(off)
        for where, pose in (("far", (130.0, 130.0, 0.3)), ("near", (x + 2.0, y - 1.0, ang + 0.01))):
            for e in (off, on):
                place(e, *pose, 60, 60)
            times = {}
            for name, e in (("off", off), ("on", on)):
                e._get_obs(); t = time.perf_counter()
                for _ in range(60):
                    e._get_obs()
                times[name] = (time.perf_counter() - t) / 60 * 1e3
            out[f"{shape}-{where}"] = times
    mean_off = float(np.mean([v["off"] for v in out.values()])); mean_on = float(np.mean([v["on"] for v in out.values()]))
    RESULTS["cost_ms_per_frame"] = {"per_state": out, "mean_off": mean_off, "mean_on": mean_on}
    print(f"  render ms/frame: official {mean_off:.2f}, overlay {mean_on:.2f} (+{mean_on - mean_off:.2f})", flush=True)


# ------------------------------------------------------------------------------------------------- loader
def loader(a):
    import zarr
    sys.path.insert(0, str(ROOT/"scripts/learning"))
    import bc_train
    from types import SimpleNamespace
    tmp = Path(a.out)/"loader"
    if tmp.exists():
        shutil.rmtree(tmp)
    demo_dir = tmp/"demos"; (demo_dir/"frames").mkdir(parents=True)
    src = sorted(glob.glob(str(W/"runs/teacher-plan-demos26/G-*.npz")))[0]
    with np.load(src) as z:                                            # first 3 episodes of one demonstration file
        np.savez_compressed(demo_dir/"G-test.npz", **{k: (z[k][:3] if z[k].ndim and len(z[k]) == len(z["seed"]) else z[k]) for k in z.files})
    with np.load(Path(src).parent/"frames"/f"{Path(src).stem}_frames.npz") as z:
        np.savez_compressed(demo_dir/"frames"/"G-test_frames.npz", frames=z["frames"][:3])
    g = zarr.open(str(HUMAN), "r"); end = int(g["meta"]["episode_ends"][1])   # first two human T episodes
    root = zarr.open(str(tmp/"human"/"pusht_test.zarr"), "w")
    for k in ("img", "state", "action"):
        root.create_dataset(f"data/{k}", data=g["data"][k][:end])
    root.create_dataset("meta/episode_ends", data=g["meta"]["episode_ends"][:2])
    py = [sys.executable, "-B", str(ROOT/"scripts/learning/overlay_frames.py")]
    subprocess.run(py + ["demos", "--goal-overlay", str(a.radius), "--workers", "2", str(demo_dir)], check=True)
    subprocess.run(py + ["human", "--goal-overlay", str(a.radius), "--workers", "2", str(tmp/"human"/"pusht_test.zarr")], check=True)
    args = lambda ov: SimpleNamespace(demos=[demo_dir], human_t=tmp/"human"/"pusht_test.zarr", records=[], image_size=96,
                                      goal_overlay=ov, data_fraction=1.0, label="plan", near_coverage=0.8)
    eps0, eps1 = list(bc_train.iter_episodes(args(0))), list(bc_train.iter_episodes(args(a.radius)))
    same_layout = len(eps0) == len(eps1) and all(e0[0].shape == e1[0].shape and np.array_equal(e0[2], e1[2]) for e0, e1 in zip(eps0, eps1))
    teacher_red = [red_px(fr) for e in eps1 if e[3].endswith("teacher") for fr in e[0]]
    human_red = [red_px(fr) for e in eps1 if e[3] == "T-human" for fr in e[0]]
    check("loader: overlay data has the same samples and labels as the official data", same_layout)
    check("loader: overlay teacher and human T frames show red where the goal is uncovered",
          np.mean(np.array(teacher_red) > 0) > 0.9 and np.mean(np.array(human_red) > 0) > 0.9,
          f"(teacher {np.mean(np.array(teacher_red) > 0):.3f}, human {np.mean(np.array(human_red) > 0):.3f} of frames)")
    # bc_train's streamed parallel build gives the same arrays as the serial one (overlay on)
    ar = args(a.radius)
    n = bc_train.count_frames(ar)
    mm = np.lib.format.open_memmap(tmp/"frames.npy", mode="w+", dtype=np.uint8, shape=(n, 96, 96, 3))
    unnorm = lambda x: x
    par = bc_train.build_streamed_parallel(ar, unnorm, mm, 2)
    ser = bc_train.build_index(list(bc_train.iter_episodes(ar)), [], ar.near_coverage)
    same_build = all(np.array_equal(np.asarray(p), np.asarray(s)) for p, s in zip(par, ser))
    check("loader: streamed parallel build == serial build with the overlay", same_build)
    # record tags: overlay records accepted with the overlay, refused otherwise unless --mixed-records
    rec_on, rec_off = tmp/"rec-on", tmp/"rec-off"
    rec_on.mkdir(); rec_off.mkdir()
    np.savez_compressed(rec_on/"G-1.npz", goal_overlay=a.radius, choice=np.array([8]))
    np.savez_compressed(rec_off/"G-1.npz", choice=np.array([8]))
    ok = bc_train.check_record_overlay([rec_on], a.radius, False) == {a.radius: 1}
    refused = []
    for dirs, ov in (([rec_off], a.radius), ([rec_on], 0)):
        try:
            bc_train.check_record_overlay(dirs, ov, False); refused.append(False)
        except SystemExit:
            refused.append(True)
    mixed = bc_train.check_record_overlay([rec_on, rec_off], a.radius, True) == {a.radius: 1, 0: 1}
    check("loader: record overlay tags checked (match kept, mismatch refused, --mixed-records allowed)", ok and all(refused) and mixed)
    RESULTS["loader"] = {"teacher_frames": len(teacher_red), "human_frames": len(human_red), "samples": len(par[5])}


def human_rerender(a):
    """How far the project renderer (overlay off) is from the dataset's human T images (why overlay runs re-render)."""
    import zarr
    sys.path.insert(0, str(ROOT/"scripts/learning"))
    import overlay_frames
    g = zarr.open(str(HUMAN), "r")
    n = 2000
    _, mine = overlay_frames._render_human((HUMAN, 0, n, 96, 0))
    ref = np.asarray(g["data"]["img"][:n]).round().clip(0, 255).astype(np.uint8)
    d = np.abs(mine.astype(int) - ref.astype(int)).max(-1)
    RESULTS["human_t_rerender_vs_dataset"] = {"frames": n, "identical_frames": int((d.max(axis=(1, 2)) == 0).sum()),
                                              "mean_pixels_differing": float((d > 0).sum(axis=(1, 2)).mean()),
                                              "mean_pixels_differing_gt30": float((d > 30).sum(axis=(1, 2)).mean())}
    print(f"  human T re-render vs dataset images: {RESULTS['human_t_rerender_vs_dataset']}", flush=True)


# ------------------------------------------------------------------------------------------------- e2e
def e2e(a):
    import torch
    ck = torch.load(a.student, map_location="cpu", weights_only=True)
    ck["arch"] = dict(ck.get("arch") or {"image_size": 96, "down_dims": None, "scratch_model": False}, goal_overlay=a.radius)
    tagged = Path(a.out)/"student_overlay_tagged.pt"
    torch.save(ck, tagged); del ck
    sys.path.insert(0, str(ROOT/"scripts/learning"))
    import base_rollouts
    agent, cls = base_rollouts.build(str(tagged), device="cpu")
    check("e2e: build() reads goal_overlay from the checkpoint", getattr(agent, "goal_overlay", 0) == a.radius)
    seen = []

    class Spy(cls):                                                     # records what the policy's camera shows
        def _get_obs(self):
            obs = super()._get_obs()
            seen.append((self.goal_overlay, red_px(u8(obs))))
            return obs
    from collections import defaultdict
    from types import SimpleNamespace
    from tdyssey.policy_training import evaluate_batched
    torch.set_num_threads(a.threads)
    cfg = SimpleNamespace(eval_shapes=("T", "G"), eval_seeds=[400000], eval_steps=a.e2e_steps, seed=42, execute=8,
                          render_size=96, goal_overlay=agent.goal_overlay)
    t = time.time()
    episodes = evaluate_batched(agent, Spy, cfg, defaultdict(int), lambda: None)
    secs = time.time() - t
    check("e2e: evaluation cameras rendered with the overlay", all(ov == a.radius for ov, _ in seen) and any(r > 0 for _, r in seen),
          f"({len(seen)} frames, {sum(r > 0 for _, r in seen)} with red)")
    RESULTS["e2e"] = {"episodes": [{k: e[k] for k in ("shape", "success", "controls", "peak_coverage", "final_coverage")} for e in episodes],
                      "seconds": secs, "controls_per_episode_cap": a.e2e_steps, "threads": a.threads}
    print(f"  e2e in-process: {RESULTS['e2e']['episodes']} in {secs:.0f}s", flush=True)
    # the command line itself (base_rollouts.py eval --device cpu), one scene, T only, few controls
    out = Path(a.out)/"e2e_eval.json"
    cmd = [sys.executable, "-B", str(ROOT/"scripts/learning/base_rollouts.py"), "eval", "--device", "cpu", "--student", str(tagged),
           "--scenes", "1", "--seed-base", "400000", "--shapes", "T", "G", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(a.threads)})
    ok = r.returncode == 0 and out.exists() and json.loads(out.read_text()).get("goal_overlay") == a.radius
    check("e2e: base_rollouts.py eval --device cpu runs and records goal_overlay", ok, r.stderr[-400:] if not ok else "")
    if ok:
        RESULTS["e2e"]["cli"] = {k: json.loads(out.read_text())[k] for k in ("summary", "seconds", "goal_overlay")}
    tagged.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--radius", type=int, default=9)
    p.add_argument("--baseline-rev", default="origin/feat/last-mile-rl")
    p.add_argument("--student", type=Path, default=W/"runs/dagger26m/round50/train/ema-step54000.pt")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--e2e-steps", type=int, default=300)
    p.add_argument("--skip", nargs="*", default=[])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    steps = {"identity": identity, "geometry": geometry, "response": response, "radius": radius_sweep, "near-miss": near_miss,
             "cost": cost, "human": human_rerender, "loader": loader, "e2e": e2e}
    for name, fn in steps.items():
        if name in a.skip:
            continue
        print(f"== {name}", flush=True)
        fn(a)
    RESULTS["failed"] = FAILED
    (a.out/"overlay_check.json").write_text(json.dumps(RESULTS, indent=1, default=float))
    print(f"wrote {a.out/'overlay_check.json'}; {len(FAILED)} failed: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
