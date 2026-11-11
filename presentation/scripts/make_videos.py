"""Record example episodes of every model (official PushT simulator, standard 100-step DDPM sampling) and turn them into
the presentation videos and still images. Needs a CUDA GPU, the repository environment and the checkpoints
(paths relative to $TDYSSEY_WORK, default /tmp/Tdyssey; looked up in runs/ and slides_data/ckpt/runs/).

    .venv/bin/python -B presentation/scripts/make_videos.py rollouts MODEL [MODEL ...]   # record (cached in $WORK/slides_data/lanes)
    .venv/bin/python -B presentation/scripts/make_videos.py compose                        # write videos and images from the cache

Episodes use scenes from the validation pool (seeds 400000 + 20*i), set up exactly as in the evaluations (matched
starts across shapes) with the evaluation's random stream per scene. They are examples: success rates come from the
tests in presentation/data, not from these clips.
"""
from collections import deque
import io
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]                        # this checkout (outputs go to presentation/)
REPO = Path(os.environ.get("REPO", HERE))                         # checkout with .venv, checkpoints/ and third_party/
WORK = Path(os.environ.get("TDYSSEY_WORK", "/tmp/Tdyssey"))
OUT = HERE / "presentation"
CACHE = WORK / "slides_data" / "lanes"
sys.path.insert(0, str(REPO / "scripts" / "learning"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SHAPES9 = ["T", "T_narrow", "Y", "D", "L", "E", "K", "R", "digit7"]
ALL33 = ["T", "T_narrow", "Y", "D", "L", "E", "F", "J", "P", "U"] + list("ABCGHIKMNOQRSVWXZ") + \
        ["digit3", "digit4", "digit7", "plus", "hash", "arrow"]
HELD = {"digit3", "digit4", "digit7", "plus", "hash", "arrow"}
SEEDS = [400000, 400020]
TILE = 384                                       # stored frame size (px)
NOISE_RNG = np.random.default_rng(0)
SPEED = "1 frame per control at 30 fps = 3x real time (the simulator runs 10 controls per second)"
MODELS = {   # key: (title, checkpoint relative to runs/, None = original official policy, "all33" = record all 33 shapes)
    "original": ("Original policy (trained on T only)", None, True),
    "forward_rl": ("Forward RL, after 150 updates (degraded)", "long-s2/advantage_flow-dense-lr3e-5-seed42/advantage_flow-dense-lr3e-5-seed42/last.pt", False),
    "five_copy": ("5 shapes: copying the teacher, no corrections", "bc/v2-long/ema-step130000.pt", False),
    "five_r4": ("5 shapes: correction round 4", "dagger-loop/round4/ema-step20000.pt", False),
    "five_r7": ("5 shapes: correction round 7", "dagger-loop/round7/ema-step20000.pt", True),
    "blend": ("Mid-project encoder blend (50%)", "blend/big4-unet0-enc0.5.pt", False),
    "ten_exec": ("10 shapes: executed-push labels, round 16", "dagger10/round16/train/ema-step20000.pt", False),
    "ten_96": ("10 shapes: final 96 px model", "dagger10m/round13/train/ema-step20000.pt", True),
    "ten_192": ("10 shapes: 192 px model", "dagger10h/round6/train/ema-step20000.pt", False),
    "l26_r0": ("26 letters: round 0 (retrained on 27 shapes)", "dagger26m/round0/train/ema-step30000.pt", False),
    "l26_r13": ("26 letters: round 13 (before the larger budget)", "dagger26m/round13/train/ema-step20000.pt", False),
    "l26_r44": ("26 letters: round 44 (final-test model)", "dagger26m/round44/train/ema-step54000.pt", False),
    "l26_r50": ("26 letters: round 50 (main model)", "dagger26m/round50/train/ema-step54000.pt", True),
    "flowdit": ("Flow-DiT (505M, experiment)", "flow26b16/v1/ema-step160000.pt", True),
    "l26_192": ("26 letters: 192 px camera", "dagger26h/round2/train/ema-step54000.pt", True),
}


def label(shape):
    if shape == "T_narrow":
        return "near-T"
    if shape in HELD:
        return shape.replace("digit", "digit ") + " (never seen)"
    return shape


def checkpoint(rel):
    for base in (WORK / "runs", WORK / "slides_data" / "ckpt" / "runs"):
        if (base / rel).exists():
            return base / rel
    raise FileNotFoundError(rel)


def load(key):
    import base_rollouts
    rel = MODELS[key][1]
    if rel is None:
        return base_rollouts.build(None)
    path = checkpoint(rel)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") == "official_forward_rl":           # forward-RL trainer state
        from tdyssey.forward_policy import ForwardPolicy
        from tdyssey.forward_protocol import ForwardConfig
        from tdyssey.official import OfficialDiffusionPolicy, load_official_source
        from tdyssey.envs.official import make_official_shape_env_class
        source = REPO / "third_party/diffusion_policy"
        agent = ForwardPolicy(OfficialDiffusionPolicy(source, REPO / "checkpoints/official-dp/ema_policy.pt"),
                              ForwardConfig(**payload["config"]))
        agent.load(payload)
        agent.render_size = 96
        return agent, make_official_shape_env_class(load_official_source(source))
    return base_rollouts.build(str(path))


def frame(env, obs, plan, done_targets, execute, inset=True):
    import pygame
    img = np.ascontiguousarray(np.transpose(pygame.surfarray.array3d(env.screen), (1, 0, 2)))
    if plan is not None:
        for j, (x, y) in enumerate(plan):
            if j < done_targets:
                continue
            cv2.circle(img, (int(x), int(y)), 6 if j < execute else 5, (230, 60, 60) if j < execute else (240, 165, 165), -1, cv2.LINE_AA)
    img = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
    if not inset:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()
    cam = np.ascontiguousarray((np.moveaxis(obs["image"], 0, -1) * 255).clip(0, 255).astype(np.uint8))
    s = TILE * 30 // 100
    cam = cv2.resize(cam, (s, s), interpolation=cv2.INTER_NEAREST)
    img[6:s + 10, 6:s + 10] = 50
    img[8:s + 8, 8:s + 8] = cam
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


@torch.no_grad()
def rollout(agent, cls, shapes, seed, execute=8, noise_seed=42, action_noise=0.0):
    from tdyssey.collection import derived_seed
    from tdyssey.envs.official import matched_reset
    from tdyssey.official import physical_metrics
    size = getattr(agent, "render_size", 96)
    envs = {s: cls(shape=s, legacy=True, render_size=size) for s in shapes}
    observations, _ = matched_reset(envs, seed, {"reset_physics_substeps": 0})
    lanes = [{"shape": s, "env": envs[s], "done": False, "history": deque([observations[s]] * 2, maxlen=2),
              "generator": torch.Generator(device=agent.device).manual_seed(derived_seed(noise_seed, "rl-eval", seed)),
              "frames": [frame(envs[s], observations[s], None, 0, execute)], "coverage": [physical_metrics(envs[s])["coverage"]]}
             for s in shapes]
    for start in range(0, 300, execute):
        active = [l for l in lanes if not l["done"]]
        if not active:
            break
        condition = agent.encode_batch([l["history"] for l in active])
        actions, samples = agent.sample_batch(condition, [l["generator"] for l in active])
        plans = agent.network.normalizer["action"].unnormalize(samples)[:, 1:].cpu().numpy()
        for i, lane in enumerate(active):
            for j, action in enumerate(actions[i][:min(execute, 300 - start)]):
                if action_noise > 0:                      # as in the world-model data: Gaussian noise on each target
                    action = np.clip(action + NOISE_RNG.normal(0, action_noise, 2), 0, 512)
                obs, _, done, _ = lane["env"].step(action)
                lane["history"].append(obs)
                lane["frames"].append(frame(lane["env"], obs, plans[i], j + 1, execute))
                lane["coverage"].append(physical_metrics(lane["env"])["coverage"])
                if done:
                    lane["done"] = True
                    break
    out = {}
    for l in lanes:
        cov = l["coverage"]
        out[l["shape"]] = {"frames": l["frames"], "coverage": cov, "solved": l["done"], "moves": len(cov) - 1,
                           "final": cov[-1], "peak": max(cov), "peak_move": int(np.argmax(cov))}
        l["env"].close()
    return out


def record(keys):
    CACHE.mkdir(parents=True, exist_ok=True)
    for key in keys:
        title, _, all33 = MODELS[key]
        agent, cls = load(key)
        for seed in (SEEDS if all33 else SEEDS[:1]):
            path = CACHE / f"{key}_{seed}.pkl"
            if path.exists():
                continue
            lanes = rollout(agent, cls, ALL33 if all33 else SHAPES9, seed)
            pickle.dump(lanes, open(path, "wb"))
            solved = [s for s, l in lanes.items() if l["solved"]]
            print(f"{key} seed {seed}: solved {len(solved)}/{len(lanes)}: {' '.join(solved)}", flush=True)
        del agent
        torch.cuda.empty_cache()


TEACHER_SHAPES = ["T", "T_narrow", "Y", "D", "L", "E", "K", "R", "W"]                # the 9-shape grid
TRAINED27 = ["T", "T_narrow", "Y", "D", "L", "E", "F", "J", "P", "U"] + list("ABCGHIKMNOQRSVWXZ")
TEACHER_ARGS = ["--select", "best", "--gate-margin", "0", "--pushes", "128", "--horizon", "48", "--knots", "8",
                "--approach", "0", "--kp", "area", "--tight", "2", "--state-noise", "1", "--angle-noise", "1"]
LETTERS27 = "T,T_narrow,Y,D,L,E,F,J,P,U,A,B,C,G,H,I,K,M,N,O,Q,R,S,V,W,X,Z"


def teacher(shapes):
    """Run the planner teacher (world model v3) on one validation scene per shape, then replay each episode in the
    simulator to draw it with the teacher's intended 16-step plan (it re-plans after every push)."""
    import subprocess
    from collections import defaultdict
    from tdyssey.envs.official import make_official_shape_env_class, matched_reset
    from tdyssey.official import load_official_source, physical_metrics, physical_state
    out = WORK / "slides_data" / "teacher"; out.mkdir(parents=True, exist_ok=True)
    env_vars = dict(os.environ, TDYSSEY_SHAPES=LETTERS27)
    for s in shapes:
        if not (out / f"{s}.npz").exists():
            subprocess.run([sys.executable, "-B", str(REPO / "scripts/learning/mpc_teacher.py"), "--wm", str(WORK / "runs/wm2/v3/model.pt"),
                            *TEACHER_ARGS, "--shape", s, "--scenes", "1", "--seed-base", str(SEEDS[0]),
                            "--save-traj", str(out / f"{s}.npz"), "--out", str(out / f"{s}.json")], env=env_vars, cwd=REPO, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cls = make_official_shape_env_class(load_official_source(REPO / "third_party" / "diffusion_policy"))
    lanes = {}
    for s in shapes:
        z = np.load(out / f"{s}.npz")
        partner = "L" if s == "T" else s
        envs = {"T": cls(shape="T", legacy=True), partner: cls(shape=partner, legacy=True)}
        matched_reset(envs, int(z["seed"][0]), defaultdict(int))
        env, n = envs[s], int(z["mask"][0].sum())
        obs = env._get_obs()
        frames, cov = [frame(env, obs, z["plan"][0, 0], 0, 1, inset=False)], [physical_metrics(env)["coverage"]]
        for k in range(n - 1):
            st = physical_state(env); o, pu = st["object"], st["pusher"]
            err = np.abs(np.array([*o["position"], o["angle"], *pu["position"]]) - z["state"][0, k, [0, 1, 2, 6, 7]]).max()
            if err > 1e-3:
                raise RuntimeError(f"teacher replay of {s} differs from its record by {err}")
            obs, _, _, _ = env.step(z["action"][0, k])
            nxt = z["plan"][0, k + 1] if k + 1 < n - 1 else None
            frames.append(frame(env, obs, nxt, 0, 1, inset=False)); cov.append(physical_metrics(env)["coverage"])
        lanes[s] = {"frames": frames, "coverage": cov, "solved": bool(z["success"][0]), "moves": len(cov) - 1,
                    "final": cov[-1], "peak": max(cov), "peak_move": int(np.argmax(cov))}
        for e in envs.values():
            e.close()
        print(f"teacher {s}: {'solved' if lanes[s]['solved'] else 'not solved'} in {lanes[s]['moves']} moves", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(lanes, open(CACHE / f"teacher_{SEEDS[0]}.pkl", "wb"))


def worldmodel(shapes=("T", "D", "K", "R", "E", "A"), horizon=32, windows=3):
    """What the world model imagines vs what really happens: replay a recorded teacher episode in the simulator and, at
    moments when the pusher is about to touch the block, roll the model forward open-loop over the next `horizon`
    controls with the actions that were really executed. The predicted block outline (red, dashed) is drawn over the
    real block; the caption gives the centre-of-mass error. Needs the world model (runs/wm2/v3) and teacher records."""
    import math
    from collections import defaultdict
    os.environ["TDYSSEY_SHAPES"] = LETTERS27
    import wm2
    import teacher2
    from tdyssey.envs.official import make_official_shape_env_class, matched_reset
    from tdyssey.official import load_official_source
    import pygame
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geo = teacher2.Geometry(dev)
    blob = torch.load(WORK / "runs/wm2/v3/model.pt", map_location=dev, weights_only=True)
    wm2.check_shapes(blob)
    model = wm2.WM2(geo.rects, blob["width"], blob["depth"]).to(dev); model.load_state_dict(blob["state_dict"]); model.eval()
    cls = make_official_shape_env_class(load_official_source(REPO / "third_party" / "diffusion_policy"))
    V = OUT / "videos" / "world_model"; I = OUT / "images" / "world_model"
    V.mkdir(parents=True, exist_ok=True); I.mkdir(parents=True, exist_ok=True)
    summary = []
    for s in shapes:
        z = np.load(WORK / "slides_data" / "teacher" / f"{s}.npz")
        n = int(z["mask"][0].sum())
        k_sh = wm2.SHAPES.index(s)
        sh = torch.tensor([k_sh], device=dev)
        st, pv, act, con = z["state"][0], z["pusher_velocity"][0], z["action"][0], z["contacts"][0]
        starts, t0 = [], 0                                     # window starts: a few controls before contact begins
        while t0 < n - horizon - 1 and len(starts) < windows:
            if con[t0 + 1:t0 + 9].max() > 0 and con[max(0, t0 - 4):t0 + 1].max() == 0:
                starts.append(t0); t0 += horizon
            else:
                t0 += 1
        partner = "L" if s == "T" else s
        frames_out, stills_out = [], []
        for t0 in starts:
            envs = {"T": cls(shape="T", legacy=True), partner: cls(shape=partner, legacy=True)}
            matched_reset(envs, int(z["seed"][0]), defaultdict(int))
            env = envs[s]
            for k in range(t0):
                env.step(act[k])
            raw = torch.tensor([[*st[t0, :6], *st[t0, 6:8], *pv[t0]]], device=dev, dtype=torch.float32)
            s0 = wm2.to_cog(raw, sh, geo.cogs)
            with torch.no_grad():
                pred = model.rollout(s0, torch.tensor(act[None, t0:t0 + horizon], device=dev, dtype=torch.float32), sh, hard=True)[0].cpu().numpy()
            cog = geo.cogs[k_sh].cpu().numpy()
            local = [[(float(v.x), float(v.y)) for v in shp.get_vertices()] for shp in env.block.shapes]
            for k in range(horizon + 1):
                if k > 0:
                    env.step(act[t0 + k - 1])
                env._get_obs()
                img = np.ascontiguousarray(np.transpose(pygame.surfarray.array3d(env.screen), (1, 0, 2)))
                real = wm2.to_cog(torch.tensor([[*st[t0 + k, :6], 0, 0, 0, 0]], device=dev, dtype=torch.float32), sh, geo.cogs)[0].cpu().numpy()
                if k > 0:
                    px, py, th = pred[k - 1, 0], pred[k - 1, 1], pred[k - 1, 2]
                    ox, oy = cog[0] * math.cos(th) - cog[1] * math.sin(th), cog[0] * math.sin(th) + cog[1] * math.cos(th)
                    bx, by = px - ox, py - oy                             # body origin from the predicted centre of mass
                    for poly in local:
                        pts = np.array([[bx + x * math.cos(th) - y * math.sin(th), by + x * math.sin(th) + y * math.cos(th)] for x, y in poly], np.int32)
                        cv2.polylines(img, [pts], True, (220, 40, 40), 2, cv2.LINE_AA)
                    err = math.hypot(px - real[0], py - real[1]); aerr = abs((th - real[2] + math.pi) % (2 * math.pi) - math.pi) * 180 / math.pi
                    cap = f"{k} controls ahead: error {err:.1f} px, {aerr:.1f} deg"
                else:
                    err = aerr = 0.0; cap = "start of the prediction"
                img = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
                bar = text_bar(TILE, 40, [(f"{label(s)}: world model (red) vs simulator (grey)", 0.48, (20, 20, 20)), (cap, 0.45, (90, 90, 90))])
                frame_img = np.vstack([bar, img])
                frames_out.append(frame_img)
                if k in (0, 8, 16, 32):
                    stills_out.append(frame_img)
                if k in (8, 32):
                    summary.append((s, t0, k, round(err, 2), round(aerr, 2)))
            for e in envs.values():
                e.close()
        with writer(V / f"imagined_vs_real_{s}.mp4", fps=15) as w:   # half speed: 15 controls per second
            for f in frames_out:
                w.append_data(pad16(f))
        if stills_out:
            rows = [np.hstack(stills_out[i:i + 4]) for i in range(0, len(stills_out) - len(stills_out) % 4, 4)]
            cv2.imwrite(str(I / f"imagined_vs_real_{s}.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
        print(f"world model {s}: {len(starts)} windows", flush=True)
    with open(OUT / "data" / "world_model_examples.csv", "w") as f:
        f.write("shape,window_start,controls_ahead,centre_error_px,angle_error_deg\n")
        for r in summary:
            f.write(",".join(map(str, r)) + "\n")


def dataviews():
    """How training data is made: (1) correction-round records - at a state the camera policy reached, its own 8 sampled
    plans vs the teacher's label; (2) the teacher's intended 16-step plan vs the 16 pushes it actually executed (it
    re-plans after every push, so executed pushes stitch pieces of different plans)."""
    import glob
    from collections import defaultdict
    import pygame
    from tdyssey.envs.official import make_official_shape_env_class, matched_reset
    from tdyssey.official import load_official_source
    I = OUT / "images" / "training_data"; I.mkdir(parents=True, exist_ok=True)
    sd = torch.load(REPO / "checkpoints/official-dp/ema_policy.pt", map_location="cpu", weights_only=True)["state_dict"]
    scale = sd["normalizer.params_dict.action.scale"].numpy(); offset = sd["normalizer.params_dict.action.offset"].numpy()
    unnorm = lambda x: (np.asarray(x, np.float32) - offset) / scale
    tiles = []
    for s in ["K", "R", "E", "W", "A", "D"]:
        files = sorted(glob.glob(str(WORK / "runs/mpc-data/dagger26m-r50/data" / f"{s}-*.npz")))
        if not files:
            continue
        z = np.load(files[0])
        k = len(z["control"]) // 2                                   # a state in the middle of the episode
        cam = z["image"][k, -1].transpose(1, 2, 0)                    # latest camera frame (96 px)
        img = cv2.resize(cam, (384, 384), interpolation=cv2.INTER_NEAREST)
        f = 384 / 512
        for j in range(8):                                            # the policy's own samples
            pts = (unnorm(z["candidate_x"][k, j]) * f).astype(np.int32)
            cv2.polylines(img, [pts], False, (90, 140, 230), 1, cv2.LINE_AA)
        lab = (unnorm(z["candidate_x"][k, 8]) * f).astype(np.int32)  # the teacher's label
        cv2.polylines(img, [lab], False, (230, 160, 20), 3, cv2.LINE_AA)
        for q in lab:
            cv2.circle(img, tuple(int(v) for v in q), 3, (230, 160, 20), -1, cv2.LINE_AA)
        bar = text_bar(384, 40, [(f"{label(s)}, control {int(z['control'][k])}", 0.5, (20, 20, 20)),
                                 ("blue: policy's 8 samples, gold: teacher's label", 0.42, (90, 90, 90))])
        tiles.append(np.vstack([bar, img]))
    if tiles:
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
        cv2.imwrite(str(I / "correction_round_examples.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    cls = make_official_shape_env_class(load_official_source(REPO / "third_party" / "diffusion_policy"))
    tiles, jumps = [], []
    for s in ["T", "Y", "E", "R"]:
        path = WORK / "slides_data" / "teacher" / f"{s}.npz"
        if not path.exists():
            continue
        z = np.load(path); n = int(z["mask"][0].sum()); act, plan = z["action"][0], z["plan"][0]
        for t0 in (n // 4,):
            partner = "L" if s == "T" else s
            envs = {"T": cls(shape="T", legacy=True), partner: cls(shape=partner, legacy=True)}
            matched_reset(envs, int(z["seed"][0]), defaultdict(int))
            env = envs[s]
            for k in range(t0):
                env.step(act[k])
            env._get_obs()
            img = np.ascontiguousarray(np.transpose(pygame.surfarray.array3d(env.screen), (1, 0, 2)))
            intended = plan[t0].astype(np.int32); executed = act[t0:t0 + 16].astype(np.int32)
            cv2.polylines(img, [executed], False, (200, 60, 60), 2, cv2.LINE_AA)
            for q in executed:
                cv2.circle(img, tuple(int(v) for v in q), 4, (200, 60, 60), -1, cv2.LINE_AA)
            cv2.polylines(img, [intended], False, (230, 160, 20), 3, cv2.LINE_AA)
            for q in intended:
                cv2.circle(img, tuple(int(v) for v in q), 4, (230, 160, 20), -1, cv2.LINE_AA)
            img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
            step = np.linalg.norm(np.diff(executed.astype(float), axis=0), axis=1).max()
            step_i = np.linalg.norm(np.diff(intended.astype(float), axis=0), axis=1).max()
            jumps.append((s, t0, round(float(step_i), 1), round(float(step), 1)))
            bar = text_bar(384, 40, [(f"{label(s)}: gold intended plan, red executed pushes", 0.45, (20, 20, 20)),
                                     (f"largest step: {step_i:.0f} px intended vs {step:.0f} px executed", 0.42, (90, 90, 90))])
            tiles.append(np.vstack([bar, img]))
            for e in envs.values():
                e.close()
    if tiles:
        cv2.imwrite(str(I / "labels_intended_vs_executed.png"), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
        with open(OUT / "data" / "label_smoothness_examples.csv", "w") as f:
            f.write("shape,control,intended_max_step_px,executed_max_step_px\n" + "".join(",".join(map(str, r)) + "\n" for r in jumps))
    print("wrote training-data images", flush=True)


def extras():
    """Last-mile close-ups (camera-pixel grid), a near-miss gallery and the video legend card."""
    I = OUT / "images"; (I / "last_mile").mkdir(parents=True, exist_ok=True)
    lanes = []
    for seed in SEEDS:
        if have("l26_r50", seed):
            lanes += [(seed, l) for l in lanes_of("l26_r50", seed, ALL33)]
    near = sorted([(seed, l) for seed, l in lanes if not l["solved"] and l["peak"] >= 0.9 and l["shape_name"] not in HELD],
                  key=lambda x: -x[1]["peak"])
    solved = [(seed, l) for seed, l in lanes if l["solved"]]

    def zoom(l, k):
        img = decode(l["frames"][k]).copy()
        img[:132, :132] = 255                                         # hide the camera inset in the corner
        g = img[96:288, 96:288]                                       # centre of the table (the goal area)
        g = cv2.resize(g, (576, 576), interpolation=cv2.INTER_NEAREST)
        step = 576 / 192 * (384 / 96)                                 # one 96 px camera pixel = 4 frame px = 12 zoomed px
        for x in np.arange(0, 576, step):
            cv2.line(g, (int(x), 0), (int(x), 575), (200, 200, 200), 1)
            cv2.line(g, (0, int(x)), (575, int(x)), (200, 200, 200), 1)
        return g
    tiles = []
    for (seed, l), tag in ([(near[0], "near miss")] if near else []) + ([(solved[0], "solved")] if solved else []):
        k = l["peak_move"] if tag == "near miss" else len(l["frames"]) - 1
        cov = l["coverage"][k]
        tiles.append(np.vstack([text_bar(576, 40, [(f"{tag}: {label(l['shape_name'])}, {100 * cov:.1f}% covered", 0.6, (20, 20, 20)),
                                                   ("grid: one 96 px camera pixel (about 5 simulator px)", 0.48, (90, 90, 90))]), zoom(l, k)]))
    if tiles:
        cv2.imwrite(str(I / "last_mile" / "near_miss_vs_solved_zoom.png"), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
    gal = []
    for seed, l in near[:12]:
        k = l["peak_move"]
        img = cv2.resize(decode(l["frames"][k]), (240, 240), interpolation=cv2.INTER_AREA)
        gal.append(np.vstack([text_bar(240, 30, [(f"{label(l['shape_name'])}: best {100 * l['peak']:.1f}%", 0.45, (150, 40, 40))]), img]))
    if gal:
        blank = np.full_like(gal[0], 255)
        rows = [np.hstack(gal[i:i + 6] + [blank] * (6 - len(gal[i:i + 6]))) for i in range(0, len(gal), 6)]
        cv2.imwrite(str(I / "last_mile" / "near_miss_gallery.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    card = np.full((300, 760, 3), 255, np.uint8)
    items = [((144, 238, 144), "rect", "goal outline"), ((119, 136, 153), "rect", "block (the letter)"),
             ((65, 105, 225), "circle", "pusher"), ((230, 60, 60), "dot", "next targets the policy will execute"),
             ((240, 165, 165), "dot", "rest of its 16-step plan (replaced when it plans again)"),
             ((50, 50, 50), "inset", "top-left inset: the camera image the policy sees")]
    for i, (col, kind, text) in enumerate(items):
        y = 30 + i * 45
        if kind == "rect":
            cv2.rectangle(card, (20, y - 12), (60, y + 12), col, -1)
        elif kind == "circle":
            cv2.circle(card, (40, y), 14, col, -1, cv2.LINE_AA)
        elif kind == "dot":
            cv2.circle(card, (40, y), 7, col, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(card, (22, y - 14), (58, y + 14), col, 2)
        cv2.putText(card, text, (80, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(I / "video_legend.png"), cv2.cvtColor(card, cv2.COLOR_RGB2BGR))
    print(f"wrote last-mile images ({len(near)} near misses) and the legend", flush=True)


def dynamics():
    """The world model's training data: the original policy with Gaussian noise on its targets (0, 10, 25, 50 px)."""
    agent, cls = load("original")
    shapes = ["T", "D", "K", "E"]
    lanes = []
    for noise in (0.0, 10.0, 25.0, 50.0):
        d = rollout(agent, cls, shapes, 600000, action_noise=noise)
        for s in shapes:
            d[s]["shape_name"] = s; d[s]["noise"] = noise
            lanes.append(d[s])
    for l in lanes:
        l["frames"] = l["frames"]
    V = OUT / "videos" / "training_data"
    n = max(len(l["frames"]) for l in lanes) + 30
    with writer(V / "dynamics_data_action_noise.mp4") as w:
        for i in range(n):
            rows = []
            for r, noise in enumerate((0, 10, 25, 50)):
                tiles = [tile_at(l, i, 192, name=f"{label(l['shape_name'])}, noise {noise} px") for l in lanes[r * 4:(r + 1) * 4]]
                rows.append(np.hstack(tiles))
            body = np.vstack(rows)
            w.append_data(pad16(np.vstack([header_bar(body.shape[1], ("World-model training data", "the original policy with random noise on its targets (rows: 0, 10, 25, 50 px)")), body])))
    print("wrote dynamics-data video", flush=True)


def stills():
    """Shape gallery (every block on its goal outline) and what each camera sees of one scene."""
    from collections import defaultdict
    import pygame
    from tdyssey.envs.official import make_official_shape_env_class, matched_reset
    from tdyssey.official import load_official_source
    cls = make_official_shape_env_class(load_official_source(REPO / "third_party" / "diffusion_policy"))
    I = OUT / "images"; I.mkdir(parents=True, exist_ok=True)
    tiles = []
    for s in ALL33:
        env = cls(shape=s, legacy=True)
        env.seed(0); env.reset()
        from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
        g = env.goal_pose
        goal = pymunk_to_shapely(env._get_goal_pose_body(g), env.block.shapes).centroid
        st = np.array([60.0, 60.0, goal.x, goal.y, g[2]])
        for _ in range(3):                                   # put the block's centre on the goal's centre
            env._set_state(st)
            c = pymunk_to_shapely(env.block, env.block.shapes).centroid
            st[2:4] += [goal.x - c.x, goal.y - c.y]
        env._set_state(st)
        env._get_obs()
        img = np.ascontiguousarray(np.transpose(pygame.surfarray.array3d(env.screen), (1, 0, 2)))
        img = cv2.resize(img[96:416, 96:416], (200, 200), interpolation=cv2.INTER_AREA)
        tiles.append(np.vstack([text_bar(200, 30, [(label(s), 0.5, (20, 20, 20))]), img]))
        env.close()
    blank = np.full_like(tiles[0], 255)
    rows = [np.hstack(tiles[r * 6:(r + 1) * 6] + [blank] * (6 - len(tiles[r * 6:(r + 1) * 6]))) for r in range(6)]
    cv2.imwrite(str(I / "shape_gallery_33.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    tiles = []                                                       # the same shapes upright, without goals
    for s in ALL33:
        env = cls(shape=s, legacy=True)
        env.seed(0); env.reset()
        st = np.array([30.0, 30.0, 256.0, 256.0, 0.0])
        for _ in range(3):
            env._set_state(st)
            from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
            c = pymunk_to_shapely(env.block, env.block.shapes).centroid
            st[2:4] += [256 - c.x, 256 - c.y]
        env._set_state(st); env.goal_pose[:2] = -1000                 # move the goal off the table
        env._get_obs()
        img = np.ascontiguousarray(np.transpose(pygame.surfarray.array3d(env.screen), (1, 0, 2)))
        img = cv2.resize(img[136:376, 136:376], (180, 180), interpolation=cv2.INTER_AREA)
        tiles.append(np.vstack([text_bar(180, 28, [(label(s), 0.48, (20, 20, 20))]), img]))
        env.close()
    blank = np.full_like(tiles[0], 255)
    rows = [np.hstack(tiles[r * 11:(r + 1) * 11] + [blank] * (11 - len(tiles[r * 11:(r + 1) * 11]))) for r in range(3)]
    cv2.imwrite(str(I / "shape_gallery_33_upright.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    views = []
    for size in (512, 192, 96):
        envs = {"T": cls(shape="T", legacy=True, render_size=size), "K": cls(shape="K", legacy=True, render_size=size)}
        matched_reset(envs, SEEDS[0], defaultdict(int))
        cam = np.ascontiguousarray((np.moveaxis(envs["K"]._get_obs()["image"], 0, -1) * 255).clip(0, 255).astype(np.uint8))
        cam = cv2.resize(cam, (384, 384), interpolation=cv2.INTER_NEAREST)
        name = "simulator drawing (512 px)" if size == 512 else f"policy camera input ({size} px)"
        views.append(np.vstack([text_bar(384, 34, [(name, 0.55, (20, 20, 20))]), cam]))
        for e in envs.values():
            e.close()
    cv2.imwrite(str(I / "camera_views_512_192_96.png"), cv2.cvtColor(np.hstack(views), cv2.COLOR_RGB2BGR))
    print("wrote shape gallery and camera views")


# ---------------------------------------------------------------- composition
def decode(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def text_bar(width, height, lines, size=0.6, color=(25, 25, 25), bg=255):
    bar = np.full((height, width, 3), bg, np.uint8)
    y = int(height * 0.42) if len(lines) > 1 else int(height * 0.68)
    for k, (t, sc, col) in enumerate(lines):
        cv2.putText(bar, t, (8, y + k * int(height * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
    return bar


def outcome(lane):
    if lane["solved"]:
        return f"solved in {lane['moves']} moves"
    return f"not solved, best {100 * lane['peak']:.1f}%"


def tile_at(lane, i, size, caption=True, name=None):
    k = min(i, len(lane["frames"]) - 1)
    img = decode(lane["frames"][k])
    if size != TILE:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if not caption:
        return img
    end = i >= len(lane["frames"]) - 1
    status = outcome(lane) if end else f"move {k}, {lane['coverage'][k]:.0%} covered"
    col = (20, 120, 40) if end and lane["solved"] else ((170, 40, 40) if end else (90, 90, 90))
    sc = 0.5 if size >= 256 else 0.38
    bar = text_bar(size, 32 if size >= 256 else 26, [(name or label(lane_shape(lane)), sc, (20, 20, 20)), (status, sc * 0.92, col)])
    return np.vstack([bar, img])


def lane_shape(lane):
    return lane.get("shape_name", "")


def writer(path, fps=30):
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=16,
                              quality=None, ffmpeg_params=["-crf", "28", "-preset", "medium"])


def pad16(img):
    h, w = img.shape[:2]
    return cv2.copyMakeBorder(img, 0, -h % 16, 0, -w % 16, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def header_bar(width, header):
    if isinstance(header, tuple):
        return text_bar(width, 56, [(header[0], 0.75, (15, 15, 15)), (header[1], 0.55, (90, 90, 90))])
    return text_bar(width, 44, [(header, 0.8, (15, 15, 15))])


def grid_video(path, lanes, cols, size, header):
    n = max(len(l["frames"]) for l in lanes) + 45
    with writer(path) as w:
        for i in range(n):
            tiles = [tile_at(l, i, size) for l in lanes]
            blank = np.full_like(tiles[0], 255)
            rows = [np.hstack(tiles[r * cols:(r + 1) * cols] + [blank] * (cols - len(tiles[r * cols:(r + 1) * cols])))
                    for r in range((len(tiles) + cols - 1) // cols)]
            body = np.vstack(rows)
            w.append_data(pad16(np.vstack([header_bar(body.shape[1], header), body])))


def grid_image(path, lanes, cols, size, header, which):
    tiles = []
    for l in lanes:
        i = 0 if which == "start" else len(l["frames"]) - 1
        tiles.append(tile_at(l, i if which == "start" else 10**6, size))
    blank = np.full_like(tiles[0], 255)
    rows = [np.hstack(tiles[r * cols:(r + 1) * cols] + [blank] * (cols - len(tiles[r * cols:(r + 1) * cols])))
            for r in range((len(tiles) + cols - 1) // cols)]
    body = np.vstack(rows)
    img = np.vstack([text_bar(body.shape[1], 44, [(header, 0.8, (15, 15, 15))]), body])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def lanes_of(key, seed, shapes):
    d = pickle.load(open(CACHE / f"{key}_{seed}.pkl", "rb"))
    out = []
    for s in shapes:
        if s in d:
            d[s]["shape_name"] = s
            out.append(d[s])
    return out


def have(key, seed):
    return (CACHE / f"{key}_{seed}.pkl").exists()


def arch_clips(V):
    """Architecture experiment: main model (official network) vs Flow-DiT (505M), side by side on the same scenes:
    videos/flowdit/<shape>[_scene2].mp4. Returns INDEX.md rows."""
    rows = []
    for seed in SEEDS:
        if not (have("l26_r50", seed) and have("flowdit", seed)):
            continue
        a, b = lanes_of("l26_r50", seed, ALL33), lanes_of("flowdit", seed, ALL33)
        for la, lb in zip(a, b):
            s = la["shape_name"]
            tag = "" if seed == SEEDS[0] else "_scene2"
            p = V / "flowdit" / f"{s}{tag}.mp4"
            n = max(len(la["frames"]), len(lb["frames"])) + 45
            with writer(p) as w:
                for i in range(n):
                    body = np.hstack([tile_at(la, i, TILE, name="Official network (main model)"),
                                      tile_at(lb, i, TILE, name="Flow-DiT (experiment)")])
                    w.append_data(pad16(np.vstack([text_bar(body.shape[1], 40, [(label(s), 0.8, (15, 15, 15))]), body])))
            rows.append((p, f"{label(s)}: main model vs Flow-DiT", f"main model {outcome(la)} | Flow-DiT {outcome(lb)}"))
    return rows


def flowdit_clips():
    """Only the architecture-experiment clips and their posters (last frames), without touching the other videos."""
    import imageio.v2 as imageio
    V = OUT / "videos"; P = V / "posters"
    rows = arch_clips(V)
    for path, _, _ in rows:
        r = imageio.get_reader(path)
        last = None
        for last in r:
            pass
        r.close()
        cv2.imwrite(str(P / f"flowdit_{path.stem}.png"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    for path, title, result in rows:
        print(f"| `{path.relative_to(V)}` | {title} | {result} |")


def lastmile(seed=400000, near_shape="G", solved_shape="E", half=34.0):
    """Sharp last-millimetre close-ups for the deck: replays the main model's recorded scene (same batch, seed and random
    stream as record(), so the episodes match the cached ones) to read the exact block pose, then draws the goal (green),
    the block (grey) and the uncovered part of the goal (red) at the near miss's best moment and at the solved
    episode's end, cropped to a window of +-half simulator px around the largest gap, with a grid of one 96 px camera
    pixel (512/96 sim px). Writes images/last_mile/zoom_<shape>.png, images/slide/near_miss_<shape>.png (the scene at
    the best moment) and videos/near_misses/near_miss_<shape>_<seed>.mp4."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import MultiPolygon, Polygon
    from tdyssey.collection import derived_seed
    from tdyssey.envs.official import matched_reset
    from tdyssey.official import physical_metrics
    cached = pickle.load(open(CACHE / f"l26_r50_{seed}.pkl", "rb"))
    agent, cls = load("l26_r50")
    from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely     # importable once load() set up the source
    envs = {s: cls(shape=s, legacy=True, render_size=agent.render_size) for s in ALL33}
    observations, _ = matched_reset(envs, seed, {"reset_physics_substeps": 0})
    want = {near_shape, solved_shape}
    lanes = [{"shape": s, "env": envs[s], "done": False, "history": deque([observations[s]] * 2, maxlen=2),
              "generator": torch.Generator(device=agent.device).manual_seed(derived_seed(42, "rl-eval", seed)),
              "cov": [physical_metrics(envs[s])["coverage"]], "geo": []} for s in ALL33]
    def geo(env):
        return (pymunk_to_shapely(env._get_goal_pose_body(env.goal_pose), env.block.shapes), pymunk_to_shapely(env.block, env.block.shapes))
    for l in lanes:
        if l["shape"] in want:
            l["geo"].append(geo(l["env"]))
    for start in range(0, 300, 8):
        active = [l for l in lanes if not l["done"]]
        if not active:
            break
        actions, _ = agent.sample_batch(agent.encode_batch([l["history"] for l in active]), [l["generator"] for l in active])
        for i, lane in enumerate(active):
            for action in actions[i][:min(8, 300 - start)]:
                obs, _, done, _ = lane["env"].step(action)
                lane["history"].append(obs)
                lane["cov"].append(physical_metrics(lane["env"])["coverage"])
                if lane["shape"] in want:
                    lane["geo"].append(geo(lane["env"]))
                if done:
                    lane["done"] = True
                    break
    I = OUT / "images"
    for lane in lanes:
        s = lane["shape"]
        if s not in want:
            continue
        same = np.allclose(lane["cov"], cached[s]["coverage"])
        print(f"{s}: replay matches the recorded episode: {same}", flush=True)
        if not same:
            raise RuntimeError("replay differs from the cached recording; record the lanes again on this GPU")
        k = int(np.argmax(lane["cov"])) if s == near_shape else len(lane["cov"]) - 1
        goal, block = lane["geo"][k]
        gap = goal.difference(block)
        parts = list(gap.geoms) if hasattr(gap, "geoms") else [gap]
        big = max(parts, key=lambda g: g.area)
        cx, cy = big.centroid.x, big.centroid.y
        fig = plt.figure(figsize=(3, 3), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        def fill(g, **kw):
            for q in (list(g.geoms) if hasattr(g, "geoms") else [g]):
                if not q.is_empty:
                    ax.fill(*q.exterior.xy, **kw)
        fill(goal, color="#90EE90", lw=0)
        fill(block, color="#778899", alpha=0.92, lw=0)
        fill(gap, color="#E03C3C", lw=0)
        cam = 512 / 96
        for v in np.arange(0, 512 + cam, cam):
            ax.axvline(v, color="#c8c8c8", lw=0.6, zorder=0.5)
            ax.axhline(v, color="#c8c8c8", lw=0.6, zorder=0.5)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half, cy - half)                     # screen convention: y grows downward
        ax.set_axis_off()
        (I / "last_mile").mkdir(parents=True, exist_ok=True)
        fig.savefig(I / "last_mile" / f"zoom_{s}.png", facecolor="white")
        plt.close(fig)
        print(f"{s}: coverage {100 * lane['cov'][k]:.1f}% at move {k}, largest gap {big.area:.1f} sim px^2", flush=True)
    for e in envs.values():
        e.close()
    # the near miss as a scene still (at its best moment) and a clip, from the cached frames
    l = cached[near_shape]; l["shape_name"] = near_shape
    k = l["peak_move"]
    img = decode(l["frames"][k]).copy()
    cv2.imwrite(str(I / "slide" / f"near_miss_{near_shape}.png"), cv2.cvtColor(img[7:-7, 7:-7], cv2.COLOR_RGB2BGR))
    p = OUT / "videos" / "near_misses" / f"near_miss_{near_shape}_{seed}.mp4"
    with writer(p) as w:
        for i in range(len(l["frames"]) + 45):
            w.append_data(pad16(tile_at(l, i, TILE, name=f"Ours on {label(near_shape)}")))
    print(f"wrote {p.name}: best {100 * l['peak']:.1f}% at move {k}", flush=True)


NEVER9 = ["digit2", "digit3", "digit4", "digit5", "digit7", "digit8", "plus", "hash", "arrow"]


def gallery():
    """Title and question-slide shape pictures (no GPU): the 26 capital letters in alphabetical order and the nine
    never-seen shapes (lighter), upright and without goals: images/slide/shapes_title.png (13 per row, the never-seen
    row centred) and images/slide/shapes_26_row.png (the letters in one row). Needs a checkout whose tdyssey has the
    digits 2, 5 and 8 (REPO)."""
    import pygame
    from tdyssey.envs.official import make_official_shape_env_class
    from tdyssey.official import load_official_source
    cls = make_official_shape_env_class(load_official_source(REPO / "third_party" / "diffusion_policy"))
    from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
    def glyph(s, light):
        env = cls(shape=s, legacy=True)
        env.seed(0); env.reset()
        st = np.array([30.0, 30.0, 256.0, 256.0, 0.0])
        for _ in range(3):
            env._set_state(st)
            c = pymunk_to_shapely(env.block, env.block.shapes).centroid
            st[2:4] += [256 - c.x, 256 - c.y]
        env._set_state(st); env.goal_pose[:2] = -1000                 # move the goal off the table
        env._get_obs()
        img = np.ascontiguousarray(np.transpose(pygame.surfarray.array3d(env.screen), (1, 0, 2)))
        env.close()
        img = cv2.resize(img[136:376, 136:376], (180, 180), interpolation=cv2.INTER_AREA)[22:158, 22:158]
        if light:
            img = (img.astype(float) + (255 - img.astype(float)) * 0.55).round().astype(np.uint8)
        return img
    def grid(tiles, cols, gap, centre_last=False):
        h, w = tiles[0].shape[:2]
        rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
        out = []
        for r in rows:
            pad = cols - len(r)
            left = pad // 2 if centre_last else 0
            cells = [np.full_like(tiles[0], 255)] * left + r + [np.full_like(tiles[0], 255)] * (pad - left)
            line = cells[0]
            for c in cells[1:]:
                line = np.hstack([line, np.full((h, gap, 3), 255, np.uint8), c])
            out.append(line)
        img = out[0]
        for line in out[1:]:
            img = np.vstack([img, np.full((gap, img.shape[1], 3), 255, np.uint8), line])
        return img
    letters = [glyph(chr(c), False) for c in range(ord("A"), ord("Z") + 1)]
    never = [glyph(s, True) for s in NEVER9]
    S = OUT / "images" / "slide"
    cv2.imwrite(str(S / "shapes_title.png"), cv2.cvtColor(grid(letters + never, 13, 8, centre_last=True), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(S / "shapes_26_row.png"), cv2.cvtColor(grid(letters, 26, 10), cv2.COLOR_RGB2BGR))
    print("wrote shapes_title.png (26 letters + 9 never-seen) and shapes_26_row.png", flush=True)


def compose():
    V, I = OUT / "videos", OUT / "images"
    index = []
    # 1. every model on the same 9 shapes and scene
    for key, (title, _, all33) in MODELS.items():
        if not have(key, SEEDS[0]):
            continue
        lanes = lanes_of(key, SEEDS[0], SHAPES9)
        p = V / "models" / f"{key}_9shapes.mp4"
        grid_video(p, lanes, 3, 256, title)
        grid_image(I / "models" / f"{key}_9shapes_end.png", lanes, 3, 256, title + ": last frame", "end")
        index.append((p, f"{title}: 9 shapes, same scene", " ".join(f"{label(l['shape_name'])}={'Y' if l['solved'] else 'n'}" for l in lanes)))
    # 2. all 33 shapes
    for key, (title, _, all33) in MODELS.items():
        for seed in SEEDS:
            if not all33 or not have(key, seed):
                continue
            lanes = lanes_of(key, seed, ALL33)
            tag = "" if seed == SEEDS[0] else "_scene2"
            p = V / "all_shapes" / f"{key}_33shapes{tag}.mp4"
            grid_video(p, lanes, 6, 160, f"{title}: all 33 shapes")
            grid_image(I / "all_shapes" / f"{key}_33shapes{tag}_end.png", lanes, 6, 192, f"{title}: last frames", "end")
            if key == "original" and seed == SEEDS[0]:
                grid_image(I / "all_shapes" / "start_positions_33shapes.png", lanes, 6, 192, "Starting scene for all 33 shapes", "start")
            index.append((p, f"{title}: all 33 shapes", f"solved {sum(l['solved'] for l in lanes)}/33"))
            if seed == SEEDS[0]:
                p = V / "all_shapes" / f"{key}_33shapes_wide.mp4"             # 11 x 3: fits a 16:9 slide
                grid_video(p, lanes, 11, 160, f"{title}: all 33 shapes")
                index.append((p, f"{title}: all 33 shapes, wide layout", f"solved {sum(l['solved'] for l in lanes)}/33"))
    # 3. original vs main model, side by side, every shape and both scenes
    for seed in SEEDS:
        if not (have("original", seed) and have("l26_r50", seed)):
            continue
        a, b = lanes_of("original", seed, ALL33), lanes_of("l26_r50", seed, ALL33)
        for la, lb in zip(a, b):
            s = la["shape_name"]
            tag = "" if seed == SEEDS[0] else "_scene2"
            p = V / "compare" / f"{s}{tag}.mp4"
            n = max(len(la["frames"]), len(lb["frames"])) + 45
            with writer(p) as w:
                for i in range(n):
                    body = np.hstack([tile_at(la, i, TILE, name="Original (T only)"), tile_at(lb, i, TILE, name="Tdyssey (26 letters)")])
                    w.append_data(pad16(np.vstack([text_bar(body.shape[1], 40, [(label(s), 0.8, (15, 15, 15))]), body])))
            index.append((p, f"{label(s)}: original vs main model", f"original {outcome(la)} | Tdyssey {outcome(lb)}"))
    # 3b. architecture experiment: main model (official network) vs Flow-DiT, same scenes
    index += arch_clips(V)
    # 4. the same scene across the whole project
    prog = ["original", "five_r7", "ten_96", "l26_r0", "l26_r13", "l26_r50"]
    if all(have(k, SEEDS[0]) for k in prog):
        for s in ["K", "R", "D", "digit7", "T", "T_narrow", "Y", "E", "L"]:
            lanes = [lanes_of(k, SEEDS[0], [s])[0] for k in prog]
            names = ["Original", "5 shapes r7", "10 shapes", "26 letters r0", "26 letters r13", "26 letters r50"]
            p = V / "progression" / f"{s}_across_models.mp4"
            n = max(len(l["frames"]) for l in lanes) + 45
            with writer(p) as w:
                for i in range(n):
                    tiles = [tile_at(l, i, 256, name=nm) for l, nm in zip(lanes, names)]
                    body = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
                    w.append_data(pad16(np.vstack([text_bar(body.shape[1], 44, [(f"{label(s)}: the same scene through the project", 0.8, (15, 15, 15))]), body])))
            index.append((p, f"{label(s)} across 6 models", " | ".join(f"{nm}: {outcome(l)}" for nm, l in zip(names, lanes))))
    # 5. near misses (trained shapes) and every never-seen shape of the main model
    for seed in SEEDS:
        if not have("l26_r50", seed):
            continue
        for l in lanes_of("l26_r50", seed, ALL33):
            s = l["shape_name"]
            near = (not l["solved"]) and l["peak"] >= 0.9 and s not in HELD
            if near or s in HELD:
                kind = "near_miss" if near else "held_out"
                folder = "near_misses" if near else "held_out"
                p = V / folder / f"{kind}_{s}_{seed}.mp4"
                with writer(p) as w:
                    for i in range(len(l["frames"]) + 45):
                        w.append_data(pad16(tile_at(l, i, TILE)))
                index.append((p, f"main model, {label(s)} ({kind.replace('_', ' ')})", outcome(l)))
    # 6. the planner teacher (privileged; training only), and teacher vs main model on the same scene
    if (CACHE / f"teacher_{SEEDS[0]}.pkl").exists():
        d = pickle.load(open(CACHE / f"teacher_{SEEDS[0]}.pkl", "rb"))
        for s in d:
            d[s]["shape_name"] = s
        head = ("Planner teacher", "reads exact simulator state; used only to make training data")
        lanes = [d[s] for s in TEACHER_SHAPES if s in d]
        p = V / "teacher" / "teacher_9shapes.mp4"
        grid_video(p, lanes, 3, 256, head)
        index.append((p, "planner teacher, 9 shapes", " ".join(f"{label(l['shape_name'])}={'Y' if l['solved'] else 'n'}" for l in lanes)))
        all27 = [d[s] for s in TRAINED27 if s in d]
        if len(all27) == 27:
            p = V / "teacher" / "teacher_27shapes.mp4"
            grid_video(p, all27, 9, 192, head)
            index.append((p, "planner teacher, all 27 trained shapes", f"solved {sum(l['solved'] for l in all27)}/27"))
        if have("l26_r50", SEEDS[0]):
            student = {l["shape_name"]: l for l in lanes_of("l26_r50", SEEDS[0], ALL33)}
            for s in TRAINED27:
                if s not in d or s not in student:
                    continue
                la, lb = d[s], student[s]
                p = V / "teacher_vs_student" / f"{s}.mp4"
                n = max(len(la["frames"]), len(lb["frames"])) + 45
                with writer(p) as w:
                    for i in range(n):
                        body = np.hstack([tile_at(la, i, TILE, name="Teacher (exact state)"), tile_at(lb, i, TILE, name="Camera policy (round 50)")])
                        w.append_data(pad16(np.vstack([text_bar(body.shape[1], 40, [(f"{label(s)}: teacher vs the camera policy that learned from it", 0.65, (15, 15, 15))]), body])))
                index.append((p, f"{label(s)}: teacher vs main model, same scene", f"teacher {outcome(la)} | policy {outcome(lb)}"))
        lanes = all27 if len(all27) == 27 else lanes
        for l in lanes:
            p = V / "teacher" / f"teacher_{l['shape_name']}.mp4"
            with writer(p) as w:
                for i in range(len(l["frames"]) + 45):
                    w.append_data(pad16(tile_at(l, i, TILE)))
            index.append((p, f"planner teacher, {label(l['shape_name'])} (red dot: next push, pink: rest of its plan)", outcome(l)))
    # 7. filmstrips: five moments of one episode in a row (for slides and the paper)
    def filmstrip(path, lane, title, size=224):
        n = len(lane["frames"]) - 1
        moments = sorted({0, n // 4, n // 2, 3 * n // 4, n})
        tiles = []
        for k in moments:
            img = cv2.resize(decode(lane["frames"][k]), (size, size), interpolation=cv2.INTER_AREA)
            tiles.append(np.vstack([text_bar(size, 26, [(f"move {k}, {lane['coverage'][k]:.0%}", 0.45, (40, 40, 40))]), img]))
        body = np.hstack(tiles)
        img = np.vstack([text_bar(body.shape[1], 34, [(title, 0.6, (15, 15, 15))]), body])
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    for key in ("l26_r50", "original"):
        if have(key, SEEDS[0]):
            for l in lanes_of(key, SEEDS[0], ALL33):
                filmstrip(I / "filmstrips" / key / f"{l['shape_name']}.png", l, f"{MODELS[key][0]}, {label(l['shape_name'])}: {outcome(l)}")
    if (CACHE / f"teacher_{SEEDS[0]}.pkl").exists():
        d = pickle.load(open(CACHE / f"teacher_{SEEDS[0]}.pkl", "rb"))
        for s, l in d.items():
            filmstrip(I / "filmstrips" / "teacher" / f"{s}.png", l, f"Planner teacher, {label(s)}: {outcome(l)}")
    for v in sorted((V / "world_model").glob("*.mp4")):
        index.append((v, f"{v.stem.replace('imagined_vs_real_', '')}: what the world model imagines (red) vs the simulator, open-loop 32 controls",
                      "errors per window: data/world_model_examples.csv"))
    for v in sorted((V / "training_data").glob("*.mp4")):
        index.append((v, "the world model's training data: original policy with 0/10/25 px noise on its targets (plus a 50 px row for illustration only)", "data collection, not a test"))
    # index
    with open(V / "INDEX.md", "w") as f:
        f.write("# Video index (generated by make_videos.py)\n\n"
                f"Playback: {SPEED}; world-model clips run at half that speed. Clips are examples from validation scenes "
                "(seeds 400000 and 400020), which were also used to pick checkpoints; success rates come from the fresh-scene "
                "tests in ../data. Legend: ../images/video_legend.png.\n\n| File | What | Outcome |\n| --- | --- | --- |\n")
        for p, what, res in index:
            f.write(f"| `{p.relative_to(V)}` | {what} | {res} |\n")
    print(f"wrote {len(index)} videos")
    posters()


def posters():
    """A still per video (its last frame, which shows the outcome) for slides: videos/posters/<name>.png, where name is
    compare_<shape>, teacher_vs_student_<shape>, progression_<shape> or the video's own file name; plus mid-episode
    stills for the clips the deck uses."""
    import imageio.v2 as imageio
    V = OUT / "videos"; P = V / "posters"; P.mkdir(parents=True, exist_ok=True)
    for old in P.glob("*.png"):
        old.unlink()
    n = 0
    for v in sorted(V.rglob("*.mp4")):
        folder, stem = v.parent.name, v.stem
        name = (f"compare_{stem}" if folder == "compare" else f"teacher_vs_student_{stem}" if folder == "teacher_vs_student"
                else f"flowdit_{stem}" if folder == "flowdit"
                else f"progression_{stem.replace('_across_models', '')}" if folder == "progression" else stem)
        r = imageio.get_reader(v)
        last = None
        for last in r:
            pass
        r.close()
        cv2.imwrite(str(P / f"{name}.png"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        n += 1
    print(f"wrote {n} posters")
    mid_stills()


DECK_STILLS = ["compare/P.mp4", "compare/A.mp4", "compare/M.mp4", "compare/S_scene2.mp4", "compare/digit7_scene2.mp4",
               "teacher/teacher_9shapes.mp4", "teacher_vs_student/K.mp4", "teacher_vs_student/E.mp4",
               "progression/E_across_models.mp4", "near_misses/near_miss_M_400020.mp4"]


def mid_stills(where=0.4):
    """Mid-episode stills (the frame at 40% of the clip, with the plan dots visible) for the clips the deck uses:
    videos/posters/<poster name>_mid.png."""
    import imageio.v2 as imageio
    V = OUT / "videos"; P = V / "posters"
    for rel in DECK_STILLS:
        v = V / rel
        folder, stem = v.parent.name, v.stem
        name = (f"compare_{stem}" if folder == "compare" else f"teacher_vs_student_{stem}" if folder == "teacher_vs_student"
                else f"flowdit_{stem}" if folder == "flowdit"
                else f"progression_{stem.replace('_across_models', '')}" if folder == "progression" else stem)
        r = imageio.get_reader(v)
        n = r.count_frames()
        img = r.get_data(int(n * where))
        r.close()
        cv2.imwrite(str(P / f"{name}_mid.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"wrote {len(DECK_STILLS)} mid-episode stills")


def reel():
    """videos/talk/demo_reel.mp4: a 1280x720 recording for the talk that plays without clicking links. Title card,
    legend, then original vs ours on P, A, M and S (2x speed), then the main model on all 33 shapes (2x speed)."""
    import imageio.v2 as imageio
    W, H, fps = 1280, 720, 30
    V = OUT / "videos"

    def card(lines, frames):
        img = np.full((H, W, 3), 255, np.uint8)
        y0 = H // 2 - 40 * (len(lines) - 1)
        for k, (txt, sc, col) in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
            cv2.putText(img, txt, ((W - tw) // 2, y0 + 80 * k), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)
        return [img] * frames

    def fit(img, caption=None):
        top = 60 if caption else 0
        h, w = img.shape[:2]
        s = min(W / w, (H - top) / h)
        img = cv2.resize(img, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
        out = np.full((H, W, 3), 255, np.uint8)
        y, x = top + (H - top - img.shape[0]) // 2, (W - img.shape[1]) // 2
        out[y:y + img.shape[0], x:x + img.shape[1]] = img
        if caption:
            (tw, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(out, caption, ((W - tw) // 2, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        return out

    def clip(rel, caption, step=2, hold=30):
        r = imageio.get_reader(V / rel)
        frames = [fit(f, caption) for k, f in enumerate(r) if k % step == 0]
        r.close()
        return frames + [frames[-1]] * hold

    legend = cv2.cvtColor(cv2.imread(str(OUT / "images" / "video_legend.png")), cv2.COLOR_BGR2RGB)
    with writer(V / "talk" / "demo_reel.mp4", fps=fps) as w:
        for f in card([("Tdyssey: one camera policy for every letter", 1.3, (42, 120, 214)),
                       ("Recorded in simulation; videos at 6x real time", 0.8, (90, 90, 90))], 60):
            w.append_data(f)
        for f in [fit(legend, "How to read the videos")] * 90:
            w.append_data(f)
        for rel, cap in [("compare/P.mp4", "P: original policy (left) vs ours (right), same scene"),
                         ("compare/A.mp4", "A: original vs ours"), ("compare/M.mp4", "M: original vs ours"),
                         ("compare/S_scene2.mp4", "S: original vs ours")]:
            for f in clip(rel, cap):
                w.append_data(f)
        for f in clip("all_shapes/l26_r50_33shapes_wide.mp4", "Ours on all 33 shapes at once (6 never seen in training)", step=2, hold=60):
            w.append_data(f)
    print("wrote videos/talk/demo_reel.mp4")


if __name__ == "__main__":
    if sys.argv[1] == "rollouts":
        record(sys.argv[2:] or list(MODELS))
    elif sys.argv[1] == "teacher":
        teacher(sys.argv[2:] or TRAINED27)
    elif sys.argv[1] == "stills":
        stills()
    elif sys.argv[1] == "worldmodel":
        worldmodel()
    elif sys.argv[1] == "dataviews":
        dataviews()
    elif sys.argv[1] == "extras":
        extras()
    elif sys.argv[1] == "dynamics":
        dynamics()
    elif sys.argv[1] == "lastmile":
        lastmile()
    elif sys.argv[1] == "gallery":
        gallery()
    elif sys.argv[1] == "flowdit":
        flowdit_clips()
    elif sys.argv[1] == "compose":
        compose()
    elif sys.argv[1] == "posters":
        posters()
    elif sys.argv[1] == "reel":
        reel()
        posters()
