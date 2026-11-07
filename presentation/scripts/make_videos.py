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
def rollout(agent, cls, shapes, seed, execute=8, noise_seed=42):
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


TEACHER_SHAPES = ["T", "T_narrow", "Y", "D", "L", "E", "K", "R", "W"]
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
    return f"not solved, best {lane['peak']:.0%}"


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


def grid_video(path, lanes, cols, size, header):
    n = max(len(l["frames"]) for l in lanes) + 45
    with writer(path) as w:
        for i in range(n):
            tiles = [tile_at(l, i, size) for l in lanes]
            blank = np.full_like(tiles[0], 255)
            rows = [np.hstack(tiles[r * cols:(r + 1) * cols] + [blank] * (cols - len(tiles[r * cols:(r + 1) * cols])))
                    for r in range((len(tiles) + cols - 1) // cols)]
            body = np.vstack(rows)
            w.append_data(pad16(np.vstack([text_bar(body.shape[1], 44, [(header, 0.8, (15, 15, 15))]), body])))


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
    # 5. near misses and held-out shapes of the main model
    for seed in SEEDS:
        if not have("l26_r50", seed):
            continue
        for l in lanes_of("l26_r50", seed, ALL33):
            s = l["shape_name"]
            near = (not l["solved"]) and l["peak"] >= 0.9
            if near or s in HELD:
                kind = "near_miss" if near else "held_out"
                p = V / "failures" / f"{kind}_{s}_{seed}.mp4"
                with writer(p) as w:
                    for i in range(len(l["frames"]) + 45):
                        w.append_data(pad16(tile_at(l, i, TILE)))
                index.append((p, f"main model, {label(s)} ({kind.replace('_', ' ')})", outcome(l)))
    # 6. the planner teacher (privileged; training only)
    if (CACHE / f"teacher_{SEEDS[0]}.pkl").exists():
        d = pickle.load(open(CACHE / f"teacher_{SEEDS[0]}.pkl", "rb"))
        lanes = []
        for s in TEACHER_SHAPES:
            if s in d:
                d[s]["shape_name"] = s; lanes.append(d[s])
        p = V / "teacher" / "teacher_9shapes.mp4"
        grid_video(p, lanes, 3, 256, "Planner teacher (reads exact simulator state; used only to make training data)")
        index.append((p, "planner teacher, 9 shapes", " ".join(f"{label(l['shape_name'])}={'Y' if l['solved'] else 'n'}" for l in lanes)))
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
    # index
    with open(V / "INDEX.md", "w") as f:
        f.write("# Video index (generated by make_videos.py)\n\n| File | What | Outcome |\n| --- | --- | --- |\n")
        for p, what, res in index:
            f.write(f"| `{p.relative_to(V)}` | {what} | {res} |\n")
    print(f"wrote {len(index)} videos")
    posters()


def posters():
    """A still per video (its last frame, which shows the outcome) for slides: videos/posters/<name>.png, where name is
    compare_<shape>, progression_<shape> or the video's own file name."""
    import imageio.v2 as imageio
    V = OUT / "videos"; P = V / "posters"; P.mkdir(parents=True, exist_ok=True)
    n = 0
    for v in sorted(V.rglob("*.mp4")):
        folder, stem = v.parent.name, v.stem
        name = f"compare_{stem}" if folder == "compare" else (f"progression_{stem.replace('_across_models', '')}" if folder == "progression" else stem)
        r = imageio.get_reader(v)
        last = None
        for last in r:
            pass
        r.close()
        cv2.imwrite(str(P / f"{name}.png"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        n += 1
    print(f"wrote {n} posters")


if __name__ == "__main__":
    if sys.argv[1] == "rollouts":
        record(sys.argv[2:] or list(MODELS))
    elif sys.argv[1] == "teacher":
        teacher(sys.argv[2:] or TEACHER_SHAPES)
    elif sys.argv[1] == "stills":
        stills()
    elif sys.argv[1] == "compose":
        compose()
    elif sys.argv[1] == "posters":
        posters()
