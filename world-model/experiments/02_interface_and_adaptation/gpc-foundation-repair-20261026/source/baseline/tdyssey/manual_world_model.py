"""Manual prediction/physics comparison. No training, dataset, or experiment resume."""

from collections import deque
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np

from .envs.official import make_official_shape_env_class, official_shape
from .evaluation import goal_metrics
from .geometry import Pose
from .state import BodyState
from .wm_compat_transform import transform_t1

ROOT = Path(__file__).resolve().parents[1]


def source_path(name):
    for root in (ROOT, ROOT.parent / "Tdyssey"):
        path = root / "third_party" / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Existing third_party/{name} checkout is missing; no download is attempted")


def make_environment(shape):
    from .official import load_official_source

    cls = make_official_shape_env_class(load_official_source(source_path("diffusion_policy")))
    env = cls(shape=shape, legacy=True)
    env.seed(5478)
    env.reset()
    initial = env.capture_state()
    # Deliberately simple, disclosed manual scene; never a held evaluation scene.
    y = 240 if shape == "T" else 285
    initial = replace(initial, object_body=BodyState(Pose((256, y), 0)),
                      pusher_body=BodyState(Pose((256, 405), 0)), pusher_target=None,
                      control_step=0)
    observation = env.restore_state(initial)
    return env, observation


class ManualSession:
    """A preview never steps physics; only execute consumes its exact pending action."""

    def __init__(self, env_factory=make_environment):
        self.factory = env_factory
        self.env = None
        self.revision = 0
        self.role, self.horizon, self.noise_seed = "frozen", 1, 5478
        self.controls = self.resets = self.forecasts = 0
        self.pending = self.prediction = self.actual = self.measurement = None
        self.target = None
        self.done = False
        self.reset("T")

    def invalidate(self):
        self.revision += 1
        self.pending = self.prediction = self.actual = self.measurement = None

    def reset(self, shape=None):
        shape = shape or self.shape
        if shape not in ("T", "L"):
            raise ValueError("Manual demo supports T and L")
        if self.env is not None:
            self.env.close()
        self.shape = shape
        self.env, self.observation = self.factory(shape)
        self.resets += 1
        self.images = deque([self.observation["image"].copy()], maxlen=4)
        self.actions = deque(maxlen=3)
        self.done = False
        self.target = None
        # Four real observations, with three recorded hold actions between them.
        hold = np.asarray(self.observation["agent_pos"], dtype=np.float64)
        for _ in range(3):
            self._step(hold)
        self.invalidate()

    def configure(self, *, role=None, horizon=None):
        if role is not None:
            if role not in ("frozen", "adapted"):
                raise ValueError("Unknown checkpoint")
            self.role = role
        if horizon is not None:
            if horizon not in (1, 4):
                raise ValueError("Preview horizon must be 1 or 4 control steps")
            self.horizon = horizon
        self.invalidate()

    def select(self, target):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all() or (target < 0).any() or (target > 511).any():
            raise ValueError("Target must be a finite 2D workspace point in [0,511]")
        self.target = target.copy()
        self.invalidate()

    def request(self):
        if self.done:
            raise ValueError("The real episode has succeeded; reset before another action")
        if self.target is None:
            raise ValueError("Choose a target in the left panel first")
        if len(self.images) != 4 or len(self.actions) != 3:
            raise ValueError("Four real frames and three intervening actions are required")
        return dict(revision=self.revision, role=self.role, shape=self.shape,
                    horizon=self.horizon, noise_seed=self.noise_seed,
                    target=self.target.copy(), images=np.stack(self.images),
                    past_actions=np.asarray(self.actions, dtype=np.float32),
                    goal=self.env.capture_state().goal)

    def accept(self, request, raw, measurement):
        if (request["revision"] != self.revision or request["role"] != self.role
                or request["shape"] != self.shape or request["horizon"] != self.horizon
                or not np.array_equal(request["target"], self.target)):
            raise ValueError("Prediction belongs to an obsolete scene/action/checkpoint")
        raw = np.asarray(raw)
        if raw.shape != (self.horizon, 3, 96, 96) or not np.isfinite(raw).all():
            raise ValueError("Prediction is missing, malformed or nonfinite")
        self.pending, self.prediction, self.measurement = request, raw.copy(), measurement
        self.actual = None
        self.forecasts += self.horizon

    def _step(self, action):
        observation, reward, done, info = self.env.step(np.asarray(action, dtype=np.float64))
        self.observation = observation
        self.images.append(observation["image"].copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.controls += 1
        self.done = bool(done)
        return dict(reward=float(reward), done=self.done, contacts=int(info["n_contacts"]))

    def execute(self):
        if self.pending is None or self.pending["revision"] != self.revision or self.done:
            raise ValueError("Generate a current prediction before executing")
        request = self.pending
        self.pending = None  # An action cannot be executed twice by key repeat.
        actual, rows = [], []
        for _ in range(request["horizon"]):
            rows.append(self._step(request["target"]))
            actual.append(transform_t1(self.observation["image"])[0])
            if self.done:
                break
        self.actual = np.stack(actual)
        matched = len(actual) - 1
        if len(actual) != request["horizon"]:
            # The decoder described the requested final horizon, not this
            # earlier real success boundary now displayed on screen.
            self.measurement = None
        state = self.env.capture_state()
        truth = goal_metrics(official_shape(self.shape), state.object_body.pose, state.goal)
        result = dict(steps=rows, executed_controls=len(rows), matched_horizon=matched + 1,
                      truth_coverage=truth.coverage, truth_success=self.done,
                      pixel_mae=float(np.abs(self.prediction[matched] - self.actual[matched]).mean()),
                      note="Manual exploration only; image MAE is not a dynamics acceptance score")
        self.revision += 1
        return result

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class PinnedPredictor:
    """One active model; reuse the audited model-only loader and native sampler."""

    def __init__(self):
        self.role = self.model = self.sampler = self.assets = None
        self.loads = 0
        self.peak_allocated = self.peak_reserved = 0

    def check_cuda(self):
        import torch

        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())
        if self.peak_allocated > 512 * 1024**2 or self.peak_reserved > 1024**3:
            raise MemoryError("Manual preview exceeded its 512MiB allocated / 1GiB reserved CUDA bounds")

    def load(self, role):
        if self.role == role:
            return
        self.close()
        from .gpc_evaluation_assets import verify_evaluation_assets, load_model_only_state
        from .gpc_supervised_adaptation import make_real_denoiser
        from .gpc_adaptation import NativeSampler
        import torch

        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no download or CPU fallback is attempted")
        if self.assets is None:
            self.assets = verify_evaluation_assets(ROOT)
        state = load_model_only_state(role, self.assets)
        try:
            self.model = make_real_denoiser(source_path("gpc_code"),
                self.assets["weights"]["frozen"]["path"], "cuda",
                dict(execute=True, evaluation_only=True,
                     validated_checkpoint_identity=self.assets["weights"][role]), state_dict=state)
        finally:
            del state
        self.model.eval().requires_grad_(False)
        self.sampler = NativeSampler(self.model, self.check_cuda)
        self.role = role
        self.loads += 1
        self.check_cuda()

    def predict(self, request):
        import torch
        from .wm_image_measurement_v2 import decode
        from .wm_compat_engine import compact_measurement

        started = time.monotonic()
        self.load(request["role"])
        context = np.stack([transform_t1(frame)[0] for frame in request["images"]])
        past = list(request["past_actions"].copy())
        raw = []
        # Same fixed sampling noise for checkpoint comparisons; external RNG restored.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
            self.sampler.seed(request["noise_seed"])
            for _ in range(request["horizon"]):
                window = np.asarray(past + [request["target"]], dtype=np.float32)
                window = window / np.float32(511) * np.float32(2) - np.float32(1)
                image = self.sampler.sample(context, window)
                if not np.isfinite(image).all():
                    raise ValueError("Nonfinite prediction")
                raw.append(image)
                # Raw predicted RGB is fed back. T1 applies only to real observations.
                context = np.concatenate((context[1:], image[None]), axis=0)
                past = past[1:] + [request["target"]]
        measurement = compact_measurement(decode(raw[-1], request["shape"], request["goal"]))
        return np.stack(raw), measurement, time.monotonic() - started

    def close(self):
        self.role = self.sampler = self.model = None
        gc.collect()
        import sys
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def event(stream, kind, **details):
    stream.write(json.dumps(dict(time=time.time(), kind=kind, **details),
                            ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
