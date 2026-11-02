"""Bounded full-parameter native GPC RGB training, with no import-time Torch.

The original 96px architecture and phase-two residual objective are reused.
Shapes are dataset metadata only. All local optimizer attempts, including probes
and tiny fits, must use the same persistent CostLedger.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy, deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import threading
import time
import uuid

from .gpc_world_model import load_world_model_source, world_model_spec
from .raw_world_model_training import OBJECTIVE, OPTIMIZER, make_generator, supervised_denoising_loss

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_KIND = "native_gpc_rgb_training.v1"
LEDGER_KIND = "native_gpc_rgb_cost.v1"
SAMPLING_POLICY = "letter_cycle_then_independent_trajectory_cycle_then_shuffled_windows.v1"
DEFAULT_LEGACY_COST = {"cancelled_structured_main_charged_updates": 262,
                       "cancelled_structured_tiny_fit_updates": 100,
                       "cancelled_structured_engineering_updates": 4,
                       "included_in_native_rgb_budget": False}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def disk_bytes(root):
    root = Path(root)
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


class BudgetExceeded(RuntimeError):
    pass


class CostLedger:
    """Durable reservations count attempted updates, including interrupted work.

    Only one stage may run at once. A hard-killed stage remains active and must
    be explicitly reconciled using recover_interrupted; merely reopening the
    ledger never resets consumed budget or silently discards the interruption.
    """
    def __init__(self, path, max_updates=5000, max_seconds=5400., legacy_cost=None):
        if type(max_updates) is not int or max_updates < 1 or not 0 < max_seconds < float("inf"):
            raise ValueError("Use positive finite aggregate budgets")
        self.path = Path(path).resolve()
        self.lock = self.path.with_name(self.path.name + ".lock")
        self.max_updates, self.max_seconds = max_updates, float(max_seconds)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if self.path.exists():
                value = self._read()
                if (value.get("kind") != LEDGER_KIND or value["max_updates"] != max_updates
                        or value["max_seconds"] != self.max_seconds):
                    raise ValueError("Existing ledger identity/budget differs; do not reset it")
            else:
                atomic_json(self.path, dict(kind=LEDGER_KIND, max_updates=max_updates,
                    max_seconds=self.max_seconds, charged_updates=0, completed_updates=0,
                    charged_seconds=0., active_stage=None, stages=[],
                    historical_cancelled_experiment=deepcopy(DEFAULT_LEGACY_COST if legacy_cost is None else legacy_cost)))

    @contextmanager
    def _locked(self):
        descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            yield
        finally:
            os.close(descriptor)
            self.lock.unlink(missing_ok=True)

    def _read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def snapshot(self):
        with self._locked():
            return self._read()

    def start(self, phase):
        with self._locked():
            value = self._read()
            if value["active_stage"] is not None:
                raise RuntimeError("Unfinished ledger stage; reconcile observed elapsed time before continuing")
            if value["charged_seconds"] >= self.max_seconds:
                raise BudgetExceeded("Aggregate native RGB time cap reached")
            token = uuid.uuid4().hex
            value["active_stage"] = dict(token=token, phase=str(phase), pid=os.getpid(),
                started_unix=time.time(), seconds_before=value["charged_seconds"],
                observed_seconds=0., reserved_updates=0, completed_updates=0)
            atomic_json(self.path, value)
        return BudgetStage(self, token)

    def _change(self, token, elapsed, reserve=False, complete=False, end=None):
        with self._locked():
            value = self._read()
            active = value["active_stage"]
            if active is None or active["token"] != token:
                raise RuntimeError("Ledger stage ownership differs")
            elapsed = max(float(elapsed), active["observed_seconds"])
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Elapsed time must be finite and nonnegative")
            active["observed_seconds"] = elapsed
            value["charged_seconds"] = active["seconds_before"] + elapsed
            if complete:
                if active["completed_updates"] >= active["reserved_updates"]:
                    raise RuntimeError("Completion needs a prior durable update reservation")
                active["completed_updates"] += 1
                value["completed_updates"] += 1
            failure = None
            if reserve:
                if value["charged_updates"] >= self.max_updates:
                    failure = "Aggregate native RGB optimizer-attempt cap reached"
                elif value["charged_seconds"] >= self.max_seconds:
                    failure = "Aggregate native RGB time cap reached"
                else:
                    value["charged_updates"] += 1
                    active["reserved_updates"] += 1
            if end is not None:
                active["status"] = str(end)
                active["finished_unix"] = time.time()
                value["stages"].append(active)
                value["active_stage"] = None
            atomic_json(self.path, value)
            if failure:
                raise BudgetExceeded(failure)
            return value

    def recover_interrupted(self, observed_seconds, reason="interrupted_process_reconciled"):
        """Explicit accounting only; elapsed may never move backwards."""
        value = self.snapshot()
        active = value["active_stage"]
        if active is None:
            raise ValueError("No interrupted ledger stage")
        if observed_seconds < active["observed_seconds"]:
            raise ValueError("Cannot reduce already observed cost")
        return self._change(active["token"], observed_seconds, end=reason)


class BudgetStage:
    def __init__(self, ledger, token):
        self.ledger, self.token, self.started = ledger, token, time.monotonic()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.finish("complete" if kind is None else "failed:" + kind.__name__)

    def heartbeat(self):
        value = self.ledger._change(self.token, time.monotonic() - self.started)
        if value["charged_seconds"] >= self.ledger.max_seconds:
            raise BudgetExceeded("Aggregate native RGB time cap reached")
        return value

    def reserve_update(self):
        return self.ledger._change(self.token, time.monotonic() - self.started, reserve=True)

    def complete_update(self):
        return self.ledger._change(self.token, time.monotonic() - self.started, complete=True)

    def finish(self, status="complete"):
        if not self.closed:
            value = self.ledger._change(self.token, time.monotonic() - self.started, end=status)
            self.closed = True
            return value


def construct_model(source_root, init_checkpoint, device="cuda", trainable=True):
    """Build the unchanged full GPC, initialized only from its published asset."""
    import torch
    from types import SimpleNamespace
    from .official_world_model import initial_profile, inspect_profile
    inspected = inspect_profile(initial_profile(init_checkpoint))
    if not inspected["ready"]:
        raise ValueError(inspected["reason"])
    source = load_world_model_source(source_root)
    spec = world_model_spec()
    config = dict(spec["model"])
    denoiser = {key: config.pop(key) for key in ("sigma_data", "sigma_offset_noise", "noise_previous_obs")}
    with torch.random.fork_rng(devices=[]):
        model = source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**config), **denoiser)).float()
    payload = torch.load(init_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(payload, strict=True)
    del payload
    model.to(device).requires_grad_(bool(trainable))
    if trainable:
        model.setup_training(SimpleNamespace(loc=-1.2, scale=1.2, sigma_min=.002, sigma_max=20.))
        model.train()
    else:
        model.eval()
    model.asset_identity = inspected
    return model, source


class WindowOrder:
    """Balance letters and independent trajectories before shuffling windows.

    Long/easy contact trajectories cannot dominate because they have more valid
    windows. Geometry/contact metadata never enters the image/action batch.
    """
    def __init__(self, dataset, seed=5478, indices=None):
        import numpy as np
        selected = list(range(len(dataset))) if indices is None else list(indices)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Need distinct training-window indices")
        self.pools = defaultdict(list)
        groups = defaultdict(lambda: defaultdict(list))
        for index in selected:
            if type(index) is not int or not 0 <= index < len(dataset):
                raise ValueError("Training window index outside dataset")
            trajectory = dataset.windows[index][0]
            letter = dataset.entries[trajectory]["shape"]
            self.pools[letter].append(index)
            groups[letter][str(trajectory)].append(index)
        self.pools = dict(sorted(self.pools.items()))
        self.groups = {letter: dict(sorted(values.items())) for letter, values in sorted(groups.items())}
        self.letters = tuple(self.pools)
        self.rng = np.random.default_rng(seed)
        self.orders = {name: self.rng.permutation(list(self.groups[name])).tolist() for name in self.letters}
        self.positions = {name: 0 for name in self.letters}
        self.epochs = {name: 0 for name in self.letters}
        self.window_orders = {letter: {group: self.rng.permutation(pool).tolist() for group, pool in values.items()}
                              for letter, values in self.groups.items()}
        self.window_positions = {letter: {group: 0 for group in values} for letter, values in self.groups.items()}
        self.window_epochs = deepcopy(self.window_positions)
        self.cursor = 0

    def take(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("Positive sample count required")
        result = []
        for _ in range(count):
            letter = self.letters[self.cursor % len(self.letters)]
            self.cursor += 1
            if self.positions[letter] == len(self.orders[letter]):
                self.orders[letter] = self.rng.permutation(list(self.groups[letter])).tolist()
                self.positions[letter] = 0
                self.epochs[letter] += 1
            group = self.orders[letter][self.positions[letter]]
            self.positions[letter] += 1
            if self.window_positions[letter][group] == len(self.window_orders[letter][group]):
                self.window_orders[letter][group] = self.rng.permutation(self.groups[letter][group]).tolist()
                self.window_positions[letter][group] = 0
                self.window_epochs[letter][group] += 1
            result.append(self.window_orders[letter][group][self.window_positions[letter][group]])
            self.window_positions[letter][group] += 1
        return result

    def state_dict(self):
        return deepcopy(dict(policy=SAMPLING_POLICY, pools=self.pools, groups=self.groups,
            orders=self.orders, positions=self.positions, epochs=self.epochs,
            window_orders=self.window_orders, window_positions=self.window_positions,
            window_epochs=self.window_epochs, cursor=self.cursor, rng=self.rng.bit_generator.state))

    def load_state_dict(self, state):
        if (state["pools"] != self.pools or state["groups"] != self.groups
                or state["policy"] != SAMPLING_POLICY):
            raise ValueError("Checkpoint training-window selection differs")
        for letter, order in state["orders"].items():
            if sorted(order) != sorted(self.groups[letter]):
                raise ValueError("Invalid checkpoint sampling order")
            if not 0 <= state["positions"][letter] <= len(order):
                raise ValueError("Invalid checkpoint sampling position")
        self.orders, self.positions, self.epochs = deepcopy((state["orders"], state["positions"], state["epochs"]))
        for letter, values in state["window_orders"].items():
            for group, order in values.items():
                if sorted(order) != sorted(self.groups[letter][group]):
                    raise ValueError("Invalid trajectory-window order")
                if not 0 <= state["window_positions"][letter][group] <= len(order):
                    raise ValueError("Invalid trajectory-window position")
        self.window_orders, self.window_positions, self.window_epochs = deepcopy(
            (state["window_orders"], state["window_positions"], state["window_epochs"]))
        self.cursor = int(state["cursor"])
        self.rng.bit_generator.state = deepcopy(state["rng"])


class BatchRenderer:
    """One effective batch in flight; independent bounded per-thread renderers.

    The caller has already selected all indices. This object never reads or
    advances WindowOrder, and no pending rendering crosses an optimizer update.
    Float windows live only in this one bounded batch, never on disk.
    """
    def __init__(self, dataset, workers=2, cache_bytes=16*1024**2, max_pending=16):
        if type(workers) is not int or not 1 <= workers <= 16 or max_pending < 1 or cache_bytes < 0:
            raise ValueError("Invalid bounded rendering configuration")
        self.dataset, self.workers = dataset, workers
        self.cache_bytes, self.max_pending = int(cache_bytes), int(max_pending)
        self.local = threading.local()
        self.guard = threading.Lock()
        self.clones = []
        self.active = False
        self.closed = False
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="native-rgb-render")

    def _item(self, index):
        if not hasattr(self.local, "dataset"):
            dataset = copy(self.dataset)
            dataset._images = OrderedDict()
            dataset._image_bytes = 0
            dataset._environments = {}
            dataset._trajectory_cache = None
            dataset.cache_bytes = self.cache_bytes // self.workers
            self.local.dataset = dataset
            with self.guard:
                self.clones.append(dataset)
        return self.local.dataset[int(index)]

    def batches(self, indices, microbatch, device):
        import numpy as np
        import torch
        if self.closed or self.active:
            raise RuntimeError("Renderer is closed or already owns a pending batch")
        if len(indices) > self.max_pending or not indices or microbatch < 1:
            raise ValueError("Rendering may prefetch at most one configured effective batch")
        self.active = True
        futures = []
        try:
            futures = [self.pool.submit(self._item, int(index)) for index in indices]
            for start in range(0, len(indices), microbatch):
                items = [future.result() for future in futures[start:start + microbatch]]
                yield {name: torch.as_tensor(np.stack([item[name] for item in items]), device=device)
                       for name in ("image", "action")}
        finally:
            for future in futures:
                future.cancel()
            # Wait for running renders before another batch/checkpoint can own
            # these thread-local environments. Discarded renders spend no RNG.
            for future in futures:
                if not future.cancelled():
                    try:
                        future.result()
                    except Exception:
                        pass
            futures.clear()
            self.active = False

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True, cancel_futures=True)
            for dataset in self.clones:
                dataset._images.clear()
                dataset._image_bytes = 0
                dataset._environments.clear()
                dataset._trajectory_cache = None
            self.clones.clear()
            self.closed = True


def write_source_snapshot(output, identity):
    """Preserve exact Python/manifest bytes used when this run began."""
    folder = Path(output) / "source-snapshot"
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in identity["source"]["files"].items():
        target = folder / name
        if target.exists():
            if sha256(target) != expected:
                raise ValueError("Existing run source snapshot differs: " + name)
        else:
            source = ROOT / name
            if sha256(source) != expected:
                raise ValueError("Source changed before its run snapshot was captured: " + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256(target) != expected:
                raise ValueError("Source changed while snapshotting: " + name)
    atomic_json(folder / "identity.json", identity)
    return dict(path=str(folder), files=len(identity["source"]["files"]))


def _cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return deepcopy(value)


def capture_rng(generator):
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().cpu(),
        cuda=[state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None,
        noise_generator=generator.get_state().cpu())


def restore_rng(state, generator):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["noise_generator"])


def save_training_checkpoint(path, model, optimizer, scaler, scheduler, generator, order, progress, config, identity):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    payload = dict(kind=CHECKPOINT_KIND, model=_cpu_tree(model.state_dict()),
        optimizer=_cpu_tree(optimizer.state_dict()), scaler=_cpu_tree(scaler.state_dict()),
        scheduler=_cpu_tree(scheduler.state_dict()), rng=capture_rng(generator),
        order=order.state_dict(), progress=deepcopy(progress), config=deepcopy(config), identity=deepcopy(identity))
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path), updates=progress["updates"])


def load_checkpoint_payload(path):
    """Load a trusted local complete checkpoint; never accepts model-only weights."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError("Expected a native GPC complete training checkpoint")
    return payload


def restore_checkpoint(path, model, optimizer, scaler, scheduler, generator, order, config, identity):
    payload = load_checkpoint_payload(path)
    if payload["config"] != config or payload["identity"] != identity:
        raise ValueError("Resume config/source/data/split identity differs")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    scheduler.load_state_dict(payload["scheduler"])
    order.load_state_dict(payload["order"])
    restore_rng(payload["rng"], generator)
    return deepcopy(payload["progress"])


def normalized_config(config):
    value = deepcopy(config)
    defaults = dict(seed=5478, effective_batch=16, microbatch=1, lr=1e-4, weight_decay=.01,
        amp=True, validation_every=250, save_every=250, val_samples=72, max_updates=5000,
        max_train_seconds=5400., max_artifact_bytes=3*1024**3, disk_reserve_bytes=256*1024**2,
        cache_bytes=32*1024**2, max_cuda_bytes=None, render_workers=2)
    for key, default in defaults.items():
        value.setdefault(key, default)
    if value.get("local_pilot", True):
        if value["seed"] != 5478 or value["effective_batch"] != 16 or value["microbatch"] not in (1, 2, 4):
            raise ValueError("Native pilot requires seed5478, effective16 and microbatch1/2/4")
    elif (type(value["seed"]) is not int or value["seed"] not in (5478, 5479, 5480)
          or type(value["microbatch"]) is not int or value["microbatch"] < 1
          or type(value["effective_batch"]) is not int or value["effective_batch"] < value["microbatch"]
          or value["effective_batch"] % value["microbatch"]):
        raise ValueError("Server requires an approved seed and divisible positive micro/effective batches")
    if value["lr"] != 1e-4 or value["weight_decay"] != .01:
        raise ValueError("Use the pinned native AdamW learning rate/weight decay")
    if type(value["render_workers"]) is not int or not 1 <= value["render_workers"] <= 16:
        raise ValueError("Use1..16 bounded render threads")
    for key in ("validation_every", "save_every", "val_samples", "max_updates", "max_artifact_bytes"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError("Positive integer setting required: " + key)
    if not 0 < float(value["max_train_seconds"]) < float("inf"):
        raise ValueError("Finite positive training time cap required")
    if value.get("local_pilot", True) and (value["max_updates"] > 5000 or value["max_train_seconds"] > 5400):
        raise ValueError("Local aggregate budget cannot exceed5000 attempts/5400seconds")
    return value


def data_identity(data_root):
    from .native_rgb_data import TRAIN_LETTERS, UNSEEN_LETTERS, verify_dataset
    root = Path(data_root).resolve()
    manifest, audit = verify_dataset(root, require_gate=True)
    digest = sha256(root / "manifest.json")
    if (audit.get("physics_collection_gate_passed") is not True
            or audit.get("dataset_manifest_sha256") != digest
            or audit.get("dataset_npz_sha256_verified") is not True):
        raise ValueError("Current hash-bound passed native collection audit required")
    splits = manifest.get("splits") or {name: sorted(entry["trajectory_id"] for entry in manifest["entries"]
        if entry["split"] == name) for name in ("train", "seen_val", "unseen_test")}
    if (not isinstance(splits, dict) or not all(name in splits for name in ("train", "seen_val", "unseen_test"))
            or set(manifest.get("train_letters", ())) != set(TRAIN_LETTERS)
            or set(manifest.get("unseen_letters", ())) != set(UNSEEN_LETTERS)):
        raise ValueError("Explicit seen-train/seen-validation/unseen-test separation required")
    return dict(manifest_sha256=digest, audit_sha256=sha256(root / "physics-audit.json"),
                split_sha256=hashlib.sha256(json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()).hexdigest())


def source_identity():
    files = ("tdyssey/native_rgb_training.py", "tdyssey/native_rgb_data.py", "tdyssey/native_rgb_env.py",
             "tdyssey/native_rgb_evaluation.py", "tdyssey/raw_world_model_training.py",
             "tdyssey/gpc_world_model.py", "tdyssey/official_world_model.py", "tdyssey/official.py",
             "tdyssey/envs/official.py", "tdyssey/letters.py", "tdyssey/geometry.py", "tdyssey/source_export.py",
             "manifests/gpc_world_model.json", "manifests/official_dp.json")
    return dict(files={name: sha256(ROOT / name) for name in files},
                original_gpc_commit=world_model_spec()["source_commit"])


def tensor_hashes(model):
    return {name: hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, parameter in model.named_parameters()}


def parameter_update_report(model, initial):
    current = tensor_hashes(model)
    changed = [name for name in current if current[name] != initial[name]]
    return dict(trainable_parameter_tensors=len(current), changed_parameter_tensors=len(changed),
        all_parameter_tensors_changed=len(changed) == len(current), changed=changed,
        unchanged=[name for name in current if current[name] == initial[name]])


def _batch(dataset, indices, device):
    import numpy as np
    import torch
    items = [dataset[int(index)] for index in indices]
    return {name: torch.as_tensor(np.stack([item[name] for item in items]), device=device)
            for name in ("image", "action")}


def measure_denoising(model, dataset, indices, microbatch=1, seed=9478, amp=True, check=None):
    """Comparable fixed-noise loss; neither model mode nor training RNG changes."""
    import torch
    device = next(model.parameters()).device
    generator = make_generator(seed, device)
    previous = model.training
    model.eval()
    total = count = 0
    try:
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            for start in range(0, len(indices), microbatch):
                if check is not None:
                    check()
                selected = indices[start:start + microbatch]
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                    loss, _ = supervised_denoising_loss(model, _batch(dataset, selected, device), generator=generator)
                total += float(loss) * len(selected)
                count += len(selected)
    finally:
        model.train(previous)
    if count == 0:
        raise ValueError("No fixed loss samples")
    return total / count


def train(config, output, data_root, source_root, init_checkpoint, *, resume=None, execute=False,
          phase="main", updates=None, indices=None, validation_callback=None, physics_source_root=None):
    """Train H1 only. Optional H4 validation callback must use seen validation.

    `updates` is the target total updates of this run, useful for engineering or
    tiny fits; it does not change the aggregate ledger or scheduler horizon.
    """
    if execute is not True:
        raise PermissionError("Explicit native RGB training execution required")
    config = normalized_config(config)
    if physics_source_root is None:
        physics_source_root = config.get("physics_source_root")
    if physics_source_root is None:
        raise ValueError("Explicit official physics_source_root is required separately from GPC source_root")
    output = Path(output).resolve()
    artifact_root = Path(config.get("artifact_root", output.parent)).resolve()
    ledger_path = Path(config.get("ledger_path", artifact_root / "native-rgb-cost-ledger.json")).resolve()
    if not output.is_relative_to(artifact_root):
        raise ValueError("Training output must be inside its artifact budget root")
    if config["max_artifact_bytes"] > 3*1024**3 and config.get("local_pilot", True):
        raise ValueError("Local native artifact ceiling is3GiB")
    target = config["max_updates"] if updates is None else updates
    if type(target) is not int or not 0 <= target <= config["max_updates"]:
        raise ValueError("Run target must fit the aggregate update cap")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError("Existing native run requires exact --resume")
    identity = dict(data=data_identity(data_root), source=source_identity(),
        initialization=world_model_spec()["checkpoint"], selected_indices=indices,
        objective=deepcopy(OBJECTIVE), phase=str(phase), sampling_policy=SAMPLING_POLICY)
    ledger = CostLedger(ledger_path, config["max_updates"], config["max_train_seconds"])
    stage = ledger.start(phase)
    model = optimizer = scaler = scheduler = generator = source = renderer = None
    failure = None
    progress = dict(updates=0, samples=0, best_validation_loss=None, best_selection_score=None,
                    validation=[], phase=str(phase), training_seconds=0.)
    last_info = best_info = None
    clean_boundary = True
    try:
        write_source_snapshot(output, identity)
        import numpy as np
        import torch
        from .native_rgb_data import NativeRGBDataset
        if not torch.cuda.is_available():
            raise RuntimeError("Native pilot training requires CUDA; no automatic CPU fallback")
        torch.set_num_threads(2)
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        dataset = NativeRGBDataset(data_root, "train", source_root=physics_source_root, cache_bytes=config["cache_bytes"] // 4)
        validation = NativeRGBDataset(data_root, "seen_val", source_root=physics_source_root, cache_bytes=config["cache_bytes"] // 4)
        if not len(dataset) or not len(validation):
            raise ValueError("Nonempty independent seen train and validation required")
        order = WindowOrder(dataset, config["seed"], indices)
        model, source = construct_model(source_root, init_checkpoint, trainable=True)
        if any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Every native GPC parameter must be trainable")
        device = next(model.parameters()).device
        optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_updates"], eta_min=config["lr"] * .1)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]), init_scale=128.)
        generator = make_generator(config["seed"], device)
        if resume is not None:
            progress = restore_checkpoint(resume, model, optimizer, scaler, scheduler, generator, order, config, identity)
        else:
            progress["initial_parameter_sha256"] = tensor_hashes(model)
        previous_training_seconds = float(progress["training_seconds"])
        micro = config["microbatch"]
        fixed = list(indices) if indices is not None else np.linspace(0, len(dataset) - 1, min(16, len(dataset)), dtype=int).tolist()
        val_indices = np.linspace(0, len(validation) - 1, min(config["val_samples"], len(validation)), dtype=int).tolist()
        torch.cuda.reset_peak_memory_stats(device)
        renderer = BatchRenderer(dataset, config["render_workers"], config["cache_bytes"] // 2,
                                 config["effective_batch"])

        def check():
            stage.heartbeat()
            if (artifact_root / "STOP").exists() or (output / "STOP").exists():
                raise BudgetExceeded("STOP file requested termination")
            used = disk_bytes(artifact_root)
            if used > config["max_artifact_bytes"] - config["disk_reserve_bytes"]:
                raise BudgetExceeded("Native RGB artifact reserve reached")
            import shutil
            if shutil.disk_usage(artifact_root).free < config["disk_reserve_bytes"]:
                raise BudgetExceeded("Native RGB free-disk reserve reached")
            if config["max_cuda_bytes"] is not None and torch.cuda.max_memory_reserved(device) > config["max_cuda_bytes"]:
                raise BudgetExceeded("Native GPC measured CUDA memory ceiling reached")

        def save(name):
            check()
            return save_training_checkpoint(output / name, model, optimizer, scaler, scheduler, generator, order,
                                            progress, config, identity)

        def validate():
            nonlocal best_info
            check()
            measured = measure_denoising(model, validation, val_indices, micro, seed=config["seed"] + 400000,
                                         amp=config["amp"], check=check)
            row = dict(update=progress["updates"], denoising_loss=measured)
            if validation_callback is not None:
                was_training = model.training
                callback_rng = capture_rng(generator)
                model.eval()
                try:
                    with torch.inference_mode():
                        row.update(validation_callback(model, source, deepcopy(progress)))
                finally:
                    model.train(was_training)
                    restore_rng(callback_rng, generator)
            score = float(row.get("selection_score", measured))
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite seen-validation selection score")
            progress["validation"].append(row)
            if progress["best_selection_score"] is None or score < progress["best_selection_score"]:
                progress["best_selection_score"] = score
                progress["best_validation_loss"] = measured
                best_info = save("best.pt")
            atomic_json(output / "validation.json", progress["validation"])

        if progress["updates"] == 0:
            progress["fixed_train_loss_before"] = measure_denoising(model, dataset, fixed, micro,
                seed=config["seed"] + 300000, amp=config["amp"], check=check)
        with (output / "history.jsonl").open("a", encoding="utf-8", buffering=1) as log:
            while progress["updates"] < target:
                check()
                if ledger.snapshot()["charged_updates"] >= config["max_updates"]:
                    progress["stop_reason"] = "aggregate_optimizer_attempt_cap"
                    break
                stage.reserve_update()
                clean_boundary = False
                selected = order.take(config["effective_batch"])
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                for batch in renderer.batches(selected, micro, device):
                    check()
                    with torch.autocast("cuda", dtype=torch.float16, enabled=bool(config["amp"])):
                        loss, _ = supervised_denoising_loss(model, batch, generator=generator)
                    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Native residual loss must remain finite FP32")
                    scaler.scale(loss * micro / config["effective_batch"]).backward()
                    loss_sum += float(loss.detach()) * micro / config["effective_batch"]
                scaler.unscale_(optimizer)
                missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
                if missing:
                    raise FloatingPointError("Missing full-parameter native gradients: " + ",".join(missing))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if "first_gradient_norms" not in progress:
                    progress["first_gradient_norms"] = {name: float(parameter.grad.float().norm())
                        for name, parameter in model.named_parameters()}
                before_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < before_scale:
                    raise FloatingPointError("Native AMP overflow; attempted update remains charged")
                if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                    raise FloatingPointError("Nonfinite updated native GPC parameter; previous checkpoint retained")
                stage.complete_update()
                scheduler.step()
                progress["updates"] += 1
                progress["samples"] += len(selected)
                clean_boundary = True
                progress["training_seconds"] = previous_training_seconds + time.monotonic() - stage.started
                row = dict(update=progress["updates"], loss=loss_sum, gradient_norm=float(norm),
                    samples=progress["samples"], learning_rate=optimizer.param_groups[0]["lr"],
                    run_seconds=progress["training_seconds"], cuda_allocated=torch.cuda.max_memory_allocated(device),
                    cuda_reserved=torch.cuda.max_memory_reserved(device))
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if progress["updates"] % config["validation_every"] == 0:
                    validate()
                if progress["updates"] % config["save_every"] == 0:
                    last_info = save("last.pt")
        check()
        if not progress["validation"] or progress["validation"][-1]["update"] != progress["updates"]:
            validate()
        progress["fixed_train_loss_after"] = measure_denoising(model, dataset, fixed, micro,
            seed=config["seed"] + 300000, amp=config["amp"], check=check)
        progress["parameter_update_report"] = parameter_update_report(model, progress["initial_parameter_sha256"])
        last_info = save("last.pt")
        result = dict(status="complete", progress=progress, last=last_info,
            best=best_info or (dict(path=str(output / "best.pt")) if (output / "best.pt").exists() else None),
            identity=identity, config=config, peak_cuda_allocated=torch.cuda.max_memory_allocated(device),
            peak_cuda_reserved=torch.cuda.max_memory_reserved(device),
            native_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            no_unseen_evaluation=True, objective="original_phase_two_H1_residual_mse")
        atomic_json(output / "result.json", result)
        return result
    except BudgetExceeded as error:
        # A control/budget stop can discard a partial accumulation. It never
        # reports an optimizer update for that attempt, and its durable ledger
        # reservation remains spent. No partial-gradient resume is implied.
        progress["stop_reason"] = str(error)
        progress["incomplete_accumulation_discarded"] = not clean_boundary
        if model is not None and optimizer is not None and generator is not None:
            optimizer.zero_grad(set_to_none=True)
            progress["training_seconds"] = previous_training_seconds + time.monotonic() - stage.started
            if disk_bytes(artifact_root) < config["max_artifact_bytes"] - 128*1024**2:
                last_info = save_training_checkpoint(output / "last.pt", model, optimizer, scaler, scheduler,
                    generator, order, progress, config, identity)
        result = dict(status="stopped", progress=progress, last=last_info, identity=identity,
                      config=config, no_unseen_evaluation=True)
        atomic_json(output / "result.json", result)
        return result
    except BaseException as error:
        failure = error
        atomic_json(output / "error.json", dict(error=type(error).__name__ + ": " + str(error),
                    progress=progress, identity=identity, last=last_info))
        raise
    finally:
        if renderer is not None:
            renderer.close()
        model = optimizer = scaler = scheduler = generator = source = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except BaseException as cleanup:
            if failure is not None:
                failure.add_note("CUDA cleanup failed: " + str(cleanup))
            else:
                raise
        finally:
            stage.finish(("stopped:" + progress["stop_reason"]) if failure is None and progress.get("stop_reason")
                         else "complete" if failure is None else "failed:" + type(failure).__name__)
