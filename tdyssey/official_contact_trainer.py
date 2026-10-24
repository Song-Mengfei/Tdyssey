"""Official-T denoiser updates sampled from all finished contact-balanced batches.

Each chunk draws train windows from the full train-split index and scores a
frozen validation subset. This does not train a policy. Image MSE versus a
still frame is logged only.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import time
from pathlib import Path

from .collection import integer
from .dataset import load_episode, read_manifest
from .gpc_denoiser_training import _image_name, _windows, anchors
from .motion_replay import object_moved
from .official_collection import ENVIRONMENT_FAMILY
from .policy import hash_file
from .region_weight import window_useful
from .trajectories import Trajectory

BATCH_NAME = re.compile(r"^batch-(\d+)$")
DEFAULT_POOL_WINDOWS = 64
DEFAULT_VAL_WINDOWS = 64
DEFAULT_UPDATES = 8
DEFAULT_SEED = 250000
REPORT_NAME = "report.json"
WEIGHTS_NAME = "denoiser.pt"
SIDECAR_NAME = "denoiser.json"


def batch_index(path):
    """Numeric suffix of a batch-* directory name."""
    match = BATCH_NAME.match(Path(path).name)
    if match is None:
        raise ValueError(f"Not an official batch directory: {path}")
    return int(match.group(1))


def discover_batches(root):
    """Finished official batch directories in numeric order. Incomplete dirs are skipped."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    found = []
    for path in root.iterdir():
        if not path.is_dir() or BATCH_NAME.match(path.name) is None:
            continue
        if not (path / "manifest.json").is_file():
            continue
        try:
            read_manifest(path)
        except ValueError:
            continue
        found.append(path)
    found.sort(key=batch_index)
    if not found:
        raise ValueError("No finished official batch directories")
    return found


def list_split_keys(batches, split):
    """Window keys for one frozen split. model_test is refused."""
    if split == "model_test":
        raise ValueError("Denoiser training must not read model_test")
    keys = []
    for batch in batches:
        root = Path(batch)
        manifest = read_manifest(root)
        if manifest["provenance"].get("environment_family") != ENVIRONMENT_FAMILY:
            raise ValueError("Contact training expects a finished official dataset")
        if split not in manifest["splits"]:
            raise ValueError(f"Official dataset is missing the {split} split")
        by_id = {entry["trajectory_id"]: entry for entry in manifest["episodes"]}
        test_ids = set(manifest["splits"]["model_test"])
        for identifier in manifest["splits"][split]:
            if identifier in test_ids:
                raise ValueError("Denoiser training must not read model_test")
            entry = by_id[identifier]
            for anchor in anchors(entry["control_steps"]):
                keys.append({"batch": root.name, "episode_id": identifier, "anchor": int(anchor)})
    keys.sort(key=lambda item: (batch_index(item["batch"]), item["episode_id"], item["anchor"]))
    return keys


def list_validation_keys(batches):
    """Validation-split window keys. model_test episode ids are never listed."""
    return list_split_keys(batches, "validation")


def list_train_keys(batches):
    """Train-split window keys from every finished batch. model_test stays closed."""
    return list_split_keys(batches, "train")


def freeze_validation_keys(batches, count=DEFAULT_VAL_WINDOWS, seed=DEFAULT_SEED):
    """Seeded 32 useful + 32 miss subset when episode JSON is present."""
    count = integer(count, "validation windows", 1, 512)
    keys = list_validation_keys(batches)
    if not keys:
        raise ValueError("Finished batches contain no validation windows")
    if len(keys) <= count:
        return keys
    flags = classify_validation_keys(batches, keys)
    if flags is None:
        return random.Random(seed).sample(keys, count)
    return stratify_validation_keys(keys, flags, count, seed)


def classify_validation_keys(batches, keys):
    """Useful/miss flags from episode JSON. None when records are missing."""
    cache = {}
    flags = []
    named = {path.name: path for path in batches}
    try:
        for key in keys:
            cache_key = (key["batch"], key["episode_id"])
            if cache_key not in cache:
                root = named[key["batch"]]
                manifest = read_manifest(root)
                entry = next(item for item in manifest["episodes"]
                             if item["trajectory_id"] == key["episode_id"])
                cache[cache_key] = _read_episode_record(root, entry)
            flags.append(_useful_from_record(cache[cache_key], key["anchor"]))
    except (FileNotFoundError, KeyError, StopIteration, ValueError, json.JSONDecodeError, OSError):
        return None
    return flags


def stratify_validation_keys(keys, flags, count, seed):
    """Take half useful and half miss, then fill from whichever class remains."""
    if len(keys) != len(flags):
        raise ValueError("Validation keys and useful flags must match")
    useful = [key for key, flag in zip(keys, flags) if flag]
    miss = [key for key, flag in zip(keys, flags) if not flag]
    rng = random.Random(seed)
    take_useful = min(count // 2, len(useful))
    take_miss = min(count - take_useful, len(miss))
    selected = (rng.sample(useful, take_useful) if take_useful else [])
    selected += rng.sample(miss, take_miss) if take_miss else []
    leftover = [key for key in keys if key not in selected]
    need = count - len(selected)
    if need and leftover:
        selected += rng.sample(leftover, min(need, len(leftover)))
    return selected


def _read_episode_record(root, entry):
    name = next(file["name"] for file in entry["files"] if file["name"].endswith(".json"))
    return json.loads((Path(root) / name).read_text())


def _useful_from_record(record, anchor):
    trajectory = Trajectory.from_json(json.dumps(record["trajectory"]))
    predicted = trajectory.transitions[anchor]
    n_contacts = integer(record["step_diagnostics"][anchor]["n_contacts"], "contacts", 0, 100000)
    return window_useful(object_moved(predicted.before, predicted.after), n_contacts)


def sample_train_keys(keys, count, rng):
    """Draw train windows from the full index. Short indexes return every key."""
    if not keys:
        raise ValueError("Finished batches contain no training windows")
    count = integer(count, "pool windows", 1, 512)
    if len(keys) <= count:
        return list(keys)
    return rng.sample(keys, count)


def load_selected_windows(datasets, keys, opened, allowed_splits=("validation",)):
    """Load only named windows. model_test files are never opened."""
    datasets = Path(datasets)
    if not keys:
        raise ValueError("Selected window keys are empty")
    allowed = tuple(allowed_splits)
    if not allowed or "model_test" in allowed:
        raise ValueError("Denoiser training must not read model_test")
    grouped = {}
    for key in keys:
        grouped.setdefault((key["batch"], key["episode_id"]), []).append(int(key["anchor"]))
    windows = []
    for (name, identifier), wanted in grouped.items():
        root = datasets / name
        manifest = read_manifest(root)
        if identifier in manifest["splits"]["model_test"]:
            raise ValueError("Denoiser training must not read model_test")
        if not any(identifier in manifest["splits"][split] for split in allowed):
            raise ValueError(f"Selected keys must name {' or '.join(allowed)} episodes")
        entry = next(item for item in manifest["episodes"] if item["trajectory_id"] == identifier)
        opened.append(_image_name(entry))
        trajectory, record, arrays = load_episode(root, entry)
        wanted_set = set(wanted)
        matched = []
        for window in _windows(trajectory, arrays, record):
            if window["anchor"] in wanted_set:
                windows.append(window)
                matched.append(window["anchor"])
        if not wanted_set <= set(matched):
            raise ValueError(f"Missing selected anchors in {name}:{identifier}")
    return windows


def train_contact_batches(datasets, output, checkpoint, gpc_root, resume=False,
                          pool_windows=DEFAULT_POOL_WINDOWS, validation_windows=DEFAULT_VAL_WINDOWS,
                          updates=DEFAULT_UPDATES, batch_size=4, seed=DEFAULT_SEED,
                          train_seconds=900, budget_seconds=0, max_updates=0, train_chunk=None):
    """Update one denoiser from globally sampled train windows. Ctrl+C keeps the last chunk."""
    datasets = Path(datasets)
    output = Path(output)
    checkpoint = Path(checkpoint)
    integer(pool_windows, "pool windows", 1, 512)
    integer(validation_windows, "validation windows", 1, 512)
    integer(updates, "updates", 1, 20)
    integer(batch_size, "batch size", 1, 8)
    integer(seed, "seed", 0, 2**32 - 1)
    integer(train_seconds, "train seconds", 1, 900)
    integer(budget_seconds, "budget seconds", 0, 86400)
    integer(max_updates, "max updates", 0, 1_000_000)
    batches = discover_batches(datasets)
    train_keys = list_train_keys(batches)
    continuing = _existing_run(output, resume)
    started = time.monotonic()
    if continuing:
        report = json.loads((output / REPORT_NAME).read_text())
        keys = list(report["validation_keys"])
        _check_stored_keys(keys, batches)
        current = output / WEIGHTS_NAME
        rounds = list(report.get("rounds", []))
        total_updates = int(report.get("updates", 0))
        prior_elapsed = float(report.get("elapsed_seconds", 0))
        source = report.get("source_checkpoint", str(checkpoint))
        print(json.dumps({"resumed": True, "updates": total_updates,
                          "train_windows": len(train_keys), "batches": len(batches)}), flush=True)
    else:
        if output.exists():
            raise FileExistsError(f"Training output already exists: {output}")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output.mkdir(parents=True)
        shutil.copy2(checkpoint, output / WEIGHTS_NAME)
        write_init_sidecar(output / SIDECAR_NAME, checkpoint)
        keys = freeze_validation_keys(batches, validation_windows, seed)
        current = output / WEIGHTS_NAME
        rounds, total_updates, prior_elapsed = [], 0, 0.0
        source = str(checkpoint)
    report = {"schema_version": 1, "kind": "official_t_contact_denoiser_training",
              "status": "running", "datasets": str(datasets),
              "batches": [path.name for path in batches], "validation_keys": keys,
              "train_windows": len(train_keys), "updates": total_updates, "rounds": rounds,
              "source_checkpoint": source, "model_test_used": False,
              "limits": {"pool_windows": pool_windows, "validation_windows": len(keys),
                         "updates": updates, "budget_seconds": budget_seconds,
                         "max_updates": max_updates},
              "elapsed_seconds": prior_elapsed}
    last_round_seconds = None
    try:
        validation = None if train_chunk is not None else load_selected_windows(datasets, keys, [])
        while True:
            if max_updates and total_updates >= max_updates:
                report["status"] = "stopped"
                report["stop_reason"] = "update cap"
                break
            if budget_seconds and rounds:
                remaining = budget_seconds - (time.monotonic() - started)
                if remaining <= 0 or (last_round_seconds is not None and remaining < last_round_seconds):
                    report["status"] = "stopped"
                    report["stop_reason"] = "time budget"
                    break
            chunk_seed = (seed + total_updates) % (2**32)
            sampled = sample_train_keys(train_keys, pool_windows, random.Random(chunk_seed))
            context = {"datasets": datasets, "output": output, "checkpoint": current,
                       "gpc_root": Path(gpc_root), "sampled_keys": sampled,
                       "train_keys": train_keys, "validation": validation,
                       "validation_keys": keys, "updates": updates, "batch_size": batch_size,
                       "seed": chunk_seed, "train_seconds": train_seconds}
            if train_chunk is None:
                context["training"] = load_selected_windows(
                    datasets, sampled, [], allowed_splits=("train",))
            round_started = time.monotonic()
            trained = (_default_chunk if train_chunk is None else train_chunk)(context)
            current = output / WEIGHTS_NAME
            total_updates += int(trained["updates"])
            last_round_seconds = time.monotonic() - round_started
            validation_row = trained["validation"]["adapted"]
            batch_names = sorted({key["batch"] for key in sampled}, key=batch_index)
            useful = validation_row.get("useful") or {}
            miss = validation_row.get("miss") or {}
            row = {"updates": total_updates, "validation_mse": validation_row["mse"],
                   "persistence_mse": validation_row["persistence_mse"],
                   "object_roi_mse": validation_row.get("object_roi_mse"),
                   "object_roi_persistence": validation_row.get("object_roi_persistence"),
                   "motion_mse": validation_row.get("motion_mse"),
                   "motion_persistence": validation_row.get("motion_persistence"),
                   "object_centroid_error_pixels": validation_row.get("object_centroid_error_pixels"),
                   "useful_windows": validation_row.get("useful_windows"),
                   "miss_windows": validation_row.get("miss_windows"),
                   "useful_motion_mse": useful.get("motion_mse"),
                   "useful_motion_persistence": useful.get("motion_persistence"),
                   "useful_object_roi_mse": useful.get("object_roi_mse"),
                   "useful_object_roi_persistence": useful.get("object_roi_persistence"),
                   "useful_centroid_error_pixels": useful.get("object_centroid_error_pixels"),
                   "useful_predicted_centroid_shift_pixels": useful.get("predicted_centroid_shift_pixels"),
                   "useful_actual_centroid_shift_pixels": useful.get("actual_centroid_shift_pixels"),
                   "miss_object_roi_mse": miss.get("object_roi_mse"),
                   "miss_object_roi_persistence": miss.get("object_roi_persistence"),
                   "miss_predicted_centroid_shift_pixels": miss.get("predicted_centroid_shift_pixels"),
                   "miss_actual_centroid_shift_pixels": miss.get("actual_centroid_shift_pixels"),
                   "previous_mse": trained["validation"]["before"]["mse"],
                   "weights_changed": trained["weights_changed"],
                   "model_test_used": trained["model_test_used"],
                   "sampled_windows": len(sampled),
                   "sampled_episodes": len({(key["batch"], key["episode_id"]) for key in sampled}),
                   "sampled_batches": len(batch_names),
                   "batches": _preview_names(batch_names)}
            rounds.append(row)
            report.update({"status": "running", "updates": total_updates, "rounds": rounds,
                           "elapsed_seconds": prior_elapsed + (time.monotonic() - started)})
            _write(output / REPORT_NAME, report)
            print(json.dumps(row), flush=True)
        report["elapsed_seconds"] = prior_elapsed + (time.monotonic() - started)
        report.update({"updates": total_updates, "rounds": rounds})
        _write(output / REPORT_NAME, report)
        return report
    except KeyboardInterrupt:
        report.update({"status": "interrupted", "updates": total_updates, "rounds": rounds,
                       "elapsed_seconds": prior_elapsed + (time.monotonic() - started)})
        _write(output / REPORT_NAME, report)
        return report
    except BaseException as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "updates": total_updates, "rounds": rounds,
                       "elapsed_seconds": prior_elapsed + (time.monotonic() - started)})
        _write(output / REPORT_NAME, report)
        raise


def write_init_sidecar(path, checkpoint):
    """Zero-update sidecar for copied-in weights. Do not inherit a source sidecar."""
    payload = {"updates": 0, "init_sha256": hash_file(checkpoint)}
    Path(path).write_text(json.dumps(payload) + "\n")
    return payload


def _existing_run(output, resume):
    output = Path(output)
    if not output.exists():
        return False
    if not resume:
        raise FileExistsError(f"Training output already exists: {output}")
    if not output.is_dir() or not (output / REPORT_NAME).is_file() or not (output / WEIGHTS_NAME).is_file():
        raise FileExistsError(f"Incomplete training output cannot be resumed: {output}")
    return True


def _check_stored_keys(keys, batches):
    names = {path.name for path in batches}
    for key in keys:
        if key["batch"] not in names:
            raise ValueError(f"Stored validation batch is missing: {key['batch']}")
        if key.get("episode_id") is None:
            raise ValueError("Stored validation keys are incomplete")


def _default_chunk(context):
    from .gpc_denoiser_training import GPCDenoiserTrainer

    trainer = GPCDenoiserTrainer(context["gpc_root"], context["checkpoint"],
                                 updates=context["updates"], batch_size=context["batch_size"],
                                 seed=context["seed"], seconds=context["train_seconds"])
    return trainer.run(context["datasets"], context["output"],
                       validation=context["validation"], training=context["training"])


def _preview_names(names):
    if len(names) <= 16:
        return list(names)
    return list(names[:8]) + ["..."] + list(names[-4:])


def _write(path, report):
    Path(path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
