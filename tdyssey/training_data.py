"""Frozen dataset catalogs and episode-safe visual training windows.

No windows are saved to disk. NumPy is imported only when a sample is requested;
PyTorch DataLoader can consume WindowDataset without an additional adapter.
"""

from bisect import bisect_right
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

from .dataset import digest, json_bytes, load_episode, read_manifest

SPLITS = ("train", "validation", "model_test")


@dataclass(frozen=True)
class WindowSpec:
    history: int = 2
    future: int = 4

    def __post_init__(self):
        for name in ("history", "future"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 32:
                raise ValueError(f"{name} must be an integer in [1, 32]")

    def anchors(self, control_steps):
        if type(control_steps) is not int or control_steps < 0:
            raise ValueError("Control-step count must be a nonnegative integer")
        return range(self.history - 1, control_steps - self.future + 1)


def initial_state_hash(trajectory):
    # Group repeated starting scenes even if run length, mode, or actions differ.
    # Otherwise repeated collection can leak the same scene across data splits.
    return digest(json_bytes(asdict(trajectory.transitions[0].before)))


def check_membership(rows):
    ids, initial_splits = set(), {}
    for row in rows:
        if row["trajectory_id"] in ids:
            raise ValueError("Duplicate trajectory across dataset runs")
        ids.add(row["trajectory_id"])
        split = row["split"]
        if split not in SPLITS:
            raise ValueError("Unknown dataset split")
        group = row["initial_state_sha256"]
        if group in initial_splits and initial_splits[group] != split:
            raise ValueError("A repeated initial state crosses dataset splits")
        initial_splits[group] = split


def write_catalog(roots, destination):
    """Verify every episode, preserve existing splits, and freeze manifest hashes.

    Catalog paths are relative to the catalog file. Repeated IDs are rejected;
    repeated initial scenes must stay in the same split. No implicit reshuffling.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    roots = [Path(root).resolve() for root in roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Choose one or more distinct dataset roots")
    runs, rows, rates = [], [], None
    for root in roots:
        manifest = read_manifest(root)
        if rates is not None and manifest["rates"] != rates:
            raise ValueError("Combined datasets must use the same simulation/control rates")
        rates = manifest["rates"]
        membership = {identifier: split for split, ids in manifest["splits"].items() for identifier in ids}
        entries, resets = [], 0
        for entry in manifest["episodes"]:
            trajectory, record, _ = load_episode(root, entry)
            if not math.isclose(trajectory.control_dt, 1 / rates["control_hz"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Trajectory timing differs from recorded control rate")
            resets += len(record["reset_attempts"])
            if trajectory.transitions[0].before.shape != manifest["config"]["shape"]:
                raise ValueError("Collected shape differs from run configuration")
            contact_steps = sum(row["n_contacts"] > 0 for row in record["step_diagnostics"])
            if contact_steps != entry["contact_steps"]:
                raise ValueError("Contact summary differs from episode diagnostics")
            row = {"trajectory_id": trajectory.trajectory_id, "split": membership[trajectory.trajectory_id],
                   "initial_state_sha256": initial_state_hash(trajectory),
                   "control_steps": entry["control_steps"], "contact_steps": contact_steps}
            entries.append(row)
        if resets != manifest["counts"]["reset_attempts"]:
            raise ValueError("Reset accounting differs from episode records")
        runs.append({"path": os.path.relpath(root, destination.parent),
                     "manifest_sha256": digest((root / "manifest.json").read_bytes()), "episodes": entries})
        rows.extend(entries)
    check_membership(rows)
    catalog = {"schema_version": 1, "rates": rates, "runs": runs,
               "split_protocol": "preserve source whole-episode splits; no implicit reassignment"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(json_bytes(catalog))
    return catalog


def read_catalog(path):
    path = Path(path).resolve()
    catalog = json.loads(path.read_text())
    if type(catalog["schema_version"]) is not int or catalog["schema_version"] != 1 or not catalog["runs"]:
        raise ValueError("Unsupported or empty training catalog")
    roots, manifests, rows = [], [], []
    for run in catalog["runs"]:
        if Path(run["path"]).is_absolute():
            raise ValueError("Catalog dataset paths must be relative")
        root = (path.parent / run["path"]).resolve()
        if root in roots:
            raise ValueError("Duplicate dataset root in catalog")
        if digest((root / "manifest.json").read_bytes()) != run["manifest_sha256"]:
            raise ValueError("Dataset manifest changed after catalog was frozen")
        manifest = read_manifest(root)
        if manifest["rates"] != catalog["rates"]:
            raise ValueError("Catalog simulation/control rates differ from source")
        entries = {entry["trajectory_id"]: entry for entry in manifest["episodes"]}
        if len(run["episodes"]) != len(entries) or {r["trajectory_id"] for r in run["episodes"]} != set(entries):
            raise ValueError("Catalog episode membership differs from source")
        for row in run["episodes"]:
            identifier = row["trajectory_id"]
            entry = entries[identifier]
            if (row["split"] not in SPLITS or identifier not in manifest["splits"][row["split"]]
                    or row["control_steps"] != entry["control_steps"] or row["contact_steps"] != entry["contact_steps"]):
                raise ValueError("Catalog split/summary differs from source")
        roots.append(root)
        manifests.append(manifest)
        rows.extend(run["episodes"])
    check_membership(rows)
    return catalog, roots, manifests


class WindowDataset:
    """Map-style training data with one decoded episode cached per instance.

    History ends at observation a. Future action k uses action[a+k] and predicts
    observation[a+k+1]. Only complete windows are emitted; there is no padding.
    Default model observations are image/proprioception/goal, never object state.
    Privileged reward/contact/overlap targets are grouped separately under labels.
    """

    def __init__(self, catalog_path, split="train", spec=None):
        if split not in SPLITS:
            raise ValueError("Choose train, validation, or model_test")
        self.spec, self.split = spec or WindowSpec(), split
        if not isinstance(self.spec, WindowSpec):
            raise ValueError("Expected a WindowSpec")
        self.catalog, self.roots, self.manifests = read_catalog(catalog_path)
        self.episodes, self.offsets = [], [0]
        for run_index, run in enumerate(self.catalog["runs"]):
            entries = {e["trajectory_id"]: e for e in self.manifests[run_index]["episodes"]}
            for row in run["episodes"]:
                if row["split"] == split:
                    self.episodes.append((run_index, row, entries[row["trajectory_id"]]))
                    self.offsets.append(self.offsets[-1] + len(self.spec.anchors(row["control_steps"])))
        self._cache_key, self._cache = None, None

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        import numpy as np

        if type(index) is not int:
            raise TypeError("Window index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.offsets, index) - 1
        run_index, row, entry = self.episodes[episode_index]
        if self._cache_key != episode_index:
            root = self.roots[run_index]
            if digest((root / "manifest.json").read_bytes()) != self.catalog["runs"][run_index]["manifest_sha256"]:
                raise ValueError("Dataset manifest changed after catalog was frozen")
            loaded = load_episode(root, entry)
            if initial_state_hash(loaded[0]) != row["initial_state_sha256"]:
                raise ValueError("Initial scene differs from frozen catalog")
            self._cache_key, self._cache = episode_index, loaded
        trajectory, record, arrays = self._cache
        anchor = self.spec.history - 1 + index - self.offsets[episode_index]
        history = slice(anchor - self.spec.history + 1, anchor + 1)
        future = slice(anchor + 1, anchor + self.spec.future + 1)
        transitions = trajectory.transitions[anchor:anchor + self.spec.future]
        diagnostics = record["step_diagnostics"][anchor:anchor + self.spec.future]

        def observations(selection):
            return {"image": arrays["images"][selection].astype(np.float32) / 255,
                    "agent_pos": arrays["agent_pos"][selection].astype(np.float32),
                    "goal_pose": arrays["goal_pose"][selection].astype(np.float32)}

        labels = {"reward": np.asarray([t.reward for t in transitions], dtype=np.float32),
                  "terminated": np.asarray([t.terminated for t in transitions], dtype=np.bool_),
                  "truncated": np.asarray([t.truncated for t in transitions], dtype=np.bool_)}
        for key, dtype in (("n_contacts", np.int64), ("iou", np.float32), ("coverage", np.float32),
                           ("upstream_terminated", np.bool_), ("upstream_truncated", np.bool_),
                           ("collector_truncated", np.bool_)):
            labels[key] = np.asarray([d[key] for d in diagnostics], dtype=dtype)
        return {"episode_id": trajectory.trajectory_id, "anchor": anchor,
                "shape": trajectory.transitions[0].before.shape, "control_dt": trajectory.control_dt,
                "observations": observations(history), "actions": np.asarray([t.action for t in transitions], dtype=np.float32),
                "targets": observations(future), "labels": labels}

    def summary(self):
        steps = sum(row["control_steps"] for _, row, _ in self.episodes)
        contacts = sum(row["contact_steps"] for _, row, _ in self.episodes)
        return {"split": self.split, "episodes": len(self.episodes), "windows": len(self),
                "control_steps": steps, "contact_steps": contacts,
                "contact_step_fraction": contacts / steps if steps else None,
                "window_spec": asdict(self.spec), "cache": "one decoded episode; no disk window cache"}
