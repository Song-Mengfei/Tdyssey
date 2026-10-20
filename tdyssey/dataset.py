"""Compact episode files with byte budgets, hashes, and frozen episode splits."""

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path

from .trajectories import Trajectory


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


class DatasetWriter:
    """Own a new directory; never overwrite an earlier collection.

    Byte budget covers published file contents. Filesystem block allocation and
    a temporary manifest replacement can use a small amount of additional space.
    A failed run is left explicitly incomplete so partial data is not mistaken
    for a finished dataset. No append/resume or automatic cleanup is performed.
    """

    def __init__(self, root, config, provenance):
        self.root, self.config = Path(root), config
        self.manifest = {"schema_version": 1, "status": "collecting", "config": asdict(config),
                         "provenance": provenance, "episodes": [], "splits": None,
                         "counts": {"reset_attempts": 0, "reset_physics_substeps": 0,
                                    "control_steps": 0, "control_physics_substeps": 0}}
        self.payload_bytes = 0
        self.closed = False
        self.root.mkdir(parents=True, exist_ok=False)
        self._manifest()

    def _manifest(self):
        payload = json_bytes(self.manifest)
        if self.payload_bytes + len(payload) > self.config.max_bytes:
            raise ValueError("Dataset byte limit exceeded")
        temporary = self.root / "manifest.tmp"
        temporary.write_bytes(payload)
        temporary.replace(self.root / "manifest.json")

    def add(self, episode):
        import numpy as np

        if self.closed:
            raise ValueError("Dataset is already closed")
        identifier = episode.trajectory.trajectory_id
        if any(row["trajectory_id"] == identifier for row in self.manifest["episodes"]):
            raise ValueError("Duplicate trajectory ID")
        prefix = f"episode-{len(self.manifest['episodes']):04d}"
        image_name, record_name = f"{prefix}.npz", f"{prefix}.json"
        buffer = io.BytesIO()
        np.savez_compressed(buffer, images=episode.images, agent_pos=episode.agent_pos, goal_pose=episode.goal_pose)
        image_data = buffer.getvalue()
        record = {"schema_version": 1, "trajectory": json.loads(episode.trajectory.to_json()),
                  "image_file": image_name, "image_sha256": digest(image_data),
                  "observations": {"layout": "T+1,CHW", "dtype": "uint8", "range": [0, 255],
                                   "image_shape": [3, 96, 96], "quantization": "round(normalized_image*255)"},
                  "reset_seed": episode.reset_seed, "action_seed": episode.action_seed,
                  "reset_attempts": episode.resets, "stop_reason": episode.stop_reason,
                  "step_diagnostics": episode.steps}
        record_data = json_bytes(record)
        entry = {"trajectory_id": identifier, "control_steps": len(episode.trajectory.transitions),
                 "observation_count": len(episode.images),
                 "contact_steps": sum(row["n_contacts"] > 0 for row in episode.steps),
                 "files": [{"name": image_name, "bytes": len(image_data), "sha256": digest(image_data)},
                           {"name": record_name, "bytes": len(record_data), "sha256": digest(record_data)}]}
        new_bytes = len(image_data) + len(record_data)
        # Leave room for the final split and run accounting before writing files.
        if self.payload_bytes + new_bytes + len(json_bytes(self.manifest)) + 131072 > self.config.max_bytes:
            raise ValueError("Dataset byte limit would be exceeded")
        for name, payload in ((image_name, image_data), (record_name, record_data)):
            with (self.root / name).open("xb") as stream:
                stream.write(payload)
        self.payload_bytes += new_bytes
        self.manifest["episodes"].append(entry)
        self._manifest()

    def finish(self, splits=None, error=None):
        if self.closed:
            raise ValueError("Dataset is already closed")
        if not error and len(self.manifest["episodes"]) != self.config.episodes:
            raise ValueError("Cannot complete a dataset with missing episodes")
        self.manifest["status"] = "failed" if error else "complete"
        self.manifest["splits"] = splits
        if error:
            self.manifest["error"] = str(error)
        self._manifest()
        self.closed = True


def load_episode(root, entry):
    """Verify one episode and load its arrays; used by the lazy training loader."""
    import numpy as np

    root = Path(root)
    payloads = {}
    for file in entry["files"]:
        name = file["name"]
        if Path(name).name != name:
            raise ValueError("Episode filenames must stay within the dataset directory")
        payload = (root / name).read_bytes()
        if len(payload) != file["bytes"] or digest(payload) != file["sha256"]:
            raise ValueError(f"File hash/size mismatch: {name}")
        payloads[name] = payload
    record_name = next(name for name in payloads if name.endswith(".json"))
    record = json.loads(payloads[record_name])
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("Unsupported episode schema")
    trajectory = Trajectory.from_json(json.dumps(record["trajectory"]))
    image_data = payloads[record["image_file"]]
    if digest(image_data) != record["image_sha256"]:
        raise ValueError("Image hash differs from episode record")
    with np.load(io.BytesIO(image_data), allow_pickle=False) as data:
        arrays = {name: data[name] for name in ("images", "agent_pos", "goal_pose")}
    count = len(trajectory.transitions)
    if (trajectory.trajectory_id != entry["trajectory_id"] or count != entry["control_steps"]
            or entry["observation_count"] != count + 1 or len(record["step_diagnostics"]) != count):
        raise ValueError("Episode metadata does not match control transitions")
    for transition, diagnostics in zip(trajectory.transitions, record["step_diagnostics"]):
        if (transition.terminated != diagnostics["upstream_terminated"]
                or transition.truncated != (diagnostics["upstream_truncated"] or diagnostics["collector_truncated"])):
            raise ValueError("Stored terminal flags differ from source/collector diagnostics")
    if not (trajectory.transitions[-1].terminated or trajectory.transitions[-1].truncated):
        raise ValueError("Collected episode has no recorded stopping condition")
    if arrays["images"].dtype != np.uint8 or arrays["images"].shape != (count + 1, 3, 96, 96):
        raise ValueError("Expected exactly T+1 uint8 image observations")
    if arrays["agent_pos"].shape != (count + 1, 2) or arrays["goal_pose"].shape != (count + 1, 3):
        raise ValueError("Proprioception/goal observation timing mismatch")
    for name in ("agent_pos", "goal_pose"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError("Non-finite observation")
    states = [trajectory.transitions[0].before, *(t.after for t in trajectory.transitions)]
    np.testing.assert_allclose(arrays["agent_pos"], [s.pusher_body.pose.position for s in states], rtol=0, atol=1e-9)
    np.testing.assert_allclose(arrays["goal_pose"],
                               [(*s.goal.position, s.goal.angle % (2 * np.pi)) for s in states], rtol=0, atol=1e-9)
    return trajectory, record, arrays


def read_manifest(root):
    """Validate episode membership and run accounting without loading image arrays."""
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["status"] != "complete":
        raise ValueError("Dataset must be complete with supported schema")
    entries = manifest["episodes"]
    if not entries or len(entries) != manifest["config"]["episodes"]:
        raise ValueError("Episode count differs from requested collection")
    ids = [entry["trajectory_id"] for entry in entries]
    splits = manifest["splits"]
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "model_test"}:
        raise ValueError("Missing frozen whole-episode splits")
    assigned = [identifier for part in splits.values() for identifier in part]
    if len(ids) != len(set(ids)) or len(assigned) != len(ids) or set(assigned) != set(ids):
        raise ValueError("Splits are overlapping, incomplete, or contain unknown episodes")
    for entry in entries:
        if type(entry["control_steps"]) is not int or entry["control_steps"] < 1 or entry["observation_count"] != entry["control_steps"] + 1:
            raise ValueError("Invalid episode length")
        names = [file["name"] for file in entry["files"]]
        if len(set(names)) != 2 or sorted(Path(name).suffix for name in names) != [".json", ".npz"]:
            raise ValueError("Episode must contain one record and one image archive")
        if any(Path(name).name != name for name in names):
            raise ValueError("Episode filenames must stay within the dataset directory")
    counts, rates = manifest["counts"], manifest["rates"]
    if (type(rates["simulation_hz"]) is not int or type(rates["control_hz"]) is not int
            or rates["simulation_hz"] <= 0 or rates["control_hz"] <= 0
            or rates["simulation_hz"] % rates["control_hz"]):
        raise ValueError("Invalid simulation/control rates")
    if (counts["control_steps"] != sum(entry["control_steps"] for entry in entries)
            or counts["reset_attempts"] < len(entries)
            or counts["reset_physics_substeps"] != counts["reset_attempts"]
            or counts["control_physics_substeps"] != counts["control_steps"] * rates["simulation_hz"] // rates["control_hz"]):
        raise ValueError("Physics/reset cost accounting mismatch")
    return manifest


def load_dataset(root):
    """Validate hashes, episode timing, frame alignment, and complete split coverage.

    Return manifest and episode records/arrays. Intended for the bounded pilot,
    not as a streaming loader for a large future training dataset.
    """
    root = Path(root)
    manifest = read_manifest(root)
    episodes = [load_episode(root, entry) for entry in manifest["episodes"]]
    resets = sum(sum(item.get("simulated", True) for item in episode[1]["reset_attempts"]) for episode in episodes)
    if manifest["counts"]["reset_attempts"] != resets:
        raise ValueError("Physics/reset cost accounting mismatch")
    return manifest, episodes
