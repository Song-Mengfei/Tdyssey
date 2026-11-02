"""Eight sealed TRAIN windows; metadata preparation has no array/runtime imports.

Only load_training_window decodes RGB, using the already reviewed selective NPY
reader. No default dataset traversal, branch/B2 access, or whole-episode decode.
Measurement and display helpers are lazy and never alter model inputs or gates.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path

FIXED_SELECTION = {
    "path": "artifacts/gpc-checkpoint500-offline-diagnosis-20261026/fixed-samples.json",
    "bytes": 252831,
    "sha256": "193096b5f23df1bec4e8f2ed73df5d15fbc4f2ef9c3edc571ad9701eb61a7ed7",
}
CURSORS = (24, 26, 0, 2, 31, 33, 1, 3)
SHAPES = ("T",) * 4 + ("L",) * 4
CONTACTS = (False, False, True, True, False, False, True, True)
PANEL_MAX_BYTES = 384 * 1024


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _helper():
    # Importing this existing script imports the standard library only.
    from scripts import gpc_checkpoint500_offline_pixels
    return gpc_checkpoint500_offline_pixels


def _identity(item):
    if (not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}
            or not isinstance(item["path"], str) or type(item["bytes"]) is not int
            or item["bytes"] <= 0 or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in item["sha256"])):
        raise ValueError("Invalid sealed source identity")
    return item


def _read_json(inputs, item):
    return json.loads(inputs.read(_identity(item)))


def _context(root):
    inputs = _helper().Inputs(root)
    manifest = _read_json(inputs, FIXED_SELECTION)
    windows = manifest["train_windows"]
    if (len(windows) != 8 or tuple(w["cursor"] for w in windows) != CURSORS
            or tuple(w["shape"] for w in windows) != SHAPES
            or tuple(w["contact"] for w in windows) != CONTACTS
            or any(type(w["cursor"]) is not int or type(w["contact"]) is not bool for w in windows)):
        raise ValueError("The fixed eight-window membership/order differs")
    # Deliberately do not iterate bindings: those include B2 and regression data.
    data = _read_json(inputs, manifest["bindings"]["data"])
    schedule = _read_json(inputs, manifest["bindings"]["schedule"])
    roles = {row["trajectory_id"]: row["role"] for row in data["episodes"]}
    fields = ("cursor", "step", "shape", "trajectory_id", "initial_state_sha256",
              "anchor", "dataset_index", "stratum", "contact", "rotation_degrees")
    for window in windows:
        if roles.get(window["trajectory_id"]) != "train":
            raise ValueError("Only source-declared TRAIN windows are permitted")
        entry = schedule["entries"][window["cursor"]]
        if any(entry.get(key) != window.get(key) for key in fields):
            raise ValueError("Window cursor/source index differs from the frozen schedule")
        record, payload = _identity(window["record"]), _identity(window["payload"])
        parent = Path(record["path"]).parent
        if (parent.as_posix() not in {
                "artifacts/gpc-data-stage-a-20261026/T-B",
                "artifacts/gpc-data-stage-a-20261026/L-A",
                "artifacts/gpc-data-stage-a-20261026/L-B"}
                or Path(record["path"]).suffix != ".json"
                or payload["path"] != str(Path(record["path"]).with_suffix(".npz")).replace("\\", "/")):
            raise ValueError("Selected record/payload path differs from its TRAIN source")
    return inputs, manifest


def _make_spec(root, window, inputs):
    record = _read_json(inputs, window["record"])
    trajectory = record["trajectory"]
    anchor = window["anchor"]
    transitions = trajectory["transitions"]
    if (trajectory["trajectory_id"] != window["trajectory_id"]
            or trajectory["action_semantics"] != "world_frame_position_target_pixels"
            or not math.isclose(trajectory["control_dt"], .1, rel_tol=0, abs_tol=1e-12)
            or record["image_file"] != Path(window["payload"]["path"]).name
            or record["image_sha256"] != window["payload"]["sha256"]
            or type(anchor) is not int or not 3 <= anchor < len(transitions)):
        raise ValueError("TRAIN record identity, timing, or five-frame anchor differs")
    initial = transitions[0]["before"]
    initial_bytes = (json.dumps(initial, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    if _sha(initial_bytes) != window["initial_state_sha256"]:
        raise ValueError("TRAIN initial-scene identity differs")
    selected = transitions[anchor-3:anchor+1]
    states = [step["before"] for step in selected] + [selected[-1]["after"]]
    actions = []
    for index, step in enumerate(selected):
        action = step["action"]
        if (step["before"]["shape"] != window["shape"] or step["after"]["shape"] != window["shape"]
                or step["after"]["control_step"] != step["before"]["control_step"] + 1
                or step["after"] != states[index+1] or step["after"]["pusher_target"] != action
                or not isinstance(action, list) or len(action) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 511 for v in action)):
            raise ValueError("Four real TRAIN controls/states are not aligned")
        actions.append(deepcopy(action))
    diagnostics = record["step_diagnostics"][anchor]
    if (diagnostics["n_contacts"] > 0) != window["contact"]:
        raise ValueError("Frozen contact stratum differs from source TRAIN metadata")
    spec = dict(kind="gpc_conditioning_train_window.v1", repository_root=str(Path(root).resolve()),
        fixed_selection=deepcopy(FIXED_SELECTION), window_id=f"train-cursor-{window['cursor']:03d}",
        selection=deepcopy(window), role="train", cursor=window["cursor"], shape=window["shape"],
        contact=window["contact"], trajectory_id=window["trajectory_id"],
        scene_sha256=window["initial_state_sha256"], anchor=anchor, dataset_index=window["dataset_index"],
        record=deepcopy(window["record"]), payload=deepcopy(window["payload"]),
        image_indices=list(range(anchor-3, anchor+2)), action_indices=list(range(anchor-3, anchor+1)),
        actions_absolute=actions, latest_state=deepcopy(states[-2]), target_state=deepcopy(states[-1]),
        goal=deepcopy(states[-2]["goal"]), target_truth=deepcopy(diagnostics),
        target_supervision="fifth real RGB; state/contact metadata is posthoc only")
    spec["spec_sha256"] = _sha(_canonical(spec))
    return spec


def _check_seal(spec):
    if (not isinstance(spec, dict) or spec.get("kind") != "gpc_conditioning_train_window.v1"
            or spec.get("role") != "train" or spec.get("cursor") not in CURSORS):
        raise ValueError("Not one of the fixed TRAIN windows")
    plain = {key: value for key, value in spec.items() if key != "spec_sha256"}
    if _sha(_canonical(plain)) != spec.get("spec_sha256"):
        raise ValueError("Prepared TRAIN window metadata changed")


def prepare_training_windows(root):
    """Metadata only: three bound metadata files and six unique TRAIN records."""
    inputs, manifest = _context(root)
    windows = [_make_spec(root, window, inputs) for window in manifest["train_windows"]]
    def counts(rows):
        return dict(windows=len(rows), trajectories=len({r["trajectory_id"] for r in rows}),
            scenes=len({r["scene_sha256"] for r in rows}),
            contact_windows=sum(r["contact"] for r in rows),
            noncontact_windows=sum(not r["contact"] for r in rows))
    return dict(kind="gpc_conditioning_training_inputs.v1", repository_root=str(Path(root).resolve()),
        fixed_selection=deepcopy(FIXED_SELECTION), identities=deepcopy(inputs.identities), windows=windows,
        denominators=dict(total=counts(windows), by_shape={s: counts([w for w in windows if w["shape"] == s])
                                                        for s in ("T", "L")}),
        scope=dict(array_decodes=0, payload_opens=0, B2_reads=0, held_payload_opens=0,
                   development_payload_opens=0, selection="fixed eight TRAIN windows; no resampling"))


def load_training_window(spec):
    """Future-only selective decode of exactly five RGB rows and four JSON actions.

    Return images (T1 once), raw_images, actions (already normalized), read_receipt.
    This intentionally rechecks TRAIN role/membership before opening any payload.
    """
    _check_seal(spec)
    inputs, manifest = _context(spec["repository_root"])
    window = next((w for w in manifest["train_windows"] if w["cursor"] == spec["cursor"]), None)
    if window is None or _make_spec(spec["repository_root"], window, inputs) != spec:
        raise ValueError("Prepared window differs from sealed TRAIN source")
    import numpy as np
    from .wm_compat_transform import transform_t1
    payload = inputs.read(spec["payload"])
    stored, receipt = _helper().read_npy_window(payload, "images.npy", spec["anchor"]-3, 5, (3, 96, 96), "u")
    raw = stored.astype(np.float32) / np.float32(255)
    images = np.stack([transform_t1(frame)[0] for frame in raw])
    absolute = np.asarray(spec["actions_absolute"], dtype=np.float32)
    actions = absolute / np.float32(511) * np.float32(2) - np.float32(1)
    return dict(images=images, raw_images=raw, actions=actions,
        read_receipt=dict(window_id=spec["window_id"], payload=deepcopy(spec["payload"]),
            selected_member=receipt, image_frames=5, action_rows=4, t1_applications=5,
            image_indices=deepcopy(spec["image_indices"]), action_indices=deepcopy(spec["action_indices"]),
            identities=deepcopy(inputs.identities), unselected_numeric_frames_decoded=0,
            actions_normalized=True, held_payload_opens=0))


def measure_h1(raw, spec, target_t1):
    """Future CPU measurement; preserve invalid/null gates and raw predictions."""
    _check_seal(spec)
    import numpy as np
    from . import wm_image_measurement_v2 as measurement
    from .wm_compat_engine import compact_measurement
    for name, value in (("raw", raw), ("target_t1", target_t1)):
        if (not isinstance(value, np.ndarray) or value.shape != (3, 96, 96)
                or value.dtype != np.float32 or not np.isfinite(value).all()):
            raise ValueError("Expected finite float32 CHW96 " + name)
    if (target_t1 < 0).any() or (target_t1 > 1).any():
        raise ValueError("Target must already be real T1 RGB [0,1]")
    full = measurement.decode(raw, spec["shape"], spec["goal"])
    chromatic, _ = _helper().pixel_diagnostics(raw, measurement)
    pusher = deepcopy(full["pusher"])
    truth_position = spec["target_state"]["pusher_body"]["pose"]["position"]
    error = math.dist(pusher["position"], truth_position) if pusher["valid"] and pusher.get("position") is not None else None
    fields = ("roi", "object_alpha_mass", "alpha_mass_in_roi", "alpha_mass_outside_roi",
              "alpha_fraction_outside_roi", "seed_component_count", "largest_component_mass_fraction",
              "raw_min", "raw_max", "raw_channel_means", "raw_out_of_range_channel_fraction")
    return dict(window_id=spec["window_id"], horizon=1, measurement=compact_measurement(full),
        pusher_diagnostics=pusher, pusher_position_error_pixels=error,
        truth_pusher_position=deepcopy(truth_position),
        chromatic={key: chromatic[key] for key in fields},
        raw_channel_means=[float(v) for v in raw.mean(axis=(1, 2))],
        target_mae=float(np.abs(raw-target_t1).mean()), target_mae_scope="raw prediction versus fifth real T1 RGB; no clipping",
        target_truth=deepcopy(spec["target_truth"]), thresholds_changed=False)


def training_panel_png(loaded, predictions_by_role):
    """Return one display-only 4-column/2-role PNG for the caller's bounded writer."""
    import numpy as np
    from PIL import Image, ImageDraw
    if set(predictions_by_role) != {"frozen", "adapted"}:
        raise ValueError("Panel requires both fixed weight roles")
    sheet = Image.new("RGB", (4*96, 2*120), "white")
    draw = ImageDraw.Draw(sheet)
    for row, role in enumerate(("frozen", "adapted")):
        arms = predictions_by_role[role]
        if set(arms) != {"A", "B"}:
            raise ValueError("Panel requires fixed A/B predictions")
        for col, (label, frame) in enumerate((("latest", loaded["images"][3]), ("truth", loaded["images"][4]),
                                              ("A", arms["A"]), ("B", arms["B"]))):
            if frame.shape != (3,96,96) or frame.dtype != np.float32 or not np.isfinite(frame).all():
                raise ValueError("Invalid panel RGB")
            pixels = np.rint(np.clip(frame, 0, 1)*255).astype(np.uint8).transpose(1, 2, 0)
            sheet.paste(Image.fromarray(pixels), (col*96, row*120))
            draw.text((col*96+2, row*120+98), role+" "+label, fill="black")
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    payload = buffer.getvalue()
    if len(payload) > PANEL_MAX_BYTES:
        raise ValueError("Fixed panel byte ceiling exceeded")
    return payload
