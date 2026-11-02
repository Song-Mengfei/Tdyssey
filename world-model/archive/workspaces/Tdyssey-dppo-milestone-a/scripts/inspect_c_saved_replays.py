"""Inspect saved C evidence and export actual replay RGB; no simulation/model.

Only the two small selected-control replay files are deserialized, using CPU
weights_only=True. Checkpoints are described through an existing verified JSON
metadata report. This module imports no project runtime or tensor library until
the offline export is explicitly invoked.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "artifacts/milestone-c-20261024"
EVALUATION = ROOT / "artifacts/milestone-c-evaluation-20261024"
DEFAULT_OUTPUT = ROOT / "artifacts/milestone-c-offline-diagnosis-20261024/replays"


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def frame_mapping(controls, frame_count):
    """Validate the producer's initial-before + one-after-each-control layout.

    control_index is an episode-local completed-control index. It is not a
    whole-job/global control index, timestamp or denoising-time index.
    """
    if not controls or frame_count != len(controls) + 1:
        raise ValueError("Saved frames must contain one initial observation plus each control's next observation")
    indices = [record.get("control_index") for record in controls]
    if any(type(index) is not int or index < 1 for index in indices) or indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("Saved episode control indices must be explicit, positive and consecutive")
    return [{"frame_index": 0, "relation": "before", "episode_control_index": indices[0]}] + [
        {"frame_index": index + 1, "relation": "after", "episode_control_index": control_index}
        for index, control_index in enumerate(indices)]


def json_file(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def load_replay(path, shape, torch):
    data = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"schema_version", "kind", "shape", "reset_seed", "actor_version", "frames_uint8", "agent_pos", "actions", "controls"}
    if (not isinstance(data, dict) or set(data) != expected or data["schema_version"] != 1
            or data["kind"] != "tdyssey.C.selected_control_trace" or data["shape"] != shape):
        raise ValueError("Unexpected selected-control replay schema/shape")
    frames, positions, actions = (data[key] for key in ("frames_uint8", "agent_pos", "actions"))
    if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in (frames, positions, actions)):
        raise ValueError("Selected replay tensors must be CPU tensors")
    if frames.dtype != torch.uint8 or frames.ndim != 4 or tuple(frames.shape[1:]) != (3, 96, 96):
        raise ValueError("Expected saved uint8 RGB frames in N,3,96,96 order")
    mapping = frame_mapping(data["controls"], len(frames))
    if (tuple(positions.shape) != (len(frames), 2) or tuple(actions.shape) != (len(data["controls"]), 2)
            or not torch.isfinite(positions).all() or not torch.isfinite(actions).all()):
        raise ValueError("Agent positions/actions must have aligned finite two-coordinate rows")
    return data, mapping


def export_shape(data, mapping, output, Image, ImageDraw, ImageFont):
    shape = data["shape"]
    directory = output / shape
    directory.mkdir()
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=13)
    tile_width, tile_height, top, margin = 308, 350, 78, 12
    sheet = Image.new("RGB", (tile_width * 3, top + tile_height * 3 + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 8), f"C TRAIN collection | shape {shape} | first selected plan | actor:0", fill="black", font=font)
    draw.text((margin, 31), f"Scene {data['reset_seed']} | saved RGB 96x96, shown 3x nearest-neighbor", fill="black", font=small)
    draw.text((margin, 51), "Observation index is local to this saved trace. Contact = recorded n_contacts.", fill="black", font=small)
    images = []
    for index, (frame, link) in enumerate(zip(data["frames_uint8"], mapping)):
        rgb = frame.permute(1, 2, 0).contiguous().numpy()
        image = Image.fromarray(rgb, mode="RGB")
        image.save(directory / f"frame-{index:02d}.png")
        images.append(image)
        x, y = (index % 3) * tile_width + margin, top + (index // 3) * tile_height
        sheet.paste(image.resize((288, 288), resample=Image.Resampling.NEAREST), (x, y))
        draw.text((x, y + 290), f"obs[{index}] {link['relation']} episode control {link['episode_control_index']}", fill="black", font=small)
        if index:
            record = data["controls"][index - 1]
            draw.text((x, y + 310), f"coverage {record['coverage']:.5f} | contact {record['n_contacts']}", fill="black", font=small)
        else:
            draw.text((x, y + 310), "Initial observation; no prior saved control", fill="black", font=small)
    draw.text((margin, sheet.height - 30), "Actual pre-update training frames; not final-delta evaluation or a later failure case.", fill="black", font=small)
    sheet.save(output / f"contact-sheet-{shape}.png")
    tensor_metadata = {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype), "device": str(data[key].device)}
                       for key in ("frames_uint8", "agent_pos", "actions")}
    details = {"schema_version": 1, "method": "C training-sampler collection before the first update", "shape": shape,
               "scene_seed": data["reset_seed"], "actor_version": data["actor_version"],
               "payload_keys": sorted(data), "tensor_metadata": tensor_metadata,
               "frame_mapping": mapping, "controls": data["controls"],
               "agent_pos": data["agent_pos"].tolist(), "actions": data["actions"].tolist(),
               "index_semantics": "control_index is episode-local. obs[0] precedes the first saved control; obs[i+1] is the next observation after saved control[i]. This producer mapping is verified in scripts/run_c_policy.py:_selected_replay.",
               "not_available": ["per-frame numeric object pose", "per-substep contact history", "full episode", "later training plans", "final-delta or paired-evaluation frames"],
               "array_semantics": {"agent_pos": "recorded observation of pusher position; not inferred from action target",
                                   "actions": "recorded executed commanded two-coordinate targets; not actual object/pusher trajectories"}}
    json_file(output / f"replay-{shape}-metadata.json", details)
    return details, images


def evaluation_inventory(data):
    cases = data.get("cases", [])
    return {"kind": data.get("kind"), "schema_version": data.get("schema_version"), "method": data.get("method"),
        "top_level_fields": sorted(data), "case_count": len(cases), "status": data.get("status"),
        "recorded_control_rows": sum(len(case.get("steps", [])) for case in cases),
        "case_fields": sorted({key for case in cases for key in case}),
        "per_control_fields": sorted({key for case in cases for row in case.get("steps", []) for key in row}),
        "initial_physical_state_fields": sorted({key for case in cases for key in (case.get("initial_state") or {})}),
        "availability": {"full_executed_action_metric_sequences": True, "full_simulator_state_trajectories": False,
                         "numeric_object_pose": "initial state only", "numeric_pusher_pose": "initial state only",
                         "per_control_contact": False, "rgb_frames": False, "video": False,
                         "planning_latency": True, "denoising_chains": False},
        "case_lengths": [{key: case[key] for key in ("shape", "scene_seed", "policy_seed", "status", "success")} |
                         {"control_rows": len(case.get("steps", [])), "counted_controls": case.get("counts", {}).get("control_steps")}
                         for case in cases]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Derived output directory already exists; existing artifacts are never overwritten")
    inputs = {}

    def remember(path):
        inputs[str(path.relative_to(ROOT))] = identity(path)
        return path

    def read(path):
        remember(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    # No checkpoint/model deserialization: only these two tiny recorded traces.
    import torch
    from PIL import Image, ImageDraw, ImageFont
    traces = {}
    for shape in ("T", "L"):
        traces[shape] = load_replay(remember(TRAIN / f"replay-first-{shape}.pt"), shape, torch)
    training = read(TRAIN / "train-report-0000.json")
    checkpoint_metadata = read(ROOT / "artifacts/milestone-c-load-memory-20261024/checkpoint-metadata-verified.json")
    groups = {method: read(EVALUATION / (method + ".json")) for method in ("base_training", "delta_training")}
    pair = read(EVALUATION / "paired-comparison.json")
    remember(ROOT / "scripts/run_c_policy.py")
    remember(ROOT / "tdyssey/policy_stream.py")
    remember(Path(__file__).resolve())
    log_path = remember(TRAIN / "training.jsonl")
    log = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    resources = {}
    for label, path in (("training", ROOT / "artifacts/milestone-c-load-memory-20261024/resources-review-final.json"),
                        ("evaluation", ROOT / "artifacts/milestone-c-evaluation-20261024-meta/resources-review-final.json")):
        data = read(path)
        resources[label] = {key: data[key] for key in ("timing", "identity", "all_checks_passed") if key in data}
    output.mkdir(parents=True)
    metadata, images = {}, {}
    for shape, (data, mapping) in traces.items():
        metadata[shape], images[shape] = export_shape(data, mapping, output, Image, ImageDraw, ImageFont)
    selected = (0, 4, 8)
    overview = Image.new("RGB", (3 * 308, 2 * 352 + 90), "white")
    draw = ImageDraw.Draw(overview)
    font = ImageFont.load_default(size=15)
    draw.text((12, 8), "Actual saved C pre-update TRAIN frames | actor:0 | first selected T/L plans", fill="black", font=font)
    draw.text((12, 31), "Each image is a recorded observation; episode-local control labels come from saved records.", fill="black", font=ImageFont.load_default(size=13))
    for row, shape in enumerate(("T", "L")):
        for column, index in enumerate(selected):
            x, y = 12 + column * 308, 66 + row * 352
            overview.paste(images[shape][index].resize((288, 288), Image.Resampling.NEAREST), (x, y))
            link = metadata[shape]["frame_mapping"][index]
            draw.text((x, y + 293), f"{shape} obs[{index}]: {link['relation']} episode control {link['episode_control_index']}", fill="black", font=ImageFont.load_default(size=13))
            draw.text((x, y + 314), f"scene {metadata[shape]['scene_seed']} | not a later failure replay", fill="black", font=ImageFont.load_default(size=13))
    overview.save(output / "contact-sheet-overview.png")
    lost = [{key: case[key] for key in ("shape", "scene_seed", "policy_seed")} for case in pair["cases"]
            if case["baseline"]["success"] is True and case["delta"]["success"] is False]
    inventory = {"schema_version": 1, "kind": "tdyssey.C.offline_saved_replay_inventory",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Offline CPU weights-only load of two selected replay payloads, actual-image export and JSON inventory; no reconstruction, policy call or physical operation.",
        "runtime_imports": {"torch": torch.__version__, "pillow": Image.__version__, "project_runtime_imported": False,
                            "cuda_initialized": torch.cuda.is_initialized()},
        "saved_rgb_scope": {"plans": 2, "controls": sum(len(data["controls"]) for data, mapping in traces.values()),
                            "frames": sum(len(data["frames_uint8"]) for data, mapping in traces.values()),
                            "whole_training_controls": training["result"]["controls"], "actor_versions": sorted({data["actor_version"] for data, mapping in traces.values()}),
                            "whole_training_fraction": sum(len(data["controls"]) for data, mapping in traces.values()) / training["result"]["controls"]},
        "selected_replays": {shape: {key: value for key, value in summary.items() if key not in ("actions", "agent_pos", "controls")}
                             for shape, summary in metadata.items()},
        "training_log": {"event_counts": dict(Counter(row.get("event") for row in log)),
                         "availability": {"batch_and_update_summaries": True, "full_per_control_trajectory": False,
                                          "per_shape_aggregate_contacts": True, "full_rgb_or_object_pose_trajectory": False}},
        "evaluation": {method: evaluation_inventory(group) for method, group in groups.items()},
        "checkpoint_json_metadata": {"source": "artifacts/milestone-c-load-memory-20261024/checkpoint-metadata-verified.json",
            "report_fields": sorted(checkpoint_metadata),
            "checkpoints": [{key: record.get(key) for key in ("path", "sha256", "bytes", "all_tensors_on_cpu", "all_tensors_finite")} |
                {"metadata_fields": sorted(record.get("metadata", {})), "saved_metadata_resumable_flag": record.get("metadata", {}).get("resumable")}
                for record in checkpoint_metadata["checkpoints"]],
            "raw_checkpoint_loaded_by_this_audit": False,
            "resume_limit": "The old boundary's saved resumable flag is historical; the spent training ledger has advanced past it. The final active-episode checkpoint is audit/evaluation-only. Neither is currently a valid continuation boundary."},
        "paired_T_lost_success_cases": lost,
        "limitations": ["These are only the first selected plan per shape from the first training batch, before any learned update. They are not representative full episodes or final-policy evaluation videos.",
            "Saved RGB cannot show later L failures, the T paired lost-success case, contact recovery later in training, or behavior of the final delta.",
            "Evaluation reports preserve every executed action/metric row but no RGB, per-control contact or numeric object-pose trajectory. New rendering from inferred/restored states is outside this offline audit.",
            "Commanded action targets are not measured object states or actual pusher paths; no state is inferred from them.",
            "The replay retains uint8-quantized 96x96 observations, episode-local control indices, observed pusher positions and reported per-control contacts; it omits full denoising chains, optimizer data and most on-policy buffer fields.",
            "Resource timings are existing monitor evidence only; this audit runs no live resource query and provides no estimate for a new experiment."],
        "existing_resource_evidence": resources, "inputs": inputs}
    json_file(output / "inventory.json", inventory)
    unchanged = all(identity(ROOT / name) == expected for name, expected in inputs.items())
    if not unchanged:
        raise ValueError("An input changed during the offline export")
    manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs, "inputs_unchanged_after_export": unchanged,
        "outputs": {str(path.relative_to(output)): identity(path) for path in sorted(output.rglob("*")) if path.is_file()},
        "manifest_self_hash_omitted": True, "new_runtime_controls": 0, "new_policy_samples": 0,
        "raw_model_checkpoints_loaded": 0, "replay_payloads_cpu_weights_only_loaded": 2}
    json_file(output / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output), "saved_rgb_scope": inventory["saved_rgb_scope"],
        "evaluation_control_rows": {method: value["recorded_control_rows"] for method, value in inventory["evaluation"].items()},
        "paired_T_lost_success_cases": lost, "output_files": len(manifest["outputs"]) + 1,
        "cuda_initialized": torch.cuda.is_initialized(), "inputs_unchanged": unchanged}, indent=2))


if __name__ == "__main__":
    main()
