"""Bounded offline pixel evidence. Import is standard-library-only.

Only run_pixels imports NumPy/Pillow and the fixed, non-fitting chromatic helpers.
It never calls a model, simulator, decoder, pose fit, or optimization routine.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import sys
import textwrap
import zipfile

WRITE_CAP = 32 * 1024 ** 2
ZERO_OPERATIONS = dict(model_loads=0, model_forwards=0, SDK_calls=0,
                       physics_calls=0, full_measurement_fits=0, collection=0)
PREDICTION_GRID = {(r, m, h) for r in ("frozen", "adapted")
                   for m, h in (("AR", 1), ("AR", 4), ("TF", 4))}


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _path(root, value):
    root = Path(root).resolve()
    relative = Path(str(value).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("only repository-relative input paths are accepted")
    result = root / relative
    for current in (result, *result.parents):
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError("linked input path")
        if current == root:
            break
    if not result.resolve().is_relative_to(root):
        raise ValueError("input escapes repository")
    return result


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}
        self.identities = []

    def read(self, item):
        path = _path(self.root, item["path"])
        identity = (int(item["bytes"]), item["sha256"])
        if path in self.cache:
            old, payload = self.cache[path]
            if old != identity:
                raise ValueError("conflicting immutable identity")
            return payload
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("input must be an ordinary single-link file")
        if before.st_size != identity[0]:
            raise ValueError("input byte count mismatch")
        payload = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("input changed during read")
        if _sha(payload) != identity[1]:
            raise ValueError("input SHA mismatch")
        self.cache[path] = (identity, payload)
        self.identities.append(dict(path=path.relative_to(self.root).as_posix(),
                                    bytes=len(payload), sha256=identity[1]))
        return payload


def read_npy_window(payload, member, start, count, tail, kinds="fiu"):
    """Decode only selected contiguous rows, never an entire image member.

    ZIP decompression/seek may transport preceding compressed bytes; these bytes
    are not converted to numeric arrays. The returned record distinguishes both.
    """
    import numpy as np
    if type(start) is not int or type(count) is not int or start < 0 or not 1 <= count <= 5:
        raise ValueError("invalid bounded window")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist().count(member) != 1:
            raise ValueError("missing/duplicate member")
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("unsupported NPY version")
            if (fortran or dtype.hasobject or dtype.kind not in kinds or dtype.itemsize > 8
                    or len(shape) != len(tail) + 1 or tuple(shape[1:]) != tuple(tail)
                    or start + count > shape[0]):
                raise ValueError("member shape/dtype/window mismatch")
            if member == "images.npy" and dtype != np.dtype("uint8"):
                raise ValueError("training images must retain uint8 source storage")
            row_bytes = math.prod(tail) * dtype.itemsize
            header_bytes = stream.tell()
            offset = header_bytes + start * row_bytes
            if stream.seek(offset) != offset:
                raise ValueError("selected member seek failed")
            selected = stream.read(count * row_bytes)
            if len(selected) != count * row_bytes:
                raise ValueError("short selected member")
    result = np.frombuffer(selected, dtype=dtype).reshape((count, *tail)).copy()
    return result, dict(member=member, stored_shape=list(shape), dtype=dtype.str,
                        start=start, rows=count, decoded_bytes=len(selected),
                        selected_sha256=_sha(selected), prefix_transport_bytes=start * row_bytes,
                        unselected_numeric_rows_decoded=0)


def conditioning(history, actual, past_actions, future_actions, mode, horizon, transform):
    import numpy as np
    actions = np.concatenate((past_actions, future_actions[:horizon]))[-4:]
    actions = (actions.astype(np.float32) / np.float32(511)
               * np.float32(2) - np.float32(1))
    if mode == "AR" and horizon != 1:
        return None, actions
    frames = history if mode == "AR" else np.concatenate((history, actual[:horizon - 1]))[-4:]
    return np.stack([transform(frame)[0] for frame in frames]), actions


def _component_sizes(seed):
    pending = set(zip(*seed.nonzero()))
    sizes = []
    while pending:
        todo = [pending.pop()]
        size = 0
        while todo:
            y, x = todo.pop()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    point = (y + dy, x + dx)
                    if point in pending:
                        pending.remove(point)
                        todo.append(point)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def pixel_diagnostics(raw, fixed):
    import numpy as np
    if raw.shape != (3, 96, 96) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("prediction must be finite float32 CHW96")
    clipped = np.clip(raw, 0, 1)
    rgb = np.moveaxis(clipped, 0, -1).astype(float) * 255
    alpha, blue, palette = fixed.continuous_maps(rgb)
    bounds, fraction = fixed._region(alpha, fixed.CONFIG["shape_gate"]["minimum_object_seed_pixels"])
    seed = alpha >= fixed.CONFIG["chromatic"]["component_seed_alpha"]
    sizes = _component_sizes(seed)
    total = float(alpha.sum())
    inside = float(alpha[bounds].sum()) if bounds else 0.0
    metrics = dict(raw_min=float(raw.min()), raw_max=float(raw.max()),
                   raw_channel_means=[float(v) for v in raw.mean((1, 2))],
                   raw_out_of_range_channel_fraction=float(((raw < 0) | (raw > 1)).mean()),
                   display_clipped_pixel_fraction=float(((raw < 0) | (raw > 1)).any(0).mean()),
                   raw_at_or_below_zero_fraction=float((raw <= 0).mean()),
                   raw_at_or_above_one_fraction=float((raw >= 1).mean()),
                   raw_chroma_mean_rgb255=float((raw.max(0) - raw.min(0)).mean() * 255),
                   palette=palette, object_alpha_mass=total, alpha_mass_in_roi=inside,
                   alpha_mass_outside_roi=max(0.0, total - inside),
                   alpha_fraction_outside_roi=(max(0.0, total - inside) / total if total else None),
                   largest_component_mass_fraction=float(fraction), seed_component_count=len(sizes),
                   seed_pixel_count=int(seed.sum()), largest_seed_pixels=sizes[0] if sizes else 0,
                   blue_alpha_mass=float(blue.sum()),
                   roi=([bounds[0].start, bounds[0].stop, bounds[1].start, bounds[1].stop]
                        if bounds else None),
                   interpretation="chromatic/ROI diagnostics only; no pose fit or validity reassignment")
    return metrics, dict(rgb=clipped, alpha=alpha, blue=blue, seed=seed, bounds=bounds)


def display_rgb(raw):
    import numpy as np
    return np.moveaxis(np.rint(np.clip(raw, 0, 1) * 255).astype(np.uint8), 0, -1)


class Outputs:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.bytes = 0
        self.files = []

    def write(self, name, payload):
        if Path(name).name != name or self.bytes + len(payload) + 65536 > WRITE_CAP:
            raise ValueError("output name/cap violation")
        with (self.path / name).open("xb") as stream:
            if stream.write(payload) != len(payload):
                raise OSError("short offline output write")
        self.bytes += len(payload)
        record = dict(path=name, bytes=len(payload), sha256=_sha(payload))
        self.files.append(record)
        return record

    def sheet(self, name, rows, title):
        from PIL import Image, ImageDraw
        width, height = 232, 270
        canvas = Image.new("RGB", (width * max(map(len, rows)), 64 + height * len(rows)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), title, fill="black")
        draw.text((8, 23), "DISPLAY ONLY: clip [0,1], round x255; raw numerical values unchanged", fill="black")
        for y, row in enumerate(rows):
            for x, (pixels, caption) in enumerate(row):
                tile = Image.fromarray(pixels).resize((192, 192), resample=Image.Resampling.NEAREST)
                canvas.paste(tile, (width * x + 20, 64 + y * height))
                draw.multiline_text((width * x + 5, 64 + y * height + 195),
                                    "\n".join(textwrap.wrap(caption, 35)[:5]), fill="black")
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return self.write(name, buffer.getvalue())


def _prediction_path(row):
    raw = Path(row["raw_path"].replace("\\", "/"))
    if raw.parts[0] == "predictions":
        return (Path(row["measurement_path"].replace("\\", "/")).parent.parent / raw).as_posix()
    return raw.as_posix()


def _validate_manifest(manifest, inputs):
    branches, windows = manifest["branches"], manifest["train_windows"]
    if not 1 <= len(branches) <= 8 or not 1 <= len(windows) <= 8:
        raise ValueError("fixed branch/window cap")
    data = json.loads(inputs.read(manifest["bindings"]["data"]))
    roles = {item["trajectory_id"]: item["role"] for item in data["episodes"]}
    # All declared bindings are metadata; held payloads are never opened.
    for item in manifest["bindings"].values():
        inputs.read(item)
    cells = set()
    for branch in branches:
        cell = (branch["shape"], branch["stratum"], branch["contact_by_h4"])
        if (cell in cells or cell[0] not in ("T", "L") or
                cell[1] not in ("regression", "independent") or type(cell[2]) is not bool):
            raise ValueError("duplicate/invalid predeclared cell")
        cells.add(cell)
        rows = branch["predictions"]
        if len(rows) != 6 or {(r["role"], r["mode"], r["horizon"]) for r in rows} != PREDICTION_GRID:
            raise ValueError("prediction grid mismatch")
        if len(branch["truth"]) != 4:
            raise ValueError("four physical truth frames required")
        inputs.read(branch["payload"])
        for row in rows:
            if row["seed"] != 5478 or row["status"] != "completed":
                raise ValueError("selected row seed/status mismatch")
            for key in ("anchor", "candidate", "shape", "stratum", "contact_by_h4", "latest"):
                if row[key] != branch[key]:
                    raise ValueError("selected branch/measurement identity mismatch: " + key)
            if row["truth"] != branch["truth"][row["horizon"] - 1]:
                raise ValueError("selected truth identity mismatch")
            saved = json.loads(inputs.read(dict(path=row["measurement_path"],
                bytes=row["file_bytes"], sha256=row["file_sha256"])))
            expected = {k: v for k, v in row.items() if k not in
                        {"measurement_path", "file_bytes", "file_sha256", "prediction_file_identity"}}
            if saved != expected:
                raise ValueError("saved measurement differs from frozen manifest")
            inputs.read(dict(path=_prediction_path(row), **row["prediction_file_identity"]))
    records = []
    for window in windows:
        if roles.get(window["trajectory_id"]) != "train":
            raise ValueError("only declared train payloads may be opened")
        record = json.loads(inputs.read(window["record"]))
        if (record["trajectory"]["trajectory_id"] != window["trajectory_id"]
                or record["image_file"] != Path(window["payload"]["path"]).name
                or record["image_sha256"] != window["payload"]["sha256"]):
            raise ValueError("training record/payload identity mismatch")
        anchor = window["anchor"]
        if type(anchor) is not int or not 3 <= anchor < len(record["trajectory"]["transitions"]):
            raise ValueError("training anchor outside a complete five-frame window")
        inputs.read(window["payload"])
        records.append(record)
    return records


def _physical_arrays(payload):
    import numpy as np
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("history", "actual", "past_actions", "future_actions")}
    for key in ("history", "actual"):
        value = arrays[key]
        if (value.shape != (4, 3, 96, 96) or value.dtype != np.float32
                or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1):
            raise ValueError("physical frame shape/dtype/range mismatch")
    for key, count in (("past_actions", 3), ("future_actions", 4)):
        value = arrays[key]
        if value.shape != (count, 2) or not np.isfinite(value).all() or value.min() < 0 or value.max() > 511:
            raise ValueError("physical action shape/range mismatch")
    return arrays


def _mask_view(value):
    import numpy as np
    return np.repeat(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)


def _roi_view(view):
    result = display_rgb(view["rgb"])
    bounds = view["bounds"]
    if bounds:
        y, x = bounds
        result[y.start:y.stop, x.start] = [255, 255, 0]
        result[y.start:y.stop, x.stop - 1] = [255, 255, 0]
        result[y.start, x.start:x.stop] = [255, 255, 0]
        result[y.stop - 1, x.start:x.stop] = [255, 255, 0]
    return result


def run_pixels(root, output, manifest):
    """Export only fixed evidence. First errors propagate without retry/repair."""
    root = Path(root).resolve()
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_bytes())
    inputs = Inputs(root)
    records = _validate_manifest(manifest, inputs)  # identity before numerical decode
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import numpy as np
    from tdyssey.wm_compat_transform import transform_t1, T1_SHA256
    from tdyssey import wm_image_measurement_v2 as fixed
    outputs = Outputs(Path(output) / "pixels")
    report = dict(kind="gpc_checkpoint500_fixed_offline_pixels.v1", branches=[], train_windows=[],
        selected_array_reads=[], forbidden_operations=dict(ZERO_OPERATIONS), t1_sha256=T1_SHA256,
        numerical_prediction_clipping=False,
        map_input="np.clip(raw,0,1), float64 HWC x255; fixed chromatic helpers",
        limitations=["Selected cases are descriptive, not pooled evaluation estimates.",
            "ROI/chromatic fragmentation is not a replacement accepted pose.",
            "AR H4 input is unknown: intermediate raw frames are outside the selected-array budget."])
    counts = dict(prediction_frames=0, physical_frames=0, training_frames=0,
        training_small_state_rows=0, verified_input_hashes=0,
        unknown_AR4_input_hashes=0, verified_action_hashes=0)
    for branch in manifest["branches"]:
        arrays = _physical_arrays(inputs.read(branch["payload"]))
        history, actual = arrays["history"], arrays["actual"]
        counts["physical_frames"] += 8
        report["selected_array_reads"].append(dict(path=branch["payload"]["path"],
            members=dict(history=4, actual=4, past_actions=3, future_actions=4), image_frames=8))
        name = branch["anchor"] + "-" + branch["candidate"]
        title = f"{branch['shape']} {branch['stratum']} {name} contact={branch['contact_by_h4']}"
        outputs.sheet(name + "-history.png", [
            [(display_rgb(frame), f"real history {i-3}") for i, frame in enumerate(history)],
            [(display_rgb(frame), f"real future H{i+1}") for i, frame in enumerate(actual)]], title)
        frames, evidence, segment_rows = {}, [], []
        for row in branch["predictions"]:
            path = _prediction_path(row)
            raw = np.load(io.BytesIO(inputs.read(dict(path=path, **row["prediction_file_identity"]))), allow_pickle=False)
            if _sha(raw.tobytes()) != row["raw_sha256"]:
                raise ValueError("raw float payload SHA mismatch")
            metrics, view = pixel_diagnostics(raw, fixed)
            if metrics["raw_min"] != row["raw_min"] or metrics["raw_max"] != row["raw_max"]:
                raise ValueError("saved raw range mismatch")
            given, actions = conditioning(history, actual, arrays["past_actions"],
                arrays["future_actions"], row["mode"], row["horizon"], transform_t1)
            if _sha(actions.tobytes()) != row["action_sha256"]:
                raise ValueError("normalized action SHA mismatch")
            counts["verified_action_hashes"] += 1
            if given is None:
                input_check = "unknown_AR4_intermediate_raw_frames_not_opened"
                counts["unknown_AR4_input_hashes"] += 1
            else:
                if _sha(given.tobytes()) != row["input_sha256"]:
                    raise ValueError("selected real-conditioning input SHA mismatch")
                input_check = "matched"
                counts["verified_input_hashes"] += 1
            target = actual[row["horizon"] - 1]
            metrics.update(raw_mae_to_real_target=float(np.abs(raw-target).mean()),
                raw_mae_to_T1_target=float(np.abs(raw-transform_t1(target)[0]).mean()))
            label = f"{row['role']} {row['mode']} H{row['horizon']}"
            measurement = row["measurement"]
            original_mass = measurement.get("shape", {}).get("component_mass_fraction")
            metrics["original_component_mass_fraction"] = original_mass
            metrics["component_mass_fraction_replay_difference"] = (
                metrics["largest_component_mass_fraction"] - original_mass
                if original_mass is not None else None)
            state = measurement.get("state", {})
            caption = f"{label} state={state.get('valid')} reason={state.get('reason')}"
            segment_rows.append([(display_rgb(raw), caption),
                (_mask_view(view["alpha"]), "fixed continuous object alpha"),
                (_mask_view(view["seed"]), f"seed components={metrics['seed_component_count']}"),
                (_roi_view(view), f"image ROI; component mass={metrics['largest_component_mass_fraction']:.4f}"),
                (_mask_view(view["blue"]), "fixed continuous pusher alpha")])
            frames[(row["role"], row["mode"], row["horizon"])] = display_rgb(raw)
            evidence.append(dict(role=row["role"], mode=row["mode"], horizon=row["horizon"],
                measurement_path=row["measurement_path"], prediction_path=path,
                prediction_file_identity=row["prediction_file_identity"], raw_sha256=row["raw_sha256"],
                original_measurement=measurement, metrics=metrics, input_hash_check=input_check,
                normalized_action_hash_check="matched", normalized_actions=actions.tolist()))
            report["selected_array_reads"].append(dict(path=path, image_frames=1,
                sha256=row["prediction_file_identity"]["sha256"], raw_sha256=row["raw_sha256"]))
            counts["prediction_frames"] += 1
        compare_rows = []
        for mode, horizon in (("AR", 1), ("AR", 4), ("TF", 4)):
            target = actual[horizon-1]
            compare_rows.append([(display_rgb(target), f"real target H{horizon}"),
                (display_rgb(transform_t1(target)[0]), f"T1 real target H{horizon}"),
                (frames[("frozen", mode, horizon)], f"frozen {mode} H{horizon}"),
                (frames[("adapted", mode, horizon)], f"adapted {mode} H{horizon}")])
        outputs.sheet(name + "-comparison.png", compare_rows, title)
        outputs.sheet(name + "-segmentation.png", segment_rows, title + "; NO FIT")
        report["branches"].append({k: branch[k] for k in
            ("shape", "stratum", "contact_by_h4", "anchor", "candidate", "payload", "goal", "latest", "truth")}
            | dict(predictions=evidence))
    for index, (window, record) in enumerate(zip(manifest["train_windows"], records)):
        payload = inputs.read(window["payload"])
        anchor = window["anchor"]
        images, image_read = read_npy_window(payload, "images.npy", anchor-3, 5, (3, 96, 96), "u")
        agent, agent_read = read_npy_window(payload, "agent_pos.npy", anchor-3, 5, (2,))
        goals, goal_read = read_npy_window(payload, "goal_pose.npy", anchor-3, 5, (3,))
        transitions = record["trajectory"]["transitions"][anchor-3:anchor+1]
        states = [transition["before"] for transition in transitions] + [transitions[-1]["after"]]
        if len(transitions) != 4:
            raise ValueError("four training transitions required")
        for previous, following in zip(transitions, transitions[1:]):
            if previous["after"] != following["before"]:
                raise ValueError("discontinuous training state time alignment")
        expected_agent = [state["pusher_body"]["pose"]["position"] for state in states]
        expected_goals = [[*state["goal"]["position"], state["goal"]["angle"] % (2*math.pi)] for state in states]
        if (not np.isfinite(agent).all() or not np.isfinite(goals).all()
                or not np.allclose(agent, expected_agent, rtol=0, atol=1e-9)
                or not np.allclose(goals, expected_goals, rtol=0, atol=1e-9)):
            raise ValueError("small observation arrays and before/after states disagree")
        actions = np.asarray([transition["action"] for transition in transitions], dtype=np.float32)
        if actions.shape != (4, 2) or not np.isfinite(actions).all() or actions.min() < 0 or actions.max() > 511:
            raise ValueError("training actions outside original normalization domain")
        raw = images.astype(np.float32) / np.float32(255)
        t1 = [transform_t1(frame)[0] for frame in raw]
        rows = [[(display_rgb(frame), f"{label} source frame {anchor-3+i}; " +
                   ("TARGET" if i == 4 else "conditioning")) for i, frame in enumerate(frames)]
                for label, frames in (("raw uint8/255", raw), ("T1 applied once", t1))]
        outputs.sheet(f"train-{index:02d}-{window['shape']}.png", rows,
            f"TRAIN ONLY {window['trajectory_id']} anchor={anchor} contact={window['contact']}")
        reads = [image_read, agent_read, goal_read]
        report["selected_array_reads"].append(dict(path=window["payload"]["path"], members=reads,
                                                   image_frames=5))
        report["train_windows"].append(dict(selection=window, selected_member_reads=reads,
            image_indices=list(range(anchor-3, anchor+2)), actions=actions.tolist(),
            normalized_actions=(actions/np.float32(511)*np.float32(2)-np.float32(1)).tolist(),
            agent_pos=agent.tolist(), goal_pose=goals.tolist(),
            numeric_state_alignment="matched before[4]+last after to five selected small-array rows",
            state_snapshots=states, transitions=transitions, image_role="first4 conditioning; fifth target"))
        counts["training_frames"] += 5
        counts["training_small_state_rows"] += 10
    if counts["prediction_frames"] > 48 or counts["physical_frames"] > 64 or counts["training_frames"] > 40:
        raise ValueError("decoded frame cap")
    report.update(decoded_counts=counts, input_identities=inputs.identities,
        fixed_manifest_sha256=_sha(_json(manifest)),
        manifest_hash_encoding="canonical sorted indent2 JSON plus newline (not source-file hash)")
    outputs.write("NOTES.txt", ("All panels use labelled display-only clipping. Raw metrics preserve float values.\n"
        "Weak alpha outside the largest component ROI is distinct from disconnected thresholded seeds.\n"
        "No optimizing fit, accepted pose repair, model, SDK, physics, or additional collection.\n"
        "Training ZIP decompression may transport unselected bytes but only five selected image rows are decoded.\n"
        "AR4 conditioning cannot be reconstructed without excluded intermediate predictions.\n").encode())
    report["outputs_before_report"] = list(outputs.files)
    report["output_bytes_before_report"] = outputs.bytes
    outputs.write("pixels.json", _json(report))
    return dict(report="pixels/pixels.json", files=outputs.files, total_output_bytes=outputs.bytes,
                decoded_counts=counts, forbidden_operations=dict(ZERO_OPERATIONS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(run_pixels(args.root, args.output, args.manifest), sort_keys=True))
