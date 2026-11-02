"""CPU media from recorded R2A pixels only; no simulator or model imports/calls.

Raw PNG/FFV1 preserve saved uint8 RGB. H264 pairs are labelled review encodes.
Every planned pair remains in the HTML, including unexecuted/partial slots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FPS = 10
SCALE = 6
WALLS = ("left", "right", "top", "bottom")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def inside(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Input/output path must stay inside the R2A directory")
    return path


def safe_id(value):
    value = str(value)
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in value):
        raise ValueError("Unsafe slot identifier")
    return value


def storage_guard(root, cap):
    total = 0
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = [d for d in directories if not (os.stat(Path(current) / d, follow_symlinks=False).st_file_attributes & 0x400)] if os.name == "nt" else [d for d in directories if not (Path(current) / d).is_symlink()]
        for name in files:
            p = Path(current) / name
            st = p.stat(follow_symlinks=False)
            if p.is_symlink() or (getattr(st, "st_file_attributes", 0) & 0x400):
                continue
            total += st.st_size
    if total > cap:
        raise RuntimeError(f"R2A128MiB artifact cap exceeded: {total}>{cap}; retain evidence and stop, never delete old data")
    return total


def font(size):
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise RuntimeError("Installed Chinese font required; do not silently generate unreadable labels")


def _first(mapping, names, default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _finite_list(value):
    if value is None:
        return None
    a = np.asarray(value, dtype=float)
    return a.tolist() if np.isfinite(a).all() else None


def measures(letter, states):
    """Pure arithmetic on saved states and fixed local geometry, never an env."""
    from tdyssey.envs.official import official_shape
    shape = official_shape(letter)
    local = np.asarray(shape.centroid)
    c, s = np.cos(states[:, 4]), np.sin(states[:, 4])
    centers = states[:, 2:4] + np.column_stack((c * local[0] - s * local[1], s * local[0] + c * local[1]))
    vertices = np.asarray([p for rect in shape.components for p in rect.vertices])
    x = c[:, None] * vertices[:, 0] - s[:, None] * vertices[:, 1] + states[:, 2, None]
    y = s[:, None] * vertices[:, 0] + c[:, None] * vertices[:, 1] + states[:, 3, None]
    gaps = np.column_stack((x.min(1) - 7, 504 - x.max(1), y.min(1) - 7, 504 - y.max(1)))
    return centers, gaps


def load_slot(root, row, request):
    sid = safe_id(row.get("slot_id", row.get("request_id", request["request_id"])))
    result = dict(slot_id=sid, request=request, index=row, attempted=row.get("attempted"), status=row.get("status", "unknown"), available=False)
    npz = inside(root, _first(row, ("trajectory_file", "npz_file", "file"), f"trajectories/{sid}/trajectory.npz"))
    meta = inside(root, row.get("metadata_file", f"trajectories/{sid}/metadata.json"))
    if not npz.exists():
        if row.get("attempted") is True:
            result["missing_reason"] = "attempted_but_no_saved_trajectory"
        else:
            result["missing_reason"] = "not_executed_or_no_saved_trajectory"
        return result
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    with np.load(npz, allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    rgb, states, actions = arrays["rgb"], arrays["states"], arrays["actions"]
    n = len(actions)
    if rgb.dtype != np.uint8 or rgb.shape != (n + 1, 3, 96, 96):
        raise ValueError(f"{sid}: uint8 RGB N+1 CHW96 contract failed")
    if states.shape != (n + 1, 15) or actions.shape != (n, 2) or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"{sid}: finite state/action timeline contract failed")
    if (actions < 0).any() or (actions > 511).any():
        raise ValueError(f"{sid}: absolute action contract failed")
    planned = np.asarray(request["actions_absolute"], dtype=np.float64)
    if n > len(planned) or not np.array_equal(actions, planned[:n].astype(actions.dtype)):
        raise ValueError(f"{sid}: recorded actions differ from frozen prefix")
    if metadata.get("request", {}).get("request_id", request["request_id"]) != request["request_id"]:
        raise ValueError("Metadata request identity mismatch")
    records = metadata.get("control_records", [])
    completed_records = {}
    failed_tail_records = []
    unclassified_records = []
    for record in records:
        if record.get("completed") is True:
            control = record.get("control_index")
            if type(control) is not int or control < 0 or control >= n or control in completed_records:
                raise ValueError(f"{sid}: completed record index outside saved boundary actions or duplicated")
            completed_records[control] = record
        elif record.get("completed") is False:
            failed_tail_records.append(record)
        else:
            unclassified_records.append(record)
    missing_completed_records = [t for t in range(n) if t not in completed_records]
    declared = _first(metadata, ("frame_sha256", "frameSHA", "frame_sha256_chw_uint8"))
    frame_hashes = [hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in rgb]
    if isinstance(declared, dict):
        declared = declared.get("sha256", declared.get("frames"))
    if declared is not None:
        declared = [v.get("sha256") if isinstance(v, dict) else v for v in declared]
        if declared != frame_hashes:
            raise ValueError(f"{sid}: saved CHW uint8 frame hash mismatch")
    centers, gaps = measures(request["letter"], states)
    transitions = []
    for t in range(n):
        # Raw boundary frames remain usable even if a failed IO left metadata
        # incomplete; unknown contact/penetration is never fabricated as zero.
        record = completed_records.get(t, {})
        contact = _first(record, ("contact", "contact_count", "object_pusher_contact_count"))
        if contact is None and "contact" in arrays:
            contact = int(arrays["contact"][t])
        arc = _first(record, ("contact_arc_bins", "actual_contact_arc_bins", "actual_arc_bins", "contacted_bin_ids"))
        pen = _first(record, ("max_substep_penetrations", "max_substep_penetrations_world_px", "penetration", "max_penetrations"))
        if pen is None and "substep_penetrations" in arrays:
            pen = np.asarray(arrays["substep_penetrations"][t]).max(0)
        delta = (centers[t + 1] - centers[t]) * 96 / 512
        transitions.append(dict(slot_id=sid, letter=request["letter"], protocol=request["protocol"], control=t,
            from_state=t, to_state=t + 1, time_start=t / FPS, time_end=(t + 1) / FPS,
            dx_image_px=float(delta[0]), dy_image_px=float(delta[1]), translation_image_px=float(np.linalg.norm(delta)),
            signed_rotation_deg=float(np.degrees(states[t + 1, 4] - states[t, 4])),
            area_centroid_world_xy=centers[t + 1].tolist(), actual_contact=bool(contact) if contact is not None else None,
            contact_count=contact, contact_arc_bins=arc, max_substep_penetrations=_finite_list(pen),
            actual_contact_point_samples=record.get("actual_contact_point_samples"),
            penetration_max=float(np.max(pen)) if pen is not None and np.isfinite(pen).all() else None,
            actual_four_wall_gaps_world_px=dict(zip(WALLS, map(float, gaps[t + 1]))),
            actual_two_nearest_walls=[dict(wall=WALLS[i], gap_world_px=float(gaps[t + 1, i])) for i in np.argsort(gaps[t + 1])[:2]],
            absolute_action_xy=actions[t].tolist(), action_target_distance_world_px=float(np.linalg.norm(actions[t] - states[t, :2])),
            action_phase=_first(record, ("action_phase", "phase"), request.get("action_schedule", ["unknown"] * n)[t]),
            valid=_first(record, ("valid", "valid_transition")), failure=record.get("failure")))
    result.update(available=True, attempted=True, arrays=arrays, metadata=metadata, frame_hashes=frame_hashes,
        frames=rgb, controls=n, transitions=transitions, gaps=gaps, centers=centers,
        npz_file=npz.relative_to(root).as_posix(), npz_sha256=sha(npz), metadata_file=meta.relative_to(root).as_posix(), metadata_sha256=sha(meta),
        declared_frame_hashes_verified=declared is not None, initial_state_raw_sha256=hashlib.sha256(np.ascontiguousarray(states[0]).tobytes()).hexdigest(),
        initial_identity=metadata.get("initial_identity"), protocol_sha256=_first(metadata, ("protocol_sha256", "protocolSHA")),
        failure=_first(metadata, ("failure", "error"), row.get("failure")),
        control_records_total=len(records), completed_record_count=len(completed_records),
        missing_completed_record_controls=missing_completed_records,
        failed_tail_records=failed_tail_records, unclassified_control_records=unclassified_records,
        partial_endpoint=metadata.get("partial_endpoint"),
        partial_control_span_seconds=metadata.get("partial_control_span_seconds"),
        partial_time_exact=metadata.get("partial_time_exact"),
        partial_state_substep_indices=metadata.get("partial_state_substep_indices"),
        unavailable_partial_state_substeps=metadata.get("unavailable_partial_state_substeps"),
        partial_states_present="partial_states" in arrays,
        partial_states_saved_count=len(arrays["partial_states"]) if "partial_states" in arrays else None,
        partial_rgb_available=bool(np.asarray(arrays["partial_rgb"]).size) if "partial_rgb" in arrays else None,
        telemetry_file=metadata.get("telemetry_file"), telemetry_sha256=metadata.get("telemetry_sha256"),
        partial_endpoint_is_boundary_frame=False,
        physical_span_seconds=n / FPS, encoded_duration_seconds=(n + 1) / FPS, final_endpoint_hold_seconds=.1)
    return result


def encode(path, frames, *, lossless=False):
    """Use installed bundled ffmpeg, CPU only; no network or codec download."""
    import imageio_ffmpeg
    frames = iter(frames)
    first = np.ascontiguousarray(next(frames), dtype=np.uint8)
    h, w = first.shape[:2]
    params = dict(path=str(path), size=(w, h), fps=FPS, pix_fmt_in="rgb24", macro_block_size=1, ffmpeg_log_level="error")
    if lossless:
        params.update(codec="ffv1", pix_fmt_out="bgr0", output_params=["-level", "3"])
    else:
        params.update(codec="libx264", pix_fmt_out="yuv420p", output_params=["-crf", "16", "-preset", "fast", "-movflags", "+faststart"])
    writer = imageio_ffmpeg.write_frames(**params)
    writer.send(None)
    try:
        writer.send(first)
        for frame in frames:
            a = np.ascontiguousarray(frame, dtype=np.uint8)
            if a.shape != first.shape:
                raise ValueError("Video canvas dimensions changed")
            writer.send(a)
    finally:
        writer.close()


def verify_lossless(path, frames):
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    if tuple(metadata["size"]) != (96, 96) or abs(metadata["fps"] - FPS) > 1e-8:
        raise RuntimeError("Native lossless video resolution/fps mismatch")
    count = 0
    for data in reader:
        if count >= len(frames) or not np.array_equal(np.frombuffer(data, dtype=np.uint8).reshape(96, 96, 3), np.moveaxis(frames[count], 0, -1)):
            raise RuntimeError("Native FFV1 decode differs from saved raw RGB")
        count += 1
    if count != len(frames):
        raise RuntimeError("Native FFV1 frame count changed")
    return count


def verify_review_timing(path, expected_frames):
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    if tuple(metadata["size"]) != (1440, 1120) or abs(metadata["fps"] - FPS) > 1e-8:
        raise RuntimeError("Review video dimensions/fps changed")
    count = sum(1 for _ in reader)
    if count != expected_frames:
        raise RuntimeError("Review encoding dropped/duplicated frames")
    return dict(decoded_frame_count=count, decoded_fps=metadata["fps"], encoded_duration_seconds=count / FPS)


def short(value, digits=3):
    return "未记录" if value is None else f"{value:.{digits}f}"


def text(draw, xy, value, size=17, fill="#17212b"):
    draw.text(xy, str(value), font=font(size), fill=fill)


def _points(draw, rect, values, color, *, zero=False, bounds=None, controls=8, threshold=None):
    x0, y0, x1, y1 = rect
    finite = [v for v in values if v is not None and math.isfinite(v)]
    lo, hi = bounds or (min(finite + ([0.] if zero else []), default=0.), max(finite + ([0.] if zero else []), default=1.))
    if hi - lo < 1e-6:
        hi = lo + 1
    draw.rectangle(rect, outline="#b2bdc7")
    if threshold is not None:
        yy = y1 - (threshold - lo) / (hi - lo) * (y1 - y0)
        draw.line((x0, yy, x1, yy), fill="#bd2639", width=1)
    points = [(x0 + (i + 1) / controls * (x1 - x0), y1 - (v - lo) / (hi - lo) * (y1 - y0)) for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    if len(points) > 1:
        draw.line(points, fill=color, width=2)
    for x, y in points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    text(draw, (x0, y1 + 3), f"共轴 {lo:.2f}～{hi:.2f}; t=0～{controls / FPS:.1f}s", 12)


def curve_canvas(slots):
    canvas = Image.new("RGB", (1440, 290), "white")
    draw = ImageDraw.Draw(canvas)
    colors = ("#196fc0", "#d65c16")
    controls = max(int(s["request"]["max_controls"]) for s in slots)
    metrics = (("translation_image_px", "面积质心 Δ图像px"), ("signed_rotation_deg", "带符号 Δ角度°"), ("penetration_max", "穿透worldpx;红线2px"))
    shared_bounds = {}
    for key, _ in metrics:
        values = [r.get(key) for s in slots for r in s.get("transitions", [])]
        values = [v for v in values if v is not None and math.isfinite(v)] + [0.]
        if key == "penetration_max":
            values.append(2.)
        shared_bounds[key] = (min(values), max(values))
    all_gaps = [s["gaps"] for s in slots if s.get("available")]
    gap_lo = min([float(a.min()) for a in all_gaps] + [0.])
    gap_hi = max([float(a.max()) for a in all_gaps] + [1.])
    for col, slot in enumerate(slots):
        left = 16 + col * 720
        text(draw, (left, 3), f"{slot['request']['protocol']} 逐步曲线：每点0.1秒；非累计幅度", 17, colors[col])
        rows = slot.get("transitions", [])
        for j, (key, label) in enumerate(metrics):
            x = left + j * 232
            text(draw, (x, 31), label, 15)
            _points(draw, (x + 3, 56, x + 218, 142), [r.get(key) for r in rows], colors[col], zero=True,
                    bounds=shared_bounds[key], controls=controls, threshold=2. if key == "penetration_max" else None)
        text(draw, (left, 171), "四墙带符号 gap(world px)，颜色 L蓝/R橙/T绿/B紫；不代表墙接触计数", 14)
        rect = (left + 3, 199, left + 680, 254)
        gaps = slot.get("gaps", np.empty((0, 4)))
        if len(gaps):
            lo, hi = gap_lo, gap_hi
            draw.rectangle(rect, outline="#b2bdc7")
            for k, color in enumerate(("#196fc0", "#d65c16", "#278c44", "#8c4bb4")):
                ps = [(rect[0] + i / controls * (rect[2] - rect[0]), rect[3] - (float(v) - lo) / (hi - lo) * (rect[3] - rect[1])) for i, v in enumerate(gaps[:, k])]
                if len(ps) > 1:
                    draw.line(ps, fill=color, width=2)
            text(draw, (left, 260), f"实际范围 {lo:.2f}～{hi:.2f}；全部逐值见 CSV/JSON", 13)
    return canvas


def pair_frame(slots, i, curves):
    canvas = Image.new("RGB", (1440, 1120), "#edf1f4")
    draw = ImageDraw.Draw(canvas)
    text(draw, (16, 4), "R2A 真实连续记录 | 10Hz / 1×时间 | 96px原图 + 仅显示放大6×(nearest)", 23)
    text(draw, (16, 38), f"状态帧 {i} / 模拟 t={i / FPS:.1f}s；动作/接触/位移数值关联上一控制；视频末帧显示0.1s为端点停留", 17)
    for col, slot in enumerate(slots):
        x = 16 + col * 720
        request = slot["request"]
        protocol_label = "P0 原D0" if request["protocol"].startswith("P0") else "P1 仅D1候选"
        label = f"{protocol_label} | {request['letter']} {request.get('initial_space','')} | {request.get('action_variant','')}"
        text(draw, (x, 72), label, 17)
        if not slot.get("available") or i >= len(slot["frames"]):
            draw.rectangle((x, 104, x + 688, 794), fill="#253242")
            text(draw, (x + 28, 300), "未执行 / 早停后无记录", 29, "white")
            text(draw, (x + 28, 345), "此为审阅占位牌，不是物理数据帧", 22, "#ffd078")
            text(draw, (x + 28, 388), slot.get("missing_reason", "短轨迹已结束"), 16, "white")
            continue
        raw = np.moveaxis(slot["frames"][i], 0, -1)
        image = Image.fromarray(raw)
        canvas.paste(image, (x, 110))
        canvas.paste(image.resize((576, 576), Image.Resampling.NEAREST), (x + 112, 218))
        if not np.array_equal(np.asarray(canvas.crop((x, 110, x + 96, 206))), raw):
            raise RuntimeError("Canvas altered the native96 image region")
        text(draw, (x + 112, 103), f"96px原像素；6×仅显示放大；物理跨度{slot['physical_span_seconds']:.1f}s", 16)
        if i == 0:
            text(draw, (x + 112, 132), "初始帧：尚无前一控制的接触/位移/转角", 17)
        else:
            r = slot["transitions"][i - 1]
            text(draw, (x + 112, 129), f"控制{i-1}  Δ面积质心={r['translation_image_px']:.3f}px  Δθ={r['signed_rotation_deg']:+.3f}°", 17)
            text(draw, (x + 112, 153), f"实际接触={r['contact_count']}  真实弧段={r['contact_arc_bins']}  阶段={r['action_phase']}", 15)
            pen = r["max_substep_penetrations"]
            ax, ay = r["absolute_action_xy"]
            text(draw, (x + 112, 177), f"穿透max={short(r['penetration_max'])}worldpx  valid={r['valid']}  action=({ax:.1f},{ay:.1f})", 14)
            samples = r["actual_contact_point_samples"]
            if samples is None:
                label = "实际接触点world坐标：未记录（不伪造）"
            else:
                preferred = sorted(samples, key=lambda p: 0 if p.get("phase") == "post_solve" else 1)
                coordinates = [p.get("object_world") for p in preferred if p.get("object_world") is not None][:2]
                label = f"实际点样本共{len(samples)}个；world " + " / ".join(f"({float(p[0]):.1f},{float(p[1]):.1f})" for p in coordinates)
            text(draw, (x + 112, 199), label, 14)
        gaps = slot["gaps"][i]
        text(draw, (x, 800), "实际四墙gap(worldpx): " + " / ".join(f"{k}={v:.2f}" for k, v in zip(WALLS, gaps)), 14)
    canvas.paste(curves, (0, 830))
    return np.asarray(canvas)


def _without_arrays(slot):
    excluded = {"arrays", "frames", "metadata", "gaps", "centers", "transitions", "request", "index"}
    return {k: v for k, v in slot.items() if k not in excluded}


def frame_fields(slot):
    rows = [dict(row_kind="initial_boundary_frame", frame_index=0, control_index=None,
        time_seconds=0., frame_sha256=slot["frame_hashes"][0], actual_contact_point_samples=None)]
    for i, control in enumerate(slot["transitions"], 1):
        rows.append(dict(row_kind="complete_boundary_frame", frame_index=i,
            control_index=i - 1, time_seconds=i / FPS, frame_sha256=slot["frame_hashes"][i],
            actual_contact_point_samples=control["actual_contact_point_samples"]))
    for failed in slot["failed_tail_records"]:
        rows.append(dict(row_kind="failed_control_evidence_NOT_a_frame", frame_index=None,
            control_index=failed.get("control_index"), time_seconds=None, frame_sha256=None,
            actual_contact_point_samples=failed.get("actual_contact_point_samples")))
    return rows


def build_media(r2a_root, *, index_path=None, requests_path=None, output=None, confirmed_cold=False, max_r2a_bytes=128 * 1024 ** 2):
    started = time.monotonic()
    root = Path(r2a_root).resolve()
    index_path = Path(index_path).resolve() if index_path else root / "index.json"
    requests_path = Path(requests_path).resolve() if requests_path else root.parent / "r1/r2-probe-requests.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not confirmed_cold and not index.get("collection_finished", False):
        raise ValueError("Require confirmed cold collection; never read a running partial collection")
    storage_guard(root, max_r2a_bytes)
    source = json.loads(requests_path.read_text(encoding="utf-8"))
    requests = source["requests"] if isinstance(source, dict) else source
    requests = [r for r in requests if r.get("phase") == "R2A"]
    if len(requests) != 64 or len({r["request_id"] for r in requests}) != 64:
        raise ValueError("Frozen R2A denominator must remain64 unique planned slots")
    indexed = index["slots"]
    indexed = list(indexed.values()) if isinstance(indexed, dict) else indexed
    lookup = {str(r.get("slot_id", r.get("request_id"))): r for r in indexed}
    if len(lookup) != len(indexed):
        raise ValueError("Duplicate index slot identities")
    output = inside(root, output or "media")
    if output.exists():
        raise FileExistsError("Media output must be new; preserve previous attempt evidence")
    output.mkdir(parents=True)
    for directory in ("slots", "pairs", "curves"):
        (output / directory).mkdir()
    slots = []
    for request in requests:
        sid = request["request_id"]
        row = lookup.get(sid, dict(slot_id=sid, attempted=False, status="not_executed", missing_from_index=True))
        slot = load_slot(root, row, request)
        if slot.get("available"):
            directory = output / "slots" / safe_id(slot["slot_id"])
            directory.mkdir()
            frames = []
            for i, raw in enumerate(slot["frames"]):
                p = directory / f"frame-{i:03d}.png"
                Image.fromarray(np.moveaxis(raw, 0, -1)).save(p)
                frames.append(dict(frame=i, state_index=i, time_seconds=i / FPS, chw_uint8_sha256=slot["frame_hashes"][i], png_file=p.relative_to(output).as_posix(), png_sha256=sha(p)))
            native = directory / "native96-10fps-1x-lossless.mkv"
            encode(native, (np.moveaxis(frame, 0, -1) for frame in slot["frames"]), lossless=True)
            count = verify_lossless(native, slot["frames"])
            slot.update(native_video=native.relative_to(output).as_posix(), native_video_sha256=sha(native), lossless_decoded_frames_verified=count)
            json_write(directory / "frames.json", dict(frames=frames, physical_span_seconds=slot["physical_span_seconds"], encoded_duration_seconds=slot["encoded_duration_seconds"], final_endpoint_hold_seconds=.1, raw_arrays_source=slot["npz_file"], no_extra_physics_render=True))
            json_write(directory / "controls.json", slot["transitions"])
            fields_for_frames = frame_fields(slot)
            json_write(directory / "frame-fields.json", dict(
                note="All actual recorded contact samples retained; repeated100Hz points are samples, not independent arcs/trajectories. Failed control evidence is not a new10Hz frame.",
                frame_and_evidence_fields=fields_for_frames, source_npz=slot["npz_file"], source_metadata=slot["metadata_file"]))
            with (directory / "frame-fields.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(fields_for_frames[0]))
                writer.writeheader()
                for row in fields_for_frames:
                    writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
            json_write(directory / "partial-evidence.json", {k: slot[k] for k in (
                "slot_id", "failure", "control_records_total", "completed_record_count", "missing_completed_record_controls",
                "failed_tail_records", "unclassified_control_records", "partial_endpoint", "partial_control_span_seconds",
                "partial_time_exact", "partial_state_substep_indices", "unavailable_partial_state_substeps", "partial_states_present", "partial_states_saved_count",
                "partial_rgb_available", "partial_endpoint_is_boundary_frame", "npz_file", "npz_sha256", "metadata_file", "metadata_sha256",
                "telemetry_file", "telemetry_sha256")})
            storage_guard(root, max_r2a_bytes)
        slots.append(slot)
    grouped = {}
    for slot in slots:
        request = slot["request"]
        grouped.setdefault((request["family_id"], request["action_variant"]), []).append(slot)
    if len(grouped) != 32 or any(len(v) != 2 for v in grouped.values()):
        raise ValueError("Expected32 complete planned protocol pairs")
    pairs = []
    all_controls = []
    for (family, variant), members in grouped.items():
        members.sort(key=lambda s: s["request"]["protocol"])
        left, right = members
        a, b = left["request"], right["request"]
        planned_equal = a["initial_parameters_sha256"] == b["initial_parameters_sha256"] and a["actions_sha256"] == b["actions_sha256"] and a["actions_absolute"] == b["actions_absolute"]
        actual_equal = bool(left.get("available") and right.get("available") and np.array_equal(left["arrays"]["states"][0], right["arrays"]["states"][0]))
        if not planned_equal:
            raise ValueError("Planned protocol pair does not share frozen initial/action input")
        pair_id = safe_id(f"{family}-{variant}")
        curves = curve_canvas(members)
        curve_file = output / "curves" / f"{pair_id}.png"
        curves.save(curve_file)
        count = max([len(s.get("frames", [])) for s in members] + [1])
        movie = output / "pairs" / f"{pair_id}-1x-6x.mp4"
        encode(movie, (pair_frame(members, i, curves) for i in range(count)))
        timing = verify_review_timing(movie, count)
        first = output / "pairs" / f"{pair_id}-first.png"
        Image.fromarray(pair_frame(members, 0, curves)).save(first)
        pairs.append(dict(pair_id=pair_id, family_id=family, action_variant=variant, slots=[s["slot_id"] for s in members],
            planned_initial_and_actions_equal=planned_equal, actual_initial_states_equal=actual_equal,
            actual_initial_comparison_assessable=bool(left.get("available") and right.get("available")),
            review_video=movie.relative_to(output).as_posix(), review_video_sha256=sha(movie), poster=first.relative_to(output).as_posix(),
            curves=curve_file.relative_to(output).as_posix(), frames=count, fps=FPS, display_duration_seconds=count / FPS,
            missing_slate_is_not_data=True, lossy_review_encode=True, decoded_timing=timing))
        for member in members:
            all_controls.extend(member.get("transitions", []))
        storage_guard(root, max_r2a_bytes)
    fields = list(all_controls[0]) if all_controls else ["slot_id", "control"]
    with (output / "controls.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_controls:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    with (output / "slots.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        columns = ("slot_id", "letter", "protocol", "planned", "attempted", "status", "available", "controls", "failure", "native_video", "physical_span_seconds", "encoded_duration_seconds", "final_endpoint_hold_seconds")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for slot in slots:
            writer.writerow({k: (slot["request"].get(k) if k in ("letter", "protocol") else True if k == "planned" else slot.get(k)) for k in columns})
    result = dict(kind="R2A-saved-RGB-only-media-v1", source_script_sha256=sha(__file__), index_sha256=sha(index_path), requests_sha256=sha(requests_path),
        planned_slots=64, planned_pairs=32, saved_slots=sum(s["available"] for s in slots), attempted_slots=sum(s.get("attempted") is True for s in slots),
        completed_controls=sum(s.get("controls", 0) for s in slots), slots=[_without_arrays(s) for s in slots], pairs=pairs,
        fps=FPS, prediction_interval_seconds=.1, display_scale=SCALE, display_interpolation="nearest", raw_codec="FFV1 lossless BGR0 MKV decoded toRGB verified", review_codec="H264yuv420p CRF16 display only",
        raw_pixels_preserved_in_PNG_and_native_video=True, all_planned_slots_retained=True, no_frame_interpolation_or_time_change=True,
        missing_slate_is_not_physics_data=True, model_calls=0, simulator_steps=0, extra_physics_render_calls=0,
        supplied_frame_hashes_missing_slots=[s["slot_id"] for s in slots if s.get("available") and not s["declared_frame_hashes_verified"]])
    sections = []
    for pair in pairs:
        links = " ".join(f'<a href="slots/{html.escape(sid)}/frames.json">{html.escape(sid)}原帧</a><a href="slots/{html.escape(sid)}/native96-10fps-1x-lossless.mkv">原96无损视频(MKV)</a><a href="slots/{html.escape(sid)}/frame-fields.json">全部接触点JSON</a><a href="slots/{html.escape(sid)}/frame-fields.csv">全部接触点CSV</a>' if next(s for s in slots if s['slot_id'] == sid).get('available') else f"{html.escape(sid)}未记录" for sid in pair["slots"])
        details = []
        for sid in pair["slots"]:
            s = next(s for s in slots if s["slot_id"] == sid)
            details.append(html.escape(sid) + ": " + html.escape(str(s.get("status"))))
            if s.get("available"):
                span = s.get("partial_control_span_seconds")
                partial_text = "未知（未补零）" if span is None else f"{span:.3f}秒（不计入完整10Hz帧）"
                details.append(f'<a href="slots/{html.escape(sid)}/partial-evidence.json">失败尾/partial证据</a> partial时长：{partial_text}；已存partial状态数={s["partial_states_saved_count"]}；未取得状态子步={html.escape(str(s["unavailable_partial_state_substeps"]))}')
                for key, label in (("npz_file", "原始NPZ"), ("metadata_file", "原始metadata"), ("telemetry_file", "100Hz原始日志")):
                    if s.get(key):
                        href = Path(os.path.relpath(inside(root, s[key]), output)).as_posix()
                        details.append(f'<a href="{html.escape(href, quote=True)}">{label}</a>')
        details = "<br>".join(details)
        sections.append(f'<section><h2>{html.escape(pair["pair_id"])}</h2><p>计划配对一致：{pair["planned_initial_and_actions_equal"]}；实际初态可比较：{pair["actual_initial_comparison_assessable"]}；一致：{pair["actual_initial_states_equal"]}</p><p>{details}</p><video controls preload="none" poster="{pair["poster"]}" src="{pair["review_video"]}"></video><p>{links} · <a href="{pair["curves"]}">逐步曲线</a></p></section>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>R2A 全部32对真实连续记录</title><style>body{font-family:Microsoft YaHei,sans-serif;background:#edf1f4;color:#17212b;margin:24px}section{background:white;padding:16px;margin:20px 0}video{width:100%;max-width:1440px}a{margin-right:16px}</style><h1>R2A：全部64计划槽 / 32对审阅</h1><p>10Hz、1×真实时间，单步0.1秒；原96px+明确6×nearest显示放大。原PNG/FFV1无损，网页H264仅审阅编码。缺失/早停占位牌不是物理帧；末端0.1秒停留不算运动。没有模型预测、插帧、重绘或额外物理渲染。</p><p><a href="manifest.json">全槽状态/SHA/真实跨度</a><a href="slots.csv">全部计划槽/缺失/失败分母</a><a href="controls.csv">全部逐控制数值</a></p>' + ''.join(sections) + '</html>'
    (output / "index.html").write_text(page, encoding="utf-8")
    result["cpu_wall_seconds"] = time.monotonic() - started
    result["r2a_bytes_before_final_manifest"] = storage_guard(root, max_r2a_bytes)
    json_write(output / "manifest.json", result)
    storage_guard(root, max_r2a_bytes)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--index")
    parser.add_argument("--requests")
    parser.add_argument("--output", default="media")
    parser.add_argument("--confirmed-cold", action="store_true")
    parser.add_argument("--max-r2a-bytes", type=int, default=128 * 1024 ** 2)
    args = parser.parse_args()
    result = build_media(args.root, index_path=args.index, requests_path=args.requests, output=args.output, confirmed_cold=args.confirmed_cold, max_r2a_bytes=args.max_r2a_bytes)
    print(json.dumps({k: result[k] for k in ("planned_slots", "planned_pairs", "saved_slots", "completed_controls", "cpu_wall_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
