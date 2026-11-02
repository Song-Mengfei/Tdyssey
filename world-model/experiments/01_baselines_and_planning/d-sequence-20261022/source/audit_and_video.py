"""Independent actual-action replay audit and fixed-pause videos; no policy imports.

Run from the repository root. Outputs always require a new directory. The video
holds each recorded actual control for a fixed playback interval; it does not
represent planning latency or interpolate physical states.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tdyssey.envs.official import make_official_shape_env_class
from tdyssey.official import load_official_source, physical_metrics
from tdyssey.simulator_planning import state_error
from tdyssey.state import Snapshot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_reference(value, parent):
    path = Path(value)
    candidates = [path] if path.is_absolute() else [ROOT / path, parent / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"Cannot resolve referenced report: {value}")


def load_record(path, shape, episode_index, seen=()):
    """Return all source rows needed for contact reconstruction plus scored rows."""
    path = Path(path).resolve()
    if path in seen:
        raise ValueError("Cyclic resume-report references")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "episodes" in data:
        episodes = [item for item in data["episodes"] if item["shape"] == shape]
        if episode_index is None and len(episodes) != 1:
            raise ValueError("Select --episode-index when a report has multiple matching episodes")
        index = 0 if episode_index is None else episode_index
        if not 0 <= index < len(episodes):
            raise ValueError("Requested shape/episode is unavailable")
        episode = episodes[index]
    else:
        episode = data
    initial = Snapshot.from_dict(episode["initial_state"])
    if initial.shape != shape or initial.control_step != 0:
        raise ValueError("Audit requires the requested shape's episode-initial control-zero snapshot")
    start = Snapshot.from_dict(episode.get("search_start_state", episode["initial_state"]))
    prefix, sources = [], {str(path): digest(path)}
    if start.control_step:
        config = data.get("config", {})
        reference = resolve_reference(config["resume_report"], path.parent)
        prior = load_record(reference, shape, episode_index, (*seen, path))
        if state_error(prior["initial"], initial) > 1e-7:
            raise ValueError("Resume report and source initial states differ")
        all_prior = prior["prefix"] + prior["rows"]
        prefix = all_prior[:start.control_step]
        if len(prefix) != start.control_step or config.get("resume_step") != start.control_step:
            raise ValueError("Resume source does not contain the complete requested prefix")
        sources.update(prior["sources"])
    rows = [{"source": str(path), "source_row": index, "row": row}
            for index, row in enumerate(episode["steps"])]
    return {"path": path, "data": data, "episode": episode, "initial": initial,
            "start": start, "prefix": prefix, "rows": rows, "sources": sources}


def result_label(data, episode):
    if episode.get("success"):
        return "NATIVE SUCCESS"
    status, reason = data.get("status", "unknown"), episode.get("stop_reason", "unknown")
    if status == "running":
        return "INTERRUPTED SNAPSHOT (report still marked running)"
    if status == "failed":
        return f"RUN FAILED: {reason}"
    if reason == "wall_time_budget":
        return "TIMEOUT / NO NATIVE SUCCESS"
    if reason == "control_budget":
        return "CONTROL BUDGET / NO NATIVE SUCCESS"
    return f"{status.upper()}: {reason}"


def bearing(dx, dy):
    names = ("east", "southeast", "south", "southwest", "west", "northwest", "north", "northeast")
    return names[int(math.floor((math.atan2(dy, dx) + math.pi / 8) / (math.pi / 4))) % 8]


def route_summary(initial, transitions):
    previous, path_length, angular_path, contact_runs, active = initial, 0.0, 0.0, [], None
    sides = Counter()
    for item in transitions:
        current = Snapshot.from_dict(item["state"])
        before, after = previous.object_body.pose, current.object_body.pose
        dx, dy = after.position[0] - before.position[0], after.position[1] - before.position[1]
        path_length += math.hypot(dx, dy)
        angular_path += abs(after.angle - before.angle)
        if item["n_contacts"] > 0:
            px = previous.pusher_body.pose.position[0] - before.position[0]
            py = previous.pusher_body.pose.position[1] - before.position[1]
            c, s = math.cos(before.angle), math.sin(before.angle)
            side = bearing(c * px + s * py, -s * px + c * py)
            sides[side] += 1
            if active is None:
                active = {"start_control": current.control_step, "end_control": current.control_step,
                          "controls": 0, "bearing_counts": {}, "translation": [0.0, 0.0], "angle_change": 0.0}
                contact_runs.append(active)
            active["end_control"] = current.control_step
            active["controls"] += 1
            active["bearing_counts"][side] = active["bearing_counts"].get(side, 0) + 1
            active["translation"][0] += dx
            active["translation"][1] += dy
            active["angle_change"] += after.angle - before.angle
        else:
            active = None
        previous = current
    for run in contact_runs:
        run["dominant_bearing"] = max(run["bearing_counts"], key=run["bearing_counts"].get)
    start, end = initial.object_body.pose, previous.object_body.pose
    translation = [end.position[0] - start.position[0], end.position[1] - start.position[1]]
    return {"descriptor": "executed pusher bearing about object COM before contact; not contact-normal inference",
            "contact_controls": sum(sides.values()), "contact_bearing_counts": dict(sides),
            "contact_runs": contact_runs, "route_signature": "/".join(run["dominant_bearing"] for run in contact_runs),
            "object_translation": translation, "object_translation_distance": math.hypot(*translation),
            "object_translation_path": path_length, "object_angle_change": end.angle - start.angle,
            "object_absolute_angle_path": angular_path}


class PausedVideo:
    def __init__(self, path, args):
        import cv2

        self.cv2, self.path, self.args = cv2, Path(path), args
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                      (args.size, args.size))
        if not self.writer.isOpened():
            raise RuntimeError("Cannot open MP4 encoder")
        self.frames, self.groups, self.references, self.last_frame = 0, [], [], None
        self.finished = False

    def make_frame(self, rgb, snapshot, metrics, outcome, phase, audit_ok):
        import numpy as np

        cv2, size = self.cv2, self.args.size
        canvas = np.full((size, size, 3), 247, dtype=np.uint8)
        board_size = size - 128
        left, top = (size - board_size) // 2, 76
        board = cv2.cvtColor(cv2.resize(rgb, (board_size, board_size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
        canvas[top:top + board_size, left:left + board_size] = board
        scale = board_size / 512
        def pixel(position):
            return (int(round(left + position[0] * scale)), int(round(top + position[1] * scale)))
        actual = snapshot.pusher_body.pose.position
        cv2.circle(canvas, pixel(actual), max(5, int(9 * scale)), (170, 70, 10), 2)
        if snapshot.pusher_target is not None:
            target = pixel(snapshot.pusher_target)
            cv2.drawMarker(canvas, target, (20, 100, 240), cv2.MARKER_CROSS, 15, 2)
        text_scale = 0.45 if size == 512 else 0.55
        def line(text, y, color=(35, 35, 35)):
            cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, 1, cv2.LINE_AA)
        line(f"D | actual control {snapshot.control_step:03d} | {phase} | {'AUDITED' if audit_ok else 'AUDIT MISMATCH'}", 21,
             (30, 110, 30) if audit_ok else (20, 20, 210))
        line(f"Coverage {metrics['coverage']:.4f} | IoU {metrics['iou']:.4f} | native success > 0.95", 43)
        line(outcome[:95 if size == 768 else 69], 65)
        y = top + board_size + 19
        target = snapshot.pusher_target
        line(f"Pusher (blue): ({actual[0]:.1f}, {actual[1]:.1f}) | target (orange): " +
             (f"({target[0]:.1f}, {target[1]:.1f})" if target else "none"), y)
        line(f"Fixed playback: {self.args.pause_frames / self.args.fps:.2f}s/control; planning time omitted", y + 20)
        return canvas

    def append(self, frame, snapshot, count, phase):
        if count < 1:
            return
        group = {"first_frame": self.frames, "frame_count": count, "control": snapshot.control_step,
                 "phase": phase, "state_sha256": json_digest(asdict(snapshot)),
                 "uncompressed_display_bgr_sha256": hashlib.sha256(frame.tobytes()).hexdigest()}
        self.groups.append(group)
        self.references.append(self.cv2.resize(frame, (192, 192), interpolation=self.cv2.INTER_AREA))
        for _ in range(count):
            self.writer.write(frame)
            self.frames += 1
        self.last_frame = frame

    def close(self):
        if self.finished:
            return
        self.writer.release()
        self.finished = True

    def verify(self):
        import numpy as np

        self.close()
        cv2, decoded, maximum_error, group_index = self.cv2, 0, 0.0, 0
        capture = cv2.VideoCapture(str(self.path))
        fps, width, height = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        try:
            while True:
                okay, frame = capture.read()
                if not okay:
                    break
                if decoded >= self.frames:
                    raise ValueError("Encoded video has unexpected extra frames")
                while decoded >= self.groups[group_index]["first_frame"] + self.groups[group_index]["frame_count"]:
                    group_index += 1
                thumb = cv2.resize(frame, (192, 192), interpolation=cv2.INTER_AREA)
                difference = float(np.abs(thumb.astype(np.float32) - self.references[group_index]).mean())
                maximum_error = max(maximum_error, difference)
                if difference > 6.0:
                    raise ValueError(f"Encoded frame {decoded} differs from its source-rendered reference: {difference}")
                decoded += 1
        finally:
            capture.release()
        if decoded != self.frames or abs(fps - self.args.fps) > 1e-5 or (width, height) != (self.args.size, self.args.size):
            raise ValueError("Encoded video frame count, rate or size differs from the playback manifest")
        return {"path": str(self.path.resolve()), "sha256": digest(self.path), "verified": True,
                "timing": "fixed_pause_per_actual_control_not_wall_time_or_interpolated_motion",
                "pause_seconds": self.args.pause_frames / self.args.fps, "fps": fps,
                "width": width, "height": height, "frames": decoded, "duration_seconds": decoded / fps,
                "decoded_reference_thumbnail_max_mean_absolute_error": maximum_error,
                "verification_thumbnail_size": [192, 192], "maximum_allowed_mean_absolute_error": 6.0,
                "frame_groups": self.groups}


def audit_one(path, directory, args, base):
    import numpy as np

    started = time.monotonic()
    audit = {"schema_version": 1, "status": "running", "source_report": str(Path(path).resolve()),
             "training": False, "checkpoint_loaded": False, "rows": [], "issues": [],
             "counts": {"reset_physics_substeps": 0, "prefix_controls": 0, "source_controls": 0,
                        "replayed_control_physics_substeps": 0}}
    env, video = None, None
    try:
        record = load_record(path, args.shape, args.episode_index)
        data, episode = record["data"], record["episode"]
        audit.update(source_report_sha256=record["sources"][str(record["path"])],
                     source_status=data.get("status"), source_stop_reason=episode.get("stop_reason"),
                     source_success=bool(episode.get("success")), source_reports=record["sources"],
                     episode_initial_state=asdict(record["initial"]), search_start_state=asdict(record["start"]),
                     search_start_state_sha256=json_digest(asdict(record["start"])),
                     source_counts=data.get("counts"), source_search_counts=data.get("search_counts"),
                     source_provenance=data.get("provenance"), result=result_label(data, episode))
        source_hashes = data.get("provenance", {}).get("source_sha256", {})
        audit["current_source_sha256"] = {name: digest(ROOT / name) for name in source_hashes if (ROOT / name).is_file()}
        audit["source_files_changed_since_run"] = [name for name, value in source_hashes.items()
                                                     if audit["current_source_sha256"].get(name) != value]

        class ReplayEnv(base):
            def _get_obs(self):
                return {}

            def step(self, action):
                return super().step(np.asarray(action, dtype=np.float64))

            def render(self, mode):
                return self._render_frame(mode)

        env = ReplayEnv(shape=args.shape, legacy=True, render_size=512)
        env.seed(episode.get("scene_seed", data.get("scene_seed", 100000)))
        env.reset()
        audit["counts"]["reset_physics_substeps"] += 1
        env.restore_state(record["initial"])
        if state_error(env.capture_state(), record["initial"]) > args.state_tolerance:
            raise ValueError("Restored episode-initial state differs")
        audit["initial_metrics"] = physical_metrics(env)
        previous = record["initial"]
        video = PausedVideo(directory / "actual-controls-paused.mp4", args) if not args.audit_only else None
        rendered_initial = False

        def draw(phase, count):
            if not video:
                return
            state = env.capture_state()
            frame = video.make_frame(env.render("rgb_array"), state, physical_metrics(env), audit["result"], phase,
                                     not audit["issues"])
            video.append(frame, state, count, phase)
            if not rendered_initial:
                video.cv2.imwrite(str(directory / "initial.png"), frame)

        if args.include_resume_prefix:
            draw("episode initial", args.initial_hold_frames)
            rendered_initial = True
        prediction_map = {}
        for decision in data.get("decisions", []):
            for index, predicted in enumerate(decision.get("selected_predictions", [])):
                control = predicted["control_step"]
                action = decision.get("selected_actions", [])[index] if index < len(decision.get("selected_actions", [])) else None
                prediction_map[control] = (predicted, action)
        actual_rows, native_terminal, checked_start = [], False, False
        all_rows = [("prefix", item) for item in record["prefix"]] + [("source", item) for item in record["rows"]]
        for phase, item in all_rows:
            if native_terminal:
                audit["issues"].append(f"Source contains controls after native success at {previous.control_step}")
                break
            if phase == "source" and not checked_start:
                error = state_error(env.capture_state(), record["start"])
                audit["search_start_state_error"] = error
                if error > args.state_tolerance:
                    audit["issues"].append(f"Search-start state differs by {error}")
                expected_initial_metrics = episode.get("initial_metrics")
                if expected_initial_metrics:
                    measured = physical_metrics(env)
                    for name in ("coverage", "iou"):
                        if abs(measured[name] - expected_initial_metrics[name]) > 1e-9:
                            audit["issues"].append(f"Search-start {name} differs")
                if not rendered_initial:
                    draw("search initial" if record["prefix"] else "episode initial", args.initial_hold_frames)
                    rendered_initial = True
                checked_start = True
            row = item["row"]
            _, reward, done, info = env.step(row["action"])
            audit["counts"][f"{phase}_controls"] += 1
            audit["counts"]["replayed_control_physics_substeps"] += env.sim_hz // env.control_hz
            state, measured = env.capture_state(), physical_metrics(env)
            check = {"phase": phase, "control": state.control_step, "source": item["source"],
                     "source_row": item["source_row"], "action": list(row["action"]),
                     "reward": float(reward), "native_done": bool(done), "n_contacts": int(info["n_contacts"]),
                     "coverage": measured["coverage"], "iou": measured["iou"], "mismatches": []}
            try:
                check["state_error"] = state_error(state, Snapshot.from_dict(row["state"]))
                if check["state_error"] > args.state_tolerance:
                    check["mismatches"].append("state")
            except Exception as exc:
                check["state_error_detail"] = f"{type(exc).__name__}: {exc}"
                check["mismatches"].append("state")
            check["reward_error"] = abs(float(reward) - float(row["reward"]))
            if check["reward_error"] > 1e-9:
                check["mismatches"].append("reward")
            if int(info["n_contacts"]) != row.get("n_contacts"):
                check["mismatches"].append("contacts")
            if bool(done) != row.get("upstream_done", row.get("native_done")):
                check["mismatches"].append("native_done")
            if bool(done) != (measured["coverage"] > 0.95):
                check["mismatches"].append("native_success_rule")
            for name in ("coverage", "iou"):
                expected = row.get("metrics", {}).get(name)
                if expected is None or abs(measured[name] - expected) > 1e-9:
                    check["mismatches"].append(name)
            if phase == "source" and state.control_step in prediction_map:
                predicted, planned_action = prediction_map[state.control_step]
                try:
                    check["selected_prediction_error"] = state_error(state, Snapshot.from_dict(predicted))
                    if check["selected_prediction_error"] > args.state_tolerance:
                        check["mismatches"].append("selected_prediction")
                except Exception as exc:
                    check["selected_prediction_detail"] = f"{type(exc).__name__}: {exc}"
                    check["mismatches"].append("selected_prediction")
                if planned_action is not None and max(abs(a - b) for a, b in zip(planned_action, row["action"])) > 1e-9:
                    check["mismatches"].append("selected_action")
            if check["mismatches"]:
                audit["issues"].append(f"{phase} control {state.control_step}: {', '.join(check['mismatches'])}")
            audit["rows"].append(check)
            if phase == "source":
                actual_rows.append({"state": asdict(state), "n_contacts": int(info["n_contacts"])})
            if phase == "source" or args.include_resume_prefix:
                draw("recorded teacher" if phase == "source" and record["prefix"] else "recorded actual", args.pause_frames)
            previous, native_terminal = state, bool(done)
        if not checked_start:
            audit["search_start_state_error"] = state_error(env.capture_state(), record["start"])
            if audit["search_start_state_error"] > args.state_tolerance:
                audit["issues"].append("Search-start state differs for empty source trajectory")
            if not rendered_initial:
                draw("search initial", args.initial_hold_frames)
                rendered_initial = True
        audit["native_success"] = native_terminal
        if native_terminal != bool(episode.get("success")):
            audit["issues"].append("Source success flag differs from independently replayed native success")
        audit["final_state"], audit["final_metrics"] = asdict(env.capture_state()), physical_metrics(env)
        audit["final_control"] = env.control_step
        expected_controls = episode.get("control_steps")
        if "episodes" not in data:
            expected_controls = data.get("counts", {}).get("control_steps")
        if expected_controls is not None and expected_controls != audit["counts"]["source_controls"]:
            audit["issues"].append("Reported actual-control count differs from replayed source controls")
        for name, value in episode.get("final_metrics", {}).items():
            if name in audit["final_metrics"] and abs(value - audit["final_metrics"][name]) > 1e-9:
                audit["issues"].append(f"Reported final {name} differs")
        if (data.get("status") == "complete" and episode.get("stop_reason") == "control_budget"
                and env.control_step != data.get("config", {}).get("steps")):
            audit["issues"].append("Completed control-budget report did not reach its episode control budget")
        audit["route"] = route_summary(record["start"], actual_rows)
        if video:
            draw("final result", args.final_hold_frames)
            video.cv2.imwrite(str(directory / "final.png"), video.last_frame)
            audit["video"] = video.verify()
        for source, expected_hash in record["sources"].items():
            if digest(source) != expected_hash:
                audit["issues"].append(f"Source report changed during audit: {source}")
        audit["status"] = "passed" if not audit["issues"] else "mismatch"
    except Exception as exc:
        audit["status"], audit["error"] = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        if video:
            video.close()
        if env:
            env.close()
        audit["elapsed_seconds"] = time.monotonic() - started
        audit["torch_loaded"] = "torch" in sys.modules
        if audit["torch_loaded"]:
            audit["issues"].append("Torch was unexpectedly imported")
            audit["status"] = "failed"
        save_json(directory / "audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT / "third_party/diffusion_policy")
    parser.add_argument("--shape", default="D")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--fps", type=int, choices=(5, 10), default=10)
    parser.add_argument("--pause-seconds", type=float, default=0.3)
    parser.add_argument("--initial-hold-seconds", type=float, default=1.0)
    parser.add_argument("--final-hold-seconds", type=float, default=2.0)
    parser.add_argument("--size", type=int, choices=(512, 768), default=768)
    parser.add_argument("--state-tolerance", type=float, default=1e-7)
    parser.add_argument("--include-resume-prefix", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    for name in ("pause_seconds", "initial_hold_seconds", "final_hold_seconds", "state_tolerance"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    if args.pause_seconds > 5 or max(args.initial_hold_seconds, args.final_hold_seconds) > 30:
        parser.error("Use bounded playback holds")
    if args.shape != "D":
        parser.error("This D-teacher helper only accepts shape D")
    args.pause_frames = max(1, round(args.pause_seconds * args.fps))
    args.initial_hold_frames = max(1, round(args.initial_hold_seconds * args.fps))
    args.final_hold_frames = max(1, round(args.final_hold_seconds * args.fps))
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory; old outputs are never overwritten")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    base = make_official_shape_env_class(load_official_source(args.source_root))
    output.mkdir(parents=True, exist_ok=False)
    audits = []
    for index, path in enumerate(args.report):
        directory = output / f"{index:02d}-{path.resolve().parent.name}"
        directory.mkdir()
        audited = audit_one(path, directory, args, base)
        audits.append(audited)
        print(json.dumps({"report": str(path), "status": audited["status"], "issues": audited["issues"],
                          "error": audited.get("error"), "counts": audited["counts"],
                          "video": audited.get("video", {}).get("path")}), flush=True)
    groups = {}
    for audit in audits:
        key = audit.get("search_start_state_sha256")
        if key:
            group = groups.setdefault(key, {"reports": 0, "successful_routes": [], "all_routes": []})
            group["reports"] += 1
            signature = audit.get("route", {}).get("route_signature", "")
            group["all_routes"].append(signature)
            if audit.get("native_success") and audit["status"] == "passed":
                group["successful_routes"].append(signature)
    for group in groups.values():
        group["unique_observed_route_signatures"] = len(set(group["all_routes"]))
        group["unique_successful_route_signatures"] = len(set(group["successful_routes"]))
    summary = {"schema_version": 1, "kind": "independent_D_teacher_replay_audit", "training": False,
               "checkpoint_loaded": False, "torch_loaded": "torch" in sys.modules,
               "all_passed": all(item["status"] == "passed" for item in audits),
               "provenance": {"script": str(Path(__file__).resolve()), "script_sha256": digest(__file__),
                              "python": sys.version, "argv": sys.argv,
                              "commit": subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                                                                 "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                              "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()
                                           if dist.metadata["Name"].lower() in ("numpy", "opencv-python", "opencv-python-headless",
                                                                                "pymunk", "pygame", "shapely")}},
               "route_diversity_by_exact_search_start": groups,
               "reports": [{key: value for key, value in item.items() if key not in ("rows", "source_provenance")}
                           for item in audits]}
    save_json(output / "summary.json", summary)
    print(json.dumps({"summary": str(output / "summary.json"), "all_passed": summary["all_passed"]}))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
