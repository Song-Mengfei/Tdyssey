"""Bounded two-contact search in the real D simulator; no training."""

import argparse
from dataclasses import asdict
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.envs.official import make_official_shape_env_class, matched_reset
from tdyssey.official import load_official_source, physical_metrics
from tdyssey.policy import hash_file
from tdyssey.simulator_planning import state_error
from tdyssey.state import Snapshot


class WallTimeVideo:
    """Repeat the previous actual frame while planning; preserve visible pauses."""

    def __init__(self, path, frame, fps=10):
        import cv2

        self.cv2, self.fps = cv2, fps
        self.size = (512, 512)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)
        if not self.writer.isOpened():
            raise RuntimeError("Cannot encode wall-time video")
        self.previous, self.start, self.frames = cv2.resize(frame, self.size), time.monotonic(), 0

    def append(self, frame):
        target = int((time.monotonic() - self.start) * self.fps)
        while self.frames < target:
            self.writer.write(self.cv2.cvtColor(self.previous, self.cv2.COLOR_RGB2BGR))
            self.frames += 1
        self.previous = self.cv2.resize(frame, self.size)

    def close(self):
        self.append(self.previous)
        self.writer.write(self.cv2.cvtColor(self.previous, self.cv2.COLOR_RGB2BGR))
        self.frames += 1
        self.writer.release()


def resume_history(path, upto, seen=()):
    """Resolve baseline/search reports, including complete recursive histories."""
    path = Path(path).resolve()
    if path in seen:
        raise ValueError("Cyclic resume source")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") in ("running", "failed"):
        raise ValueError("Resume source must be finalized without protocol failure")
    episodes = [e for e in data.get("episodes", [data]) if e.get("shape", e["initial_state"]["shape"]) == "D"]
    if len(episodes) != 1:
        raise ValueError("Resume report must contain exactly one D episode")
    episode = episodes[0]
    initial = Snapshot.from_dict(episode["initial_state"])
    if initial.control_step != 0:
        raise ValueError("Resume source must preserve the control-zero snapshot")
    start = Snapshot.from_dict(episode.get("search_start_state", episode["initial_state"]))
    history, sources = [], {str(path): hash_file(path)}
    if start.control_step:
        reference = Path(data["config"]["resume_report"])
        if not reference.is_absolute():
            reference = ROOT/reference
        prior, _, history, prior_sources = resume_history(reference, start.control_step, (*seen, path))
        if state_error(prior, initial) > 1e-7:
            raise ValueError("Resume initial-state mismatch")
        sources.update(prior_sources)
    history += episode["steps"]
    if upto < 1 or len(history) < upto:
        raise ValueError("Resume report lacks requested actual controls")
    history = history[:upto]
    if any(row["state"]["control_step"] != i+1 or row.get("upstream_done", False) for i, row in enumerate(history)):
        raise ValueError("Resume history is discontinuous or already successful")
    return initial, episode["scene_seed"], history, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "third_party/diffusion_policy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300, help="Total episode controls, including any resumed history")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--max-global-searches", type=int, default=8)
    parser.add_argument("--max-evaluations", type=int, default=40)
    parser.add_argument("--settle-steps", type=int, default=4)
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--video", action="store_true", help="Record actual wall-time planning pauses")
    parser.add_argument("--hold-seconds", type=int, default=0)
    args = parser.parse_args()
    if not (1 <= args.steps <= 300 and 1 <= args.seconds <= 3600
            and 100000 <= args.seed <= 2**32 - 21 and 0 <= args.search_seed < 2**32
            and 0 <= args.resume_step < args.steps and 0 <= args.hold_seconds <= 60):
        parser.error("Use a bounded 1–300 control episode, 1–3600 seconds and valid seeds/resume step")
    if bool(args.resume_report) != bool(args.resume_step):
        parser.error("Use --resume-report with a positive --resume-step")
    if args.hold_seconds and not args.display:
        parser.error("--hold-seconds requires --display")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory")
    if not args.display:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    elif os.environ.get("SDL_VIDEODRIVER") == "dummy":
        parser.error("Remove SDL_VIDEODRIVER=dummy for a visible window")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import numpy as np
    from tdyssey.simulator_sequence import SequenceConfig, ContactSequenceSearch, SearchBudgetExhausted, pose_rms

    config = SequenceConfig(horizon=args.horizon, max_evaluations=args.max_evaluations,
                            max_global_searches=args.max_global_searches, seed=args.search_seed,
                            settle_steps=args.settle_steps)
    start = time.monotonic()
    base = make_official_shape_env_class(load_official_source(args.source_root))

    class PhysicsOnly(base):
        def _get_obs(self):
            return {}

        def step(self, action):
            return super().step(np.asarray(action, dtype=np.float64))

        def render(self, mode):
            # Search branches skip observations; explicit live drawing must
            # still populate the original image environment's render cache.
            super()._get_obs()
            return super().render(mode)

    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "kind": "policy_free_privileged_contact_sequence_search",
              "status": "running", "config": {k: str(v) if isinstance(v, Path) else v
                                                for k, v in vars(args).items()},
              "search_config": asdict(config), "training": False, "checkpoint_loaded": False,
              "privileged_state_geometry_dynamics": True,
              "counts": {"control_steps": 0, "control_physics_substeps": 0,
                         "reset_physics_substeps": 0, "resume_replay_controls": 0},
              "decisions": [], "planning_attempts": [], "steps": [], "success": False, "stop_reason": "control_budget",
              "provenance": {"commit": subprocess.check_output(
                  ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), "rev-parse", "HEAD"],
                  text=True).strip(), "dirty": bool(subprocess.check_output(
                  ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), "status", "--porcelain"], text=True)),
                  "python": sys.version, "platform": sys.platform,
                  "source_sha256": {name: hash_file(ROOT / name) for name in (
                      "scripts/evaluate_contact_sequence.py", "tdyssey/simulator_sequence.py",
                      "tdyssey/contact_geometry.py", "tdyssey/simulator_search.py", "tdyssey/official.py",
                      "tdyssey/simulator_planning.py", "tdyssey/envs/official.py", "tdyssey/geometry.py",
                      "tdyssey/state.py", "tdyssey/evaluation.py", "manifests/official_dp.json", "pyproject.toml", "uv.lock")},
                  "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()}}}
    live, reference, planner, viewer, video = None, None, None, None, None

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    try:
        live = PhysicsOnly(shape="D", legacy=True)
        history = []
        if args.resume_report:
            initial, scene_seed, prefix_rows, sources = resume_history(args.resume_report, args.resume_step)
            report["resume_sources"] = sources
            live.seed(scene_seed)
            live.reset()
            report["counts"]["reset_physics_substeps"] += 1
            live.restore_state(initial)
            for row in prefix_rows:
                _, reward, done, info = live.step(row["action"])
                report["counts"]["resume_replay_controls"] += 1
                error = state_error(live.capture_state(), Snapshot.from_dict(row["state"]))
                if error > 1e-7 or abs(reward - row["reward"]) > 1e-9 or done or info["n_contacts"] != row["n_contacts"]:
                    raise ValueError(f"Resume history mismatch: {error}")
                history.append(row["action"])
            report["scene_seed"] = scene_seed
        else:
            reference = PhysicsOnly(shape="T", legacy=True)
            _, attempts = matched_reset({"T": reference, "D": live}, args.seed, report["counts"])
            report["reset_records"], report["scene_seed"] = attempts, attempts[-1]["seed"]
            initial = live.capture_state()
        report["initial_state"] = asdict(initial)
        report["search_start_state"] = asdict(live.capture_state())
        report["initial_metrics"] = physical_metrics(live)
        planner = ContactSequenceSearch(lambda: PhysicsOnly(shape="D", legacy=True), initial, config)
        for action in history:
            planner.record_action(action)
        report["initial_pose_rms"] = pose_rms(planner.branch.shape, live.capture_state())
        if args.display:
            from tdyssey.viewer import PushTViewer
            viewer = PushTViewer(title="Tdyssey | Simulator search | No policy or training", hold_seconds=args.hold_seconds)

        def draw():
            frame = live.render(mode="rgb_array")
            if viewer:
                viewer.draw(frame, f"D simulator search | no T policy | {live.control_step}/{args.steps} | Esc closes", 10)
            if video is not None:
                video.append(frame)
            return frame

        first = draw()
        if args.video:
            video = WallTimeVideo(output / "D-search-wall-time.mp4", first)
        save()
        while live.control_step < args.steps:
            if time.monotonic() >= start + args.seconds or (viewer and not viewer.poll()):
                report["stop_reason"] = "viewer_closed" if viewer and viewer.closed else "wall_time_budget"
                break
            before = live.capture_state()
            planned = time.monotonic()
            counts_before = dict(planner.counts)
            try:
                decision = planner.choose(before, args.steps - live.control_step, deadline=start + args.seconds)
            except SearchBudgetExhausted as exc:
                report["planning_attempts"].append({"control": live.control_step, "completed": False,
                    "budget_exhausted": True, "reason": str(exc), "seconds": time.monotonic()-planned,
                    "counts": {name: value-counts_before[name] for name, value in planner.counts.items()}})
                report["stop_reason"] = "search_budget"
                break
            except TimeoutError:
                report["planning_attempts"].append({"control": live.control_step, "completed": False,
                    "timeout": True, "seconds": time.monotonic() - planned,
                    "counts": {name: value - counts_before[name] for name, value in planner.counts.items()}})
                report["stop_reason"] = "wall_time_budget"
                break
            except BaseException as exc:
                report["planning_attempts"].append({"control": live.control_step, "completed": False,
                    "error": f"{type(exc).__name__}: {exc}", "seconds": time.monotonic()-planned,
                    "counts": {name: value-counts_before[name] for name, value in planner.counts.items()}})
                raise
            if state_error(before, live.capture_state()) != 0:
                raise ValueError("Search changed the live simulator")
            decision.report["live_control_step"] = live.control_step
            decision.report["wall_planning_seconds"] = time.monotonic() - planned
            report["planning_attempts"].append({"control": live.control_step, "completed": True,
                "timeout": bool(decision.report["timed_out"]), "seconds": decision.report["wall_planning_seconds"],
                "counts": {name: value - counts_before[name] for name, value in planner.counts.items()}})
            report["decisions"].append(decision.report)
            if not decision.actions:
                raise ValueError("Search returned no executable controls")
            print(json.dumps({"control": live.control_step, "coverage": physical_metrics(live)["coverage"],
                              "planning_seconds": decision.report["wall_planning_seconds"],
                              "selected": {k: v for k, v in decision.report["selected"].items()
                                           if k not in ("steps", "first_state")}}), flush=True)
            for offset, action in enumerate(decision.actions[:decision.report["execute_steps"]]):
                if live.control_step >= args.steps or time.monotonic() >= start + args.seconds:
                    report["stop_reason"] = "wall_time_budget" if live.control_step < args.steps else "control_budget"
                    break
                if viewer and not viewer.poll():
                    report["stop_reason"] = "viewer_closed"
                    break
                _, reward, done, info = live.step(action)
                report["counts"]["control_steps"] += 1
                report["counts"]["control_physics_substeps"] += live.sim_hz // live.control_hz
                row = {"action": list(action), "reward": float(reward), "upstream_done": bool(done),
                       "n_contacts": int(info["n_contacts"])}
                report["steps"].append(row)
                planner.record_action(action)
                actual = live.capture_state()
                row.update(state=asdict(actual), metrics=physical_metrics(live))
                row["pose_rms"] = pose_rms(planner.branch.shape, actual)
                expected = Snapshot.from_dict(decision.report["selected_predictions"][offset])
                error = state_error(actual, expected)
                row["prediction_state_error"] = error
                if error > 1e-7:
                    raise ValueError(f"Selected prediction differs from actual next state: {error}")
                row["joint_execution_check"] = planner.check_execution(actual)
                draw()
                if done:
                    report["success"], report["stop_reason"] = True, "upstream_success"
                    break
            save()
            if report["stop_reason"] != "control_budget":
                break
        report["status"] = "complete" if report["stop_reason"] in ("control_budget", "upstream_success") else "stopped_early"
        report["evaluation_seconds"] = time.monotonic() - start
        if viewer:
            viewer.finish()
    except BaseException as exc:
        report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cleanup_errors = []
        try:
            if live is not None and getattr(live, "space", None) is not None:
                report["final_metrics"] = physical_metrics(live)
        except Exception as exc:
            cleanup_errors.append(f"final metrics: {type(exc).__name__}: {exc}")
        if planner is not None:
            report["search_counts"] = dict(planner.counts)
            report["global_searches"] = planner.global_searches
            if live is not None and getattr(live, "space", None) is not None:
                report["final_pose_rms"] = pose_rms(planner.branch.shape, live.capture_state())
        for name, resource in (("video", video), ("planner", planner), ("live", live),
                               ("reference", reference), ("viewer", viewer)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    cleanup_errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if video is not None:
            report["video_frames"], report["video_fps"] = video.frames, video.fps
            report["video_timing"] = "actual_wall_time_with_previous_frame_held_during_search"
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
            report["status"] = "failed"
        report["torch_loaded"] = "torch" in sys.modules
        report["resume_sources_unchanged"] = all(hash_file(Path(p)) == value for p, value in report.get("resume_sources", {}).items())
        if not report["resume_sources_unchanged"]:
            report["status"], report["error"] = "failed", "Resume source changed during run"
        report["elapsed_seconds"] = time.monotonic() - start
        report["max_coverage"] = max([report.get("initial_metrics", {}).get("coverage", 0)] +
                                     [r["metrics"]["coverage"] for r in report["steps"] if "metrics" in r])
        report["max_iou"] = max([report.get("initial_metrics", {}).get("iou", 0)] +
                                [r["metrics"]["iou"] for r in report["steps"] if "metrics" in r])
        save()
    if report["status"] == "failed":
        raise RuntimeError(f"Search run failed during cleanup: {report.get('cleanup_errors', [])}")
    print(json.dumps({k: report[k] for k in ("status", "stop_reason", "success", "max_coverage", "final_metrics", "search_counts")}, indent=2))


if __name__ == "__main__":
    main()
