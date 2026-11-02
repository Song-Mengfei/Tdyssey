"""Four predeclared handmade CPU probes from D control176; no policy or search."""

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
from tdyssey.envs.official import make_official_shape_env_class, official_shape
from tdyssey.evaluation import goal_metrics
from tdyssey.official import load_official_source
from tdyssey.simulator_planning import state_error
from tdyssey.state import Snapshot


def metrics(state, shape):
    return asdict(goal_metrics(shape, state.object_body.pose, state.goal))


def ramp(start, finish, controls):
    return [tuple(a+(b-a)*(i+1)/controls for a, b in zip(start, finish)) for i in range(controls)]


def main():
    began = time.monotonic()
    source_report = ROOT/"artifacts/td-baseline-300-20261021/report.json"
    raw = json.loads(source_report.read_text(encoding="utf-8"))
    episode = next(e for e in raw["episodes"] if e["shape"] == "D")
    initial = Snapshot.from_dict(episode["initial_state"])
    expected = Snapshot.from_dict(episode["steps"][175]["state"])
    history = [row["action"] for row in episode["steps"][:176]]
    shape = official_shape("D")
    original = make_official_shape_env_class(load_official_source(ROOT/"third_party/diffusion_policy"))

    class PhysicsOnly(original):
        def _get_obs(self):
            return {}

        def step(self, action):
            return super().step(np.asarray(action, dtype=np.float64))

    def replay():
        env = PhysicsOnly(shape="D", legacy=True)
        env.seed(episode["scene_seed"])
        env.reset()
        env.restore_state(initial)
        for action in history:
            _, _, done, _ = env.step(action)
            if done:
                raise AssertionError("Historical replay unexpectedly reaches native success")
        error = state_error(env.capture_state(), expected)
        if error > 1e-7:
            raise AssertionError(f"Historical reconstruction error {error}")
        return env, error

    reference, error = replay()
    untouched = reference.capture_state()
    pusher = expected.pusher_body.pose.position
    center = expected.object_body.pose.position
    delta = tuple(g-x for g, x in zip(expected.goal.position, center))
    distance = math.hypot(*delta)
    direction = tuple(value/distance for value in delta)
    vertices = [v for polygon in shape.polygons(expected.object_body.pose) for v in polygon]
    min_x, max_y = min(v[0] for v in vertices), max(v[1] for v in vertices)
    back = min(sum((v[i]-center[i])*direction[i] for i in (0, 1)) for v in vertices)
    behind = tuple(center[i]+direction[i]*(back-35) for i in (0, 1))
    waypoints = [(min(min_x-35, pusher[0]), pusher[1]),
                 (min(min_x-35, pusher[0]), max_y+35), behind]
    report = {"kind": "handmade_privileged_counterfactual_not_policy_improvement",
              "source_report": str(source_report.relative_to(ROOT)), "start_control": 176,
              "remaining_task_controls": 124, "history_protocol": "initial_snapshot_plus_all_176_actual_actions",
              "initial_replay_error": error, "start_snapshot": asdict(expected),
              "start_metrics": metrics(expected, shape), "desired_object_translation": delta,
              "declared_probes": ["direct_delta_1x_24", "direct_delta_2x_24", "direct_NE_80px_s_24", "left_bottom_reposition_then_push"],
              "reposition_waypoints": waypoints, "object_back_support_projection": back,
              "geometry_prior": "known object collision geometry, goal and pusher state; no optimization or search",
              "probes": [], "recorded_policy_followup": {}}
    for controls in (24, 80):
        rows = episode["steps"][176:176+controls]
        state = Snapshot.from_dict(rows[-1]["state"])
        report["recorded_policy_followup"][str(controls)] = {
            "controls": len(rows), "max_coverage": max(r["metrics"]["coverage"] for r in rows),
            "final_metrics": metrics(state, shape), "contact_steps": sum(r["n_contacts"] > 0 for r in rows)}

    def run_probe(name, stages):
        env, replay_error = replay()
        rows = []
        try:
            for finish, controls, stage in stages:
                start = env.capture_state().pusher_body.pose.position
                for action in ramp(start, finish, controls):
                    if len(rows) >= 80:
                        raise AssertionError("Predeclared probe budget exceeded")
                    if any(not 0 <= value <= 512 for value in action):
                        raise AssertionError("Probe target is outside the workspace")
                    _, reward, done, info = env.step(action)
                    state = env.capture_state()
                    row = {"stage": stage, "action": action, "reward": float(reward),
                           "done": bool(done), "n_contacts": int(info["n_contacts"]),
                           "metrics": metrics(state, shape), "state": asdict(state)}
                    rows.append(row)
                    if done:
                        break
                if rows[-1]["done"]:
                    break
            final = rows[-1]
            summary = {"name": name, "controls": len(rows), "replay_state_error": replay_error,
                       "contact_steps": sum(r["n_contacts"] > 0 for r in rows),
                       "contact_count_sum": sum(r["n_contacts"] for r in rows),
                       "max_coverage": max(r["metrics"]["coverage"] for r in rows),
                       "max_iou": max(r["metrics"]["iou"] for r in rows), "final_metrics": final["metrics"],
                       "native_success": final["done"], "final_pusher": final["state"]["pusher_body"],
                       "steps": rows}
            if state_error(reference.capture_state(), untouched) != 0:
                raise AssertionError("Counterfactual changed the separate live reference")
            report["probes"].append(summary)
            print(json.dumps({k: v for k, v in summary.items() if k not in ("steps", "final_pusher")}), flush=True)
        finally:
            env.close()

    try:
        for scale in (1, 2):
            finish = tuple(pusher[i]+scale*delta[i] for i in (0, 1))
            run_probe(f"direct_delta_{scale}x_24", [(finish, 24, "direct_NE")])
        finish = (pusher[0]+80*2.4/math.sqrt(2), pusher[1]-80*2.4/math.sqrt(2))
        run_probe("direct_NE_80px_s_24", [(finish, 24, "direct_NE")])
        finish = tuple(behind[i]+direction[i]*(distance+35) for i in (0, 1))
        run_probe("left_bottom_reposition_then_push",
                  [(waypoints[0], 8, "clear_left"), (waypoints[1], 20, "clear_bottom"),
                   (waypoints[2], 12, "behind_SW"), (finish, 24, "push_NE")])
        report["live_reference_unchanged"] = reference.capture_state() == untouched
        torch_module = sys.modules.get("torch")
        report["cuda_initialized"] = bool(torch_module is not None and torch_module.cuda.is_initialized())
        report["counts"] = {"historical_replay_controls": 176*(1+len(report["probes"])),
                            "counterfactual_controls": sum(p["controls"] for p in report["probes"])}
        report["elapsed_seconds"] = time.monotonic()-began
        target = Path(__file__).with_suffix(".json")
        target.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({"evidence": str(target), "elapsed_seconds": report["elapsed_seconds"],
                          "cuda_initialized": report["cuda_initialized"]}), flush=True)
    finally:
        reference.close()


if __name__ == "__main__":
    main()
