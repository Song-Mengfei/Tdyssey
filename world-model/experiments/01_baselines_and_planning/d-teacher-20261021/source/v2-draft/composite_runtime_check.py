"""One predetermined staged-contact composite fixture, not a performance search."""
from dataclasses import asdict, replace
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                   "PYGAME_HIDE_SUPPORT_PROMPT": "1", "MPLBACKEND": "Agg"}.items():
    os.environ.setdefault(key, value)
spec = importlib.util.spec_from_file_location("isolated_composite_fixture", HERE / "simulator_search_v2.py")
draft = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = draft
spec.loader.exec_module(draft)
from tdyssey.envs.official import make_official_shape_env_class, official_shape
from tdyssey.geometry import Pose
from tdyssey.official import load_official_source, physical_metrics
from tdyssey.simulator_planning import state_error
from tdyssey.state import BodyState, Snapshot


def main():
    begin = time.monotonic()
    report = {"kind": "forced_composite_real_D_protocol_fixture", "status": "running",
              "performance_evidence": False, "selection_forced_for_protocol": True,
              "training": False, "checkpoint_loaded": False, "wall_budget_seconds": 30,
              "history": [], "execution": []}
    live, independent, search = None, None, None
    try:
        import numpy as np
        original = make_official_shape_env_class(load_official_source(ROOT / "third_party/diffusion_policy"))

        class PhysicsOnly(original):
            def _get_obs(self):
                return {}

            def step(self, action):
                return super().step(np.asarray(action, dtype=np.float64))

        def environment():
            env = PhysicsOnly(shape="D", legacy=True)
            env.seed(100000)
            env.reset()
            return env

        live = environment()
        shape, pose = official_shape("D"), Pose((256, 256), 0.)
        # Select one exterior midpoint with positive torque, not a scene answer.
        choices = []
        for index, edge in enumerate(draft.rectangle_exterior_edges(shape.components)):
            q, n = pose.transform(edge.point(.5)), edge.normal
            torque = draft.torque_proxy(q, pose.position, (-n[0], -n[1]))
            stage = tuple(q[i]+17*n[i] for i in (0, 1))
            if torque > 0 and draft.direct_clear(stage, stage, shape, pose, 16.9):
                choices.append((torque, index, q, n, stage))
        _, edge_id, contact, normal, start = max(choices)
        initial = replace(live.capture_state(), object_body=BodyState(pose),
                          pusher_body=BodyState(Pose(start, 0)),
                          goal=Pose((276, 256), .3), control_step=0, pusher_target=None)
        live.restore_state(initial)
        history = []
        for _ in range(4):
            action = tuple(start[i]-4*normal[i] for i in (0, 1))
            _, reward, done, info = live.step(action)
            assert not done
            history.append(action)
            report["history"].append({"action": action, "state": asdict(live.capture_state()),
                                      "contacts": info["n_contacts"], "reward": float(reward)})
        config = draft.SearchConfig(horizon=80, population=2, generations=1, elite_count=1,
                                    max_evaluations=2, execute_steps=8, settle_steps=4, seed=7)
        search = draft.SimulatorActionSearchV2(environment, initial, config)
        for action in history:
            search.record_action(action)
        proposal = {"family": "rotation", "edge_id": edge_id, "route": 1,
                    "parameters": (edge_id, .5, 2., 10., 0., 100., 1), "recover_translation": True}
        original_proposal = search._proposal

        def fixed(state, family, ordinal, elites=None, explore=False):
            return proposal if family == "long" else original_proposal(state, family, ordinal, elites, explore)

        calls, original_forecast = [], search.branch._forecast

        def observed(*args, **kwargs):
            result = original_forecast(*args, **kwargs)
            calls.append(result)
            return result

        def select(records, best, incumbent, margin):
            composites = [r for r in records if r.get("status") == "completed" and r.get("recover_translation")]
            if not composites:
                raise RuntimeError("Predetermined composite did not fit; no search or retuning performed")
            return composites[0], "forced_composite_protocol_fixture"

        before = live.capture_state()
        with patch.object(search, "_proposal", side_effect=fixed), \
                patch.object(search.branch, "_forecast", side_effect=observed), \
                patch.object(draft, "select_candidate", side_effect=select):
            decision = search.choose(before, remaining_steps=100, deadline=begin+30)
        assert state_error(live.capture_state(), before) == 0
        assert len(calls) == 3  # common hold, rotation, recovery+settling
        expected_commands = calls[1][0]+calls[2][0]
        expected_rows = calls[1][1]+calls[2][1]
        assert expected_commands == decision.actions
        independent = environment()
        independent.restore_state(initial)
        for row in report["history"]:
            _, reward, done, _ = independent.step(row["action"])
            assert not done and abs(reward-row["reward"]) <= 1e-9
            assert state_error(independent.capture_state(), Snapshot.from_dict(row["state"])) <= 1e-7
        for i, action in enumerate(decision.actions):
            if time.monotonic() >= begin+30:
                raise TimeoutError("Fixture exceeded 30-second budget")
            _, reward, done, info = independent.step(action)
            expected = expected_rows[i]
            error = state_error(independent.capture_state(), Snapshot.from_dict(decision.report["selected_predictions"][i]))
            metrics = physical_metrics(independent)
            row = {"index": i, "state_error": error, "reward_error": abs(reward-expected["reward"]),
                   "done_matches": bool(done) == expected["native_done"],
                   "contact_matches": int(info["n_contacts"]) == expected["n_contacts"],
                   "coverage_error": abs(metrics["coverage"]-expected["coverage"])}
            report["execution"].append(row)
            assert error <= 1e-7 and row["reward_error"] <= 1e-9 and row["done_matches"]
            assert row["contact_matches"] and row["coverage_error"] <= 1e-9
            if done:
                assert i+1 == len(decision.actions)
                break
        report.update(status="passed", initial_state=asdict(initial), proposal=proposal,
                      decision=decision.report, independent_history_controls=len(history),
                      independent_forecast_controls=len(report["execution"]),
                      max_state_error=max(r["state_error"] for r in report["execution"]),
                      rotation_controls=len(calls[1][0]), recovery_and_hold_controls=len(calls[2][0]),
                      live_state_unchanged=state_error(live.capture_state(), before) == 0)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if search is not None:
            report["counts"] = dict(search.counts)
            search.close()
        for env in (live, independent):
            if env is not None:
                env.close()
        report["elapsed_seconds"] = time.monotonic()-begin
        report["torch_loaded"] = "torch" in sys.modules
        (HERE / "composite-runtime.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("history", "execution", "decision", "initial_state")}, indent=2))


if __name__ == "__main__":
    main()
