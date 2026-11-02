"""One <=30-second real D contact/replay check of the isolated draft, no policy."""
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                   "PYGAME_HIDE_SUPPORT_PROMPT": "1", "MPLBACKEND": "Agg"}.items():
    os.environ.setdefault(key, value)

spec = importlib.util.spec_from_file_location("tdyssey_isolated_v2_check", HERE / "simulator_search_v2.py")
draft = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = draft
spec.loader.exec_module(draft)

from tdyssey.envs.official import make_official_shape_env_class, official_shape
from tdyssey.geometry import Pose
from tdyssey.official import load_official_source
from tdyssey.simulator_planning import state_error
from tdyssey.state import BodyState, Snapshot


def main():
    begin = time.monotonic()
    report = {"kind": "isolated_v2_tiny_real_D_protocol_check", "status": "running",
              "training": False, "checkpoint_loaded": False, "wall_budget_seconds": 30,
              "historical_controls": 0, "actual_selected_controls": 0, "historical_contacts": 0}
    live, search = None, None
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
        rect = min(official_shape("D").components, key=lambda r: (r.x0, r.y0))
        start = (256+rect.x0-20, 256+rect.center[1])
        initial = replace(live.capture_state(), object_body=BodyState(Pose((256, 256), 0)),
                          pusher_body=BodyState(Pose(start, 0)), goal=Pose((400, 120), .5),
                          control_step=0, pusher_target=None)
        live.restore_state(initial)
        config = draft.SearchConfig(horizon=32, population=6, generations=2, elite_count=2,
                                    max_evaluations=13, execute_steps=4, settle_steps=4, seed=7)
        search = draft.SimulatorActionSearchV2(environment, initial, config)
        for _ in range(8):
            action = (start[0]+65, start[1])
            _, _, done, info = live.step(action)
            assert not done
            report["historical_controls"] += 1
            report["historical_contacts"] += info["n_contacts"]
            search.record_action(action)
        assert report["historical_contacts"] > 0
        before = live.capture_state()
        decision = search.choose(before, remaining_steps=40, deadline=begin+30)
        assert state_error(live.capture_state(), before) == 0
        errors = []
        for offset, action in enumerate(decision.actions[:4]):
            _, _, done, _ = live.step(action)
            report["actual_selected_controls"] += 1
            error = state_error(live.capture_state(), Snapshot.from_dict(decision.report["selected_predictions"][offset]))
            errors.append(error)
            assert error <= 1e-7
            search.record_action(action)
            if done:
                break
        assert not decision.report["timed_out"]
        assert time.monotonic()-begin <= 30
        report.update(status="passed", config=asdict(config), initial_state=asdict(initial),
                      max_selected_state_error=max(errors), decision=decision.report,
                      final_state=asdict(live.capture_state()))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if search is not None:
            report["counts"] = dict(search.counts)
            search.close()
        if live is not None:
            live.close()
        report["elapsed_seconds"] = time.monotonic()-begin
        report["torch_loaded"] = "torch" in sys.modules
        destination = HERE / (sys.argv[1] if len(sys.argv) > 1 else "tiny-runtime.json")
        destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("decision", "initial_state", "final_state")}, indent=2))


if __name__ == "__main__":
    main()
