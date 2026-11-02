"""Capture predeclared D scenes and runtime inventory; never run a search."""

from dataclasses import asdict
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from tdyssey.envs.official import make_official_shape_env_class, matched_reset, official_shape
from tdyssey.evaluation import goal_metrics
from tdyssey.official import load_official_source


def main():
    requested = (100000, 100100, 100200, 100300)
    search_seeds = (42, 43)
    counters = {"reset_physics_substeps": 0, "actual_controls": 0,
                "search_evaluations": 0, "forecast_controls": 0}
    versions = {name: metadata.version(name) for name in (
        "numpy", "scipy", "pymunk", "pygame", "shapely", "gym", "opencv-python-headless")}
    base = make_official_shape_env_class(load_official_source(ROOT/"third_party/diffusion_policy"))

    class PhysicsOnly(base):
        def _get_obs(self):
            return {}

    report = {
        "protocol": "predeclared_d_teacher_scene_inventory_v1",
        "requested_scene_seeds": list(requested), "search_seeds": list(search_seeds),
        "selection": "Fixed before capture; existing common T/D geometry and clearance rejection only, never search success",
        "bounds_per_run": {"actual_episode_controls": 300, "wall_seconds": 600},
        "teacher_config": "pending core review and freeze; no optimizer dispatched by inventory",
        "runtime": {"python": platform.python_version(), "executable": sys.executable,
                    "platform": platform.platform(), "machine": platform.machine(),
                    "processor": platform.processor(), "packages": versions},
        "project_commit": subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True).strip(),
        "scenes": [], "matrix": [], "counts": counters,
    }
    shape = official_shape("D")
    for seed in requested:
        environments = {name: PhysicsOnly(shape=name, legacy=True) for name in ("T", "D")}
        before = counters["reset_physics_substeps"]
        try:
            _, attempts = matched_reset(environments, seed, counters)
            state = environments["D"].capture_state()
            metrics = goal_metrics(shape, state.object_body.pose, state.goal)
            vertices = [v for polygon in shape.polygons(state.object_body.pose) for v in polygon]
            signed_angle = math.atan2(math.sin(state.goal.angle-state.object_body.pose.angle),
                                      math.cos(state.goal.angle-state.object_body.pose.angle))
            scene = {"requested_seed": seed, "accepted_seed": attempts[-1]["seed"],
                     "attempts": attempts, "reset_physics_substeps": counters["reset_physics_substeps"]-before,
                     "initial_state": asdict(state), "metrics": asdict(metrics),
                     "signed_goal_angle_delta_degrees": math.degrees(signed_angle),
                     "goal_translation": [g-p for g,p in zip(state.goal.position,state.object_body.pose.position)],
                     "pusher_offset_from_object_center": [p-c for p,c in zip(state.pusher_body.pose.position,
                                                                           state.object_body.pose.position)],
                     "object_aabb": [min(v[0] for v in vertices),min(v[1] for v in vertices),
                                     max(v[0] for v in vertices),max(v[1] for v in vertices)],
                     "minimum_object_margin_to_8_503_bounds": min(min(x-8,503-x) for v in vertices for x in v)}
            report["scenes"].append(scene)
            for search_seed in search_seeds:
                report["matrix"].append({"requested_scene_seed": seed, "accepted_scene_seed": scene["accepted_seed"],
                                         "search_seed": search_seed, "steps": 300,"seconds": 600})
            print(json.dumps({"requested":seed,"accepted":scene["accepted_seed"],
                              "object":asdict(state.object_body.pose),"pusher":state.pusher_body.pose.position,
                              "metrics":asdict(metrics),"signed_angle_degrees":scene["signed_goal_angle_delta_degrees"]}),flush=True)
        finally:
            for env in environments.values():env.close()
    report["accepted_scenes_distinct"] = len({s["accepted_seed"] for s in report["scenes"]})==len(requested)
    report["torch_loaded"] = "torch" in sys.modules
    target = Path(__file__).with_name("scene_inventory.json")
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"path":str(target),"counts":counters,"torch_loaded":report["torch_loaded"]}),flush=True)


if __name__ == "__main__":main()
