"""Bounded privileged D macro optimization; CPU simulator only, no training."""

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
from scipy.optimize import minimize
from tdyssey.envs.official import make_official_shape_env_class, official_shape
from tdyssey.evaluation import goal_metrics
from tdyssey.official import load_official_source
from tdyssey.simulator_planning import state_error
from tdyssey.state import Snapshot


def ramp(start, finish, count):
    return [tuple(float(a+(b-a)*(i+1)/count) for a,b in zip(start,finish))
            for i in range(count)]


class StopProbe(Exception):
    pass


def main():
    began = time.monotonic()
    deadline = began+75
    source = json.loads((ROOT/"artifacts/td-baseline-300-20261021/report.json").read_text(encoding="utf-8"))
    episode = next(e for e in source["episodes"] if e["shape"] == "D")
    initial = Snapshot.from_dict(episode["initial_state"])
    expected = Snapshot.from_dict(episode["steps"][175]["state"])
    history = [row["action"] for row in episode["steps"][:176]]
    shape = official_shape("D")
    original = make_official_shape_env_class(load_official_source(ROOT/"third_party/diffusion_policy"))

    class PhysicsOnly(original):
        def _get_obs(self):
            return {}

        def step(self, action):
            return super().step(np.asarray(action,dtype=np.float64))

    env = PhysicsOnly(shape="D",legacy=True)
    env.seed(100000)
    env.reset()
    center = np.array(expected.object_body.pose.position)
    delta = np.array(expected.goal.position)-center
    unit = delta/np.linalg.norm(delta)
    tangent = np.array([-unit[1],unit[0]])
    points = np.array([v for polygon in shape.polygons(expected.object_body.pose) for v in polygon])
    rear_support = np.min((points-center)@unit)
    base_rear = center+unit*(rear_support-35)
    clear_x = float(min(points[:,0].min()-35,expected.pusher_body.pose.position[0]))
    clear_y = float(points[:,1].max()+35)
    rows, best = [], None
    counts = {"replay_controls":0,"forecast_controls":0,"restores":0}

    def evaluate(parameters):
        nonlocal best
        if time.monotonic() >= deadline or len(rows) >= 160:
            raise StopProbe("bounded_search_budget")
        env.restore_state(initial)
        counts["restores"] += 1
        for action in history:
            env.step(action)
            counts["replay_controls"] += 1
        error = state_error(env.capture_state(),expected)
        if error>1e-7:
            raise AssertionError(error)
        length,lateral,yaw = (float(x) for x in parameters)
        c,s = math.cos(yaw),math.sin(yaw)
        direction = np.array([c*unit[0]-s*unit[1],s*unit[0]+c*unit[1]])
        rear = base_rear+tangent*lateral
        finish = rear+length*direction
        stages = [((clear_x,expected.pusher_body.pose.position[1]),8),
                  ((clear_x,clear_y),20),(rear,12),(finish,24)]
        commands, peak, done, contacts = [],0.0,False,0
        for target,controls in stages:
            for action in ramp(env.capture_state().pusher_body.pose.position,target,controls):
                if any(not 20<=x<=492 for x in action):
                    raise AssertionError("outside_workspace")
                _,reward,done,info=env.step(action)
                counts["forecast_controls"] += 1
                commands.append(action)
                peak=max(peak,min(1.0,reward*.95))
                contacts+=bool(info["n_contacts"])
                if done:break
            if done:break
        if not done:
            brake=env.capture_state().pusher_body.pose.position
            for _ in range(8):
                _,reward,done,info=env.step(brake)
                counts["forecast_controls"] += 1
                commands.append(tuple(brake))
                peak=max(peak,min(1.0,reward*.95))
                contacts+=bool(info["n_contacts"])
                if done:break
        state=env.capture_state()
        metrics=goal_metrics(shape,state.object_body.pose,state.goal)
        score=1.0 if done else metrics.coverage-.0001*metrics.position_error-.001*metrics.angle_error
        row={"parameters":[length,lateral,yaw],"score":score,"metrics":asdict(metrics),
             "controls":len(commands),"episode_control":176+len(commands),"peak_coverage":peak,
             "native_success":bool(done),"contact_controls":contacts,"replay_error":error}
        rows.append(row)
        if best is None or score>best["score"]:
            best={**row,"actions":commands,"terminal_snapshot":asdict(state)}
            print(json.dumps({"best_index":len(rows)-1,**row}),flush=True)
        if done:
            raise StopProbe("native_success")
        return -score

    status="optimizer_complete"
    try:
        try:
            # The parent handmade route is only a proposal seed. Candidate
            # parameters are optimized against independent physical forecasts.
            x0=np.array([97.32320435220717,16.390825263585874,-0.19999442174898335])
            simplex=np.array([x0,x0+[.5,0,0],x0+[0,.8,0],x0+[0,0,-.015]])
            minimize(evaluate,x0,method="Nelder-Mead",bounds=[(65,130),(-15,45),(-.5,.3)],
                     options={"maxfev":160,"xatol":1e-3,"fatol":1e-5,"initial_simplex":simplex})
        except StopProbe as exc:
            status=str(exc)
        report={"kind":"privileged_macro_parameter_optimization_not_policy_or_training",
                "status":status,"start_control":176,"actual_task_limit":300,
                "optimizer":"scipy Nelder-Mead expanded-region refinement, max160eval/75seconds",
                "proposal":"handmade collision-geometry rear-support route plus optimized push length/lateral offset/yaw and physical braking",
                "objective":"native_success1 else terminal_coverage-.0001*position_error-.001*angle_error",
                "history_protocol":"initial_snapshot_plus_all176actual_controls_per_candidate",
                "best":best,"evaluations":rows,"counts":counts,
                "elapsed_seconds":time.monotonic()-began,"torch_loaded":"torch" in sys.modules}
        if best and best["native_success"]:
            import cv2
            video_path=Path(__file__).with_name(Path(__file__).stem+"-paused.mp4")
            writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*"mp4v"),5,(512,560))
            if not writer.isOpened():raise RuntimeError("Video output unavailable")
            live=original(shape="D",legacy=True)
            live.seed(100000)
            live.reset()
            live.restore_state(initial)
            actual_rows=[]
            video_frames=0
            def draw(phase,pause_frames=1):
                nonlocal video_frames
                image=cv2.resize(live.render(mode="rgb_array"),(512,512),interpolation=cv2.INTER_NEAREST)
                canvas=np.full((560,512,3),245,dtype=np.uint8)
                canvas[:512]=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                cv2.putText(canvas,"D simulator action search | NO TRAINING",(8,529),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1,cv2.LINE_AA)
                cv2.putText(canvas,f"{phase} | control {live.control_step} / 300",(8,550),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1,cv2.LINE_AA)
                for _ in range(pause_frames):writer.write(canvas);video_frames+=1
            try:
                draw("Initial physical scene",10)
                for index,action in enumerate(history):
                    _,reward,done,info=live.step(np.asarray(action,dtype=np.float64))
                    state=live.capture_state()
                    error=state_error(state,Snapshot.from_dict(episode["steps"][index]["state"]))
                    if error>1e-7 or done:raise AssertionError("Independent historical replay differs")
                    actual_rows.append({"phase":"baseline_history","action":action,"reward":float(reward),
                                        "native_done":bool(done),"n_contacts":int(info["n_contacts"]),
                                        "metrics":asdict(goal_metrics(shape,state.object_body.pose,state.goal)),
                                        "state":asdict(state),"baseline_replay_error":error})
                    draw("Recorded T-policy history")
                draw("Begin optimized simulator probe",10)
                for action in best["actions"]:
                    _,reward,done,info=live.step(np.asarray(action,dtype=np.float64))
                    state=live.capture_state()
                    actual_rows.append({"phase":"optimized_probe","action":action,"reward":float(reward),
                                        "native_done":bool(done),"n_contacts":int(info["n_contacts"]),
                                        "metrics":asdict(goal_metrics(shape,state.object_body.pose,state.goal)),"state":asdict(state)})
                    draw("Optimized probe, not learned policy")
                    if done:break
                terminal=live.capture_state()
                error=state_error(terminal,Snapshot.from_dict(best["terminal_snapshot"]))
                if error>1e-7 or not done or live.control_step>300:
                    raise AssertionError("Independent optimized execution failed verification")
                draw("NATIVE SUCCESS - no model updates",20)
                report["independent_execution"]={"native_success":bool(done),"task_controls":live.control_step,
                    "terminal_branch_error":error,"rows":actual_rows,"video":str(video_path),
                    "video_fps":5,"nominal_seconds_per_control":.2,"video_frames":video_frames,
                    "pause_frames":40,"historical_controls":len(history),"optimized_controls":len(actual_rows)-len(history),
                    "final_metrics":asdict(goal_metrics(shape,terminal.object_body.pose,terminal.goal))}
            finally:
                writer.release();live.close()
        Path(__file__).with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(json.dumps({"status":status,"evaluations":len(rows),"counts":counts,
                          "elapsed_seconds":report["elapsed_seconds"],"torch_loaded":report["torch_loaded"]}),flush=True)
    finally:
        env.close()


if __name__=="__main__":main()
