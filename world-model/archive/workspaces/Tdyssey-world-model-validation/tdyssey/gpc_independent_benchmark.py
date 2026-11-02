"""Fixed independent official T/L scenes, with injected simulator and budget.

Categories are intentions, not guaranteed outcomes. All fixed branches remain;
weak contact/rotation/success coverage never triggers replacement or search.
Imports perform no model, policy, native, simulator or filesystem operation.
"""
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
from pathlib import Path

from .collection import reset_issues
from .dataset import json_bytes
from .envs.official import official_shape
from .evaluation import goal_metrics
from .frozen_wm_protocol import CANDIDATES, candidate_actions, goal_for
from .geometry import Pose
from .state import BodyState

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT/"artifacts/wm-data-preparation-20261026/independent-benchmark-plan.json"
REVIEWED_PATH = ROOT/"artifacts/wm-l-adaptation-preparation-20261025/split-proposal.json"
REVIEWED_FILE_SHA256 = "8281784d8cecbf6d0c42306a22bdfe1d68498deefec7491732a5037b23255eba"
REVIEWED_LIST_SHA256 = "cd878f5c383f2420df857813e2a2d88c511cc03a3042618465eae36e9854ae08"
DEVELOPMENT_SEEDS = (100000,100001,100002)
CATEGORIES = ("no_contact","contact_translation","clockwise_rotation",
              "counterclockwise_rotation","near_success_negative","near_success_positive")
LIMITS = dict(controls=256,resets=256,restores=64,renders=512,
    control_substeps=2560,reset_substeps=256,physics_substeps=2816,seconds=600,output_bytes=64*1024**2)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def scene_hash(state):
    """Same complete-Snapshot encoding as training_data.initial_state_hash."""
    return _sha(json_bytes(asdict(state)))


def fixed_anchors():
    """Pure geometry, fixed before resets; no random/data-dependent fallback."""
    result=[]
    ordinary={
        "T":[((270,204),7,(-155,15),(1,0)),((262,215),-7,(0,137),(0,-1)),
             ((266,222),-9,(42,-17),(0,1)),((258,218),11,(-42,-17),(0,1))],
        "L":[((238,242),-7,(-130,0),(1,0)),((244,239),9,(-47,0),(1,0)),
             ((239,255),5,(-47,-45),(1,0)),((247,249),-9,(-47,30),(1,0))]}
    for shape in ("T","L"):
        for index,category in enumerate(CATEGORIES):
            if index<4:
                position,degrees,pusher,direction=ordinary[shape][index]
                pose=Pose(position,math.radians(degrees))
            else:
                goal=goal_for(shape)
                pose=Pose(goal.transform((0,3) if shape=="T" else (-3,0)),goal.angle)
                pusher=((0,230) if index==4 else (0,142)) if shape=="T" else ((-130,0) if index==4 else (-52,0))
                direction=(0,-1) if shape=="T" else (1,0)
            rotate=Pose((0,0),pose.angle)
            result.append(dict(id=f"I{shape}{index:02d}",shape=shape,category=category,
                seed=930000+len(result),pose=pose,pusher=pose.transform(pusher),
                direction=rotate.transform(direction),goal=goal_for(shape)))
    return result


def fixed_plan():
    anchors=[]
    for value in fixed_anchors():
        row={key:asdict(item) if isinstance(item,Pose) else item for key,item in value.items()}
        row["history_actions"]=[list(value["pusher"])]*3
        row["candidate_actions"]={name:candidate_actions(value,name) for name in CANDIDATES}
        anchors.append(row)
    return dict(kind="tdyssey.gpc_independent_benchmark.fixed_plan.v1",base_commit="bf5283e",
        environment_family="official-pusht-capitals-v1",anchors=anchors,candidates=list(CANDIDATES),
        categories=list(CATEGORIES),limits=LIMITS,
        ideal_counts=dict(environment_creations=2,max_live_environments=1,resets=12,restores=60,
            controls=228,renders=300,control_substeps=2280,reset_substeps=12,physics_substeps=2292),
        history="initial real RGB plus three actual hold controls; identical full history/latest per anchor",
        future="four fixed controls per candidate unless strict physical success absorbs the remainder",
        success="coverage>.95; reward=min(coverage/.95,1); no controls/new renders after success",
        action_rule="frozen candidate_actions: pusher+12*h*(hold,toward,perpendicular,away),h=1..4; no clipping",
        constructor="pinned official __init__ seeds only; no implicit reset/render/physics",
        geometry_checks="source reset_issues except already_near_goal; intentional near probes start<=.95",
        clockwise_convention="positive wrapped angle in screen coordinates (y down)",
        category_evidence=dict(no_contact="all four branches have zero future contact controls",
            contact_translation="any contact branch with >5px final displacement and <=5deg wrapped rotation",
            clockwise_rotation="any contact branch with >5deg positive wrapped rotation",
            counterclockwise_rotation="any contact branch with <-5deg wrapped rotation",
            near_success_negative="any executed future has .92<=coverage<=.95",
            near_success_positive="any executed future has .95<coverage<=.98"),
        coverage_policy="unproven intentions; retain branches/report unmet evidence; never replace/reposition/search",
        exclusions=dict(reviewed_count=316,reviewed_file_sha256=REVIEWED_FILE_SHA256,
            reviewed_list_sha256=REVIEWED_LIST_SHA256,required_development_seeds=list(DEVELOPMENT_SEEDS),
            caller_must_include="all adaptation-source scenes/seeds including all A roles/held metadata and DPPO/development",
            state_hash="SHA256(dataset.json_bytes(asdict(Snapshot))); also reject Snapshot.to_json SHA aliases",
            conflict="stop before further controls; no next seed or alternative scene"),
        payload="float32 history/actual[4,3,96,96],absolute past_actions[3,2]/future_actions[4,2]; full states/executed/absorbing JSON",
        model_or_policy_access=False)


def validate_exclusions(excluded_scene_sha256,excluded_reset_seeds):
    """Metadata only; the owner supplies every adaptation-role/held exclusion."""
    raw=REVIEWED_PATH.read_bytes()
    if _sha(raw)!=REVIEWED_FILE_SHA256:raise ValueError("Reviewed exclusion source changed")
    reviewed=sorted(json.loads(raw)["reviewed_scene_sha256"])
    if len(reviewed)!=316 or _sha(json.dumps(reviewed,separators=(",",":")).encode())!=REVIEWED_LIST_SHA256:
        raise ValueError("The 316 reviewed exclusions differ")
    scenes=set(excluded_scene_sha256);seeds=set(excluded_reset_seeds)
    if any(not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v) for v in scenes):
        raise ValueError("Expected lowercase scene SHA256 values")
    if not set(reviewed).issubset(scenes):raise ValueError("Missing reviewed scene exclusions")
    if any(type(v) is not int or not 0<=v<2**32 for v in seeds) or not set(DEVELOPMENT_SEEDS).issubset(seeds):
        raise ValueError("Missing or invalid development/reset exclusions")
    if any(row["seed"] in seeds for row in fixed_anchors()):raise ValueError("Fixed seed excluded; no replacement")
    return scenes,dict(scene_count=len(scenes),reset_seed_count=len(seeds),
        scenes_sha256=_sha(json_bytes(sorted(scenes))),seeds_sha256=_sha(json_bytes(sorted(seeds))))


def _label(state):
    metric=goal_metrics(official_shape(state.shape),state.object_body.pose,state.goal)
    return dict(pose=asdict(state.object_body.pose),pusher=list(state.pusher_body.pose.position),coverage=metric.coverage,iou=metric.iou)


def _image(observation):
    import numpy as np
    image=observation.get("image")
    if (not isinstance(image,np.ndarray) or image.shape!=(3,96,96) or image.dtype!=np.float32
            or not np.isfinite(image).all() or image.min()<0 or image.max()>1):
        raise ValueError("Expected original finite float32 official CHW RGB [0,1]")
    return np.ascontiguousarray(image).copy()


def _exclude(state,excluded,seen=()):
    digest=scene_hash(state)
    if digest in excluded or _sha(state.to_json().encode()) in excluded or digest in seen:
        raise ValueError("Excluded/repeated fixed full-state scene; no replacement")
    return digest


def _invoke(budget,costs,call):
    budget.check();operation=budget.reserve(**costs)
    try:
        result=call();budget.complete(operation,dict(status="completed"));return result
    except BaseException as error:
        budget.fail(type(error).__name__+": "+str(error));raise


def _step(env,action,budget):
    import numpy as np
    observation,reward,done,info=_invoke(budget,dict(controls=1,renders=1,control_substeps=10),
        lambda:env.step(np.asarray(action,dtype=np.float64)))
    state=env.capture_state();truth=_label(state)
    if abs(float(reward)-min(truth["coverage"]/.95,1))>1e-7 or bool(done)!=(truth["coverage"]>.95):
        raise ValueError("Physical coverage/reward/strict success disagree")
    contacts=info.get("n_contacts")
    if type(contacts) is not int or contacts<0:raise ValueError("Missing physical contact count")
    return _image(observation),state,dict(action=list(action),state=asdict(state),truth=truth,
        contacts=contacts,reward=float(reward),success=bool(done),absorbing=False,executed=True)


def _write(budget,name,payload,run_id):
    budget.check();budget.write_bytes(name,payload,run_id=run_id)
    return dict(path=name,bytes=len(payload),sha256=_sha(payload))


def _category_evidence(spec,cases):
    rows=[step for case in cases for step in case["future"] if step["executed"]]
    turns=[]
    for case in cases:
        a,b=case["latest"]["pose"],case["truth"][-1]["pose"]
        turn=math.degrees(math.atan2(math.sin(b["angle"]-a["angle"]),math.cos(b["angle"]-a["angle"])))
        turns.append((case["contact_by_h4"],math.dist(a["position"],b["position"]),turn))
    passed={"no_contact":all(not case["contact_by_h4"] for case in cases),
        "contact_translation":any(c and d>5 and abs(a)<=5 for c,d,a in turns),
        "clockwise_rotation":any(c and a>5 for c,_,a in turns),
        "counterclockwise_rotation":any(c and a< -5 for c,_,a in turns),
        "near_success_negative":any(.92<=step["truth"]["coverage"]<=.95 for step in rows),
        "near_success_positive":any(.95<step["truth"]["coverage"]<=.98 for step in rows)}[spec["category"]]
    return dict(anchor=spec["id"],category=spec["category"],passed=passed,physical_branches=len(cases),
        executed_future_controls=len(rows),contact_branches=sum(case["contact_by_h4"] for case in cases),
        success_branches=sum(any(step["success"] for step in case["future"]) for case in cases),
        final_motion=[dict(candidate=case["candidate"],translation=d,rotation_degrees=a) for case,(_,d,a) in zip(cases,turns)])


def run_benchmark(output,budget,*,env_factory,excluded_scene_sha256,excluded_reset_seeds,
                  run_id="B",plan_path=PLAN_PATH,progress=lambda *args:None):
    """Collect only the frozen 12x4 grid through one live injected environment.

    The root owns begin_run('B')/finish_run()/finish() and native execution
    authority. Its write_bytes(name,payload,run_id='B') must publish beneath
    output. No retry/resume, alternate scene, extra render or model path exists.
    """
    import numpy as np
    output=Path(output)
    if output.absolute()!=Path(budget.output).absolute():
        raise ValueError("Benchmark output must be the active budget output")
    frozen=Path(plan_path).read_bytes()
    if frozen!=json_bytes(fixed_plan()):
        raise ValueError("Construction/action rules were not frozen exactly")
    excluded,exclusion_summary=validate_exclusions(excluded_scene_sha256,excluded_reset_seeds)
    report=dict(kind="tdyssey.gpc_adaptation_benchmark.v1",status="collecting",preparation_only=False,
        fixed_plan_sha256=_sha(frozen),exclusions=exclusion_summary,cases=[],anchors=[],coverage_evidence=[],
        regression=dict(path="artifacts/frozen-wm-validation-v1-20261024",
            report_sha256="c7b1e3257468874940ad074d3ffc00c3d3d88f9f6de16531aba6d2c3cc6e3795",
            branches=63,independent=False),new_model_loads=0,new_model_forwards=0,policy_sampling=0)
    env=None;shape_name=None;seen=set();first_error=None
    try:
        _write(budget,"fixed-plan.json",frozen,run_id)
        for spec in fixed_anchors():
            budget.check()
            if shape_name!=spec["shape"]:
                if env is not None:env.close();env=None
                env=env_factory(spec["shape"]);shape_name=spec["shape"]
                if (env.project_shape!=shape_name or env.sim_hz!=100 or env.control_hz!=10
                        or getattr(env,"space",None) is not None):
                    raise ValueError("Need an unreset original official environment; constructor must not simulate")
            env.seed(spec["seed"])
            _invoke(budget,dict(resets=1,renders=1,reset_substeps=1),env.reset)
            # Upstream initializes this in _setup(), reached by the charged reset.
            if getattr(env,"success_threshold",None)!=.95:
                raise ValueError("Original official success threshold must be .95 after charged reset")
            upstream=env.capture_state();_exclude(upstream,excluded)
            state=replace(upstream,object_body=BodyState(spec["pose"]),pusher_body=BodyState(Pose(spec["pusher"],0)),
                          goal=spec["goal"],control_step=0,pusher_target=None)
            initial_hash=_exclude(state,excluded,seen)
            issues=[issue for issue in reset_issues(official_shape(shape_name),state) if issue!="already_near_goal"]
            if issues or _label(state)["coverage"]>.95:raise ValueError("Fixed initial geometry/success invalid: "+str(issues))
            for candidate in CANDIDATES:
                if any(not 0<=v<=511 for action in candidate_actions(spec,candidate) for v in action):
                    raise ValueError("Fixed action leaves official action domain")
            observation=_invoke(budget,dict(restores=1,renders=1),lambda:env.restore_state(state))
            if env.capture_state()!=state:raise ValueError("Constructed full-state restore differs")
            frames=[_image(observation)];history_states=[state];history=[]
            initial_record=dict(anchor=spec["id"],shape=shape_name,seed=spec["seed"],category=spec["category"],
                upstream_state=asdict(upstream),initial_state=asdict(state),initial_state_sha256=initial_hash)
            _write(budget,f"anchors/{spec['id']}-initial.json",json_bytes(initial_record),run_id)
            for ordinal in range(3):
                image,snapshot,step=_step(env,spec["pusher"],budget);_exclude(snapshot,excluded)
                frames.append(image);history_states.append(snapshot);history.append(dict(step,horizon=ordinal+1))
                if step["success"] and ordinal<2:raise ValueError("Success before three real history controls; no padding")
            latest=history_states[-1];_exclude(latest,excluded)
            history_digest=_sha(np.stack(frames).tobytes()+latest.to_json().encode()+
                                np.asarray([spec["pusher"]]*3,dtype=np.float64).tobytes())
            anchor=dict(initial_record,history=history,history_states=[asdict(s) for s in history_states],
                latest_state=asdict(latest),latest=_label(latest),history_sha256=history_digest,
                history_rgb_sha256=[_sha(frame.tobytes()) for frame in frames])
            _write(budget,f"anchors/{spec['id']}-history.json",json_bytes(anchor),run_id)
            report["anchors"].append(anchor);seen.add(initial_hash);branch_cases=[]
            for candidate in CANDIDATES:
                observation=_invoke(budget,dict(restores=1,renders=1),lambda:env.restore_state(latest))
                if env.capture_state()!=latest or not np.array_equal(_image(observation),frames[-1]):
                    raise ValueError("Candidates must restore identical full latest state and RGB")
                actions=candidate_actions(spec,candidate);future=[];actual=[];terminal=history[-1]["success"]
                image=frames[-1];snapshot=latest
                for horizon,action in enumerate(actions,1):
                    if not terminal:
                        image,snapshot,step=_step(env,action,budget);terminal=step["success"]
                    else:
                        step=dict(action=None,state=asdict(snapshot),truth=_label(snapshot),contacts=0,reward=1.,
                                  success=True,absorbing=True,executed=False)
                    future.append(dict(step,horizon=horizon));actual.append(image.copy())
                arrays=io.BytesIO()
                np.savez_compressed(arrays,history=np.stack(frames),actual=np.stack(actual),
                    past_actions=np.asarray([spec["pusher"]]*3,dtype=np.float64),future_actions=np.asarray(actions,dtype=np.float64))
                name=f"branches/{spec['id']}-{candidate}.npz"
                payload=_write(budget,name,arrays.getvalue(),run_id)
                case=dict(anchor=spec["id"],candidate=candidate,shape=shape_name,role="independent",prior_uses=[],
                    category=spec["category"],seed=spec["seed"],initial_state_sha256=initial_hash,
                    initial_state=asdict(state),history_sha256=history_digest,history_states=anchor["history_states"],
                    latest_state=asdict(latest),latest=_label(latest),goal=asdict(spec["goal"]),future=future,
                    truth=[row["truth"] for row in future],contact_by_h4=any(row["contacts"]>0 for row in future),payload=payload,
                    executed_future_controls=sum(row["executed"] for row in future),
                    absorbing_future_frames=sum(row["absorbing"] for row in future))
                _write(budget,name.replace(".npz",".json"),json_bytes(case),run_id)
                report["cases"].append(case);branch_cases.append(case)
                progress(spec["id"],candidate,len(report["cases"]))
            report["coverage_evidence"].append(_category_evidence(spec,branch_cases))
        report["status"]="complete";report["coverage_ready"]=all(row["passed"] for row in report["coverage_evidence"])
        report["executed_controls"]=36+sum(row["executed_future_controls"] for row in report["cases"])
        _write(budget,"benchmark.json",json_bytes(report),run_id)
        return report
    except BaseException as error:
        first_error=error;budget.fail("benchmark_failed: "+type(error).__name__+": "+str(error));raise
    finally:
        if env is not None:
            try:env.close()
            except BaseException as error:
                if first_error is None:raise
                first_error.add_note("Environment close also failed: "+str(error))
