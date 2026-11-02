"""R2B selected0.1s feedback resampling; one frozen54-slot bounded collection.

No model, damping comparisons or extra physics. Geometry decisions are current-
state-only and replayable from saved logs without constructing an environment.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
os.environ.setdefault("SDL_AUDIODRIVER","dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")
EXPERIMENT=ROOT/"artifacts/native-rgb-supervised-20261029"
OUT=EXPERIMENT/"r2b"
SELECTED_SHA="58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6"


def relative(path):return Path(path).resolve().relative_to(OUT.resolve()).as_posix()

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode()

def json_safe(value):
    """Keep failure evidence strict JSON without concealing nonfinite values."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NONFINITE:" + repr(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def disk_files():
    old = read(EXPERIMENT / "r1/storage-and-resource-audit.json")
    paths = set()
    for raw in old["roots"]:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Previously audited artifact/source disappeared: {p}")
        if p.is_dir():
            paths.update(f.resolve() for f in p.rglob("*") if f.is_file())
        else:
            paths.add(p.resolve())
    for pattern in ("tdyssey/native_rgb*.py", "scripts/native_rgb*.py",
                    "tests/test_native_rgb*.py", "configs/native_rgb*.json"):
        paths.update(p.resolve() for p in ROOT.glob(pattern) if p.is_file())
    return paths

def storage_check(label, reserve_next_bytes=0):
    files = disk_files()
    total = sum(p.stat().st_size for p in files)
    probe = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    result = dict(at_utc=utc(), label=label, combined_bytes=total,
        atomic_reserve_bytes=256 * 1024**2, combined_limit_bytes=3 * 1024**3,
        r2b_artifact_bytes=probe, r2b_limit_bytes=128 * 1024**2,
        reserve_next_bytes=reserve_next_bytes, disk_free_bytes=shutil.disk_usage(OUT).free,
        files=len(files))
    result["passed"] = (total + result["atomic_reserve_bytes"] + reserve_next_bytes
        <= result["combined_limit_bytes"] and probe + reserve_next_bytes
        <= result["r2b_limit_bytes"] and result["disk_free_bytes"] > 512 * 1024**2)
    if not result["passed"]:
        raise RuntimeError("Storage reserve/cap reached: " + json.dumps(result))
    return result

def save_slot(row, meta, states, actions, images, attempted_actions, telemetry,
              partial_states=None, partial_rgb=None):
    import numpy as np
    directory = OUT / "trajectories" / row["slot_id"]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "trajectory.npz"
    with target.with_suffix(".npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, states=np.asarray(states,dtype=np.float64).reshape(-1,15),
            actions=np.asarray(actions,dtype=np.float64).reshape(-1,2),
            rgb=np.asarray(images,dtype=np.uint8).reshape(-1,3,96,96),
            attempted_actions=np.asarray(attempted_actions,dtype=np.float64).reshape(-1,2),
            partial_states=np.asarray(partial_states if partial_states is not None else [],dtype=np.float64).reshape(-1,15),
            partial_rgb=np.asarray(partial_rgb if partial_rgb is not None else [],dtype=np.uint8))
    os.replace(target.with_suffix(".npz.tmp"),target)
    logs = directory / "telemetry.json.gz"
    with gzip.open(logs.with_suffix(".gz.tmp"), "wt", encoding="utf-8") as stream:
        json.dump(json_safe(telemetry),stream,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    os.replace(logs.with_suffix(".gz.tmp"),logs)
    meta.update(status=row["status"], charged_controls=row["charged_controls"],
        completed_controls=row["completed_controls"], actual_controls=row["actual_controls"],
        partial_substeps=row["partial_substeps"],
        frame_sha256=[hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest() for im in images],
        frame_hash_layout="uint8 CHW C-order raw bytes", trajectory_sha256=sha(target),
        telemetry_file=relative(logs),telemetry_sha256=sha(logs), physical_span_seconds=len(actions)*.1,
        partial_control_span_seconds=(row["partial_substeps"]*.01 if meta.get("partial_time_exact",True) else None),
        partial_engine_call_allowance_seconds=row["partial_substeps"]*.01,
        final_endpoint_display_hold_seconds=.1, midtrajectory_resets=0)
    atomic_json(directory / "metadata.json",meta)
    row.update(trajectory_sha256=sha(target),metadata_sha256=sha(directory / "metadata.json"))

SOURCE_FILES=["scripts/native_rgb_r2b_run.py","scripts/native_rgb_r2b_plan.py",
    "scripts/native_rgb_r2b_media.py","scripts/native_rgb_r2b_report.py",
    "tdyssey/native_rgb_r2b_sampling.py","tdyssey/native_rgb_selected_physics.py",
    "configs/native_rgb_default.json","configs/native_rgb_selected_physics.json",
    "tdyssey/native_rgb_r2a_env.py","tdyssey/native_rgb_r2a2_env.py",
    "tdyssey/native_rgb_env.py","tdyssey/native_rgb_data.py",
    "tdyssey/native_rgb_supervised_sampling.py","tdyssey/native_rgb_motion_statistics.py",
    "tdyssey/envs/official.py","tdyssey/geometry.py","tdyssey/letters.py",
    "tdyssey/state.py","tdyssey/collection.py","tdyssey/official.py","tdyssey/source_export.py",
    "scripts/native_rgb_r2a_media.py","scripts/native_rgb_r2a_report.py",
    "scripts/native_rgb_r2a2_report.py","manifests/official_dp.json"]


def verify_sources():
    for row in read(OUT/"source-identity.json")["files"]:
        if sha(row["source"])!=row["sha256"] or sha(OUT/row["snapshot"])!=row["sha256"]:
            raise RuntimeError("Frozen R2B source/rules/config changed: "+row["source"])
    for row in read(OUT/"prior-resource-identities.json")["files"]:
        if sha(row["path"])!=row["sha256"]:raise RuntimeError("Old evidence/resource ledger changed")
    for row in read(OUT/"runtime-identity.json")["files"]:
        if sha(row["path"])!=row["sha256"]:raise RuntimeError("Existing physics runtime changed")
    for row in read(OUT/"parent-review-identities.json")["files"]:
        if sha(row["path"])!=row["sha256"]:raise RuntimeError("Saved parent-review artifact changed")


def prepare():
    from tdyssey.native_rgb_r2b_sampling import RULES,canonical_sha,LETTERS,LOCATIONS
    from tdyssey.native_rgb_selected_physics import load_selected_config,selected_metadata
    if (OUT/"resource-ledger.json").exists() or (OUT/"source-identity.json").exists():
        raise FileExistsError("Do not overwrite or prepare another R2B launch")
    state=read(EXPERIMENT/"stage-state.json")
    if state["current_stage"]!="R2B" or not state["r2b_execution_authorized"] or state["budget"]["r2b_new_controls_cap"]!=1296:
        raise RuntimeError("R2B1296 not authorized")
    selected=load_selected_config();selection=selected_metadata()
    if selected["protocol_sha256"]!=SELECTED_SHA:raise RuntimeError("User-selected protocol identity changed")
    proposal=read(OUT/"proposed-requests.json");requests=proposal["requests"]
    if len(requests)!=54 or {(q["letter"],q["initial_space"]) for q in requests}!={(l,p) for l in LETTERS for p in LOCATIONS}:
        raise RuntimeError("Exactly the fixed6letterx9space54-slot matrix is required")
    if len({q["request_id"] for q in requests})!=54 or len({q["seed"] for q in requests})!=54:
        raise RuntimeError("Duplicate slot/seed; no replacements or hidden slots")
    if tuple(q["letter"] for q in requests[:6])!=tuple("TCXBLO"):
        raise RuntimeError("First6 roles/letters do not match the parent instruction")
    expected_roles=("rotation_positive","translation_1","translation_1","translation_1","rotation_negative","wall_along")
    if tuple(q["role"] for q in requests[:6])!=expected_roles:
        raise RuntimeError("First6 must keep their frozen designated roles")
    if proposal["rules"]!=RULES or read(OUT/"sampling-rules.json")!=RULES or proposal["rules_sha256"]!=canonical_sha(RULES):
        raise RuntimeError("Frozen deterministic feedback rules do not match the planner implementation")
    source_plan=ROOT/proposal["source_plan"]["path"]
    if sha(source_plan)!=proposal["source_plan"]["sha256"]:raise RuntimeError("Old initialpose source plan changed")
    for q in requests:
        if q["phase"]!="R2B" or q["protocol_sha256"]!=SELECTED_SHA or q["max_controls"]!=24 or q["policy_rules_sha256"]!=canonical_sha(RULES):
            raise RuntimeError("Illegal request/protocol/control cap")
        source=q["source"]
        for name,key in (("npz_file","npz_sha256"),("metadata_file","metadata_sha256")):
            if sha(EXPERIMENT/source[name])!=source[key]:raise RuntimeError("Old initialpose source data changed")
    shutil.copyfile(OUT/"proposed-requests.json",OUT/"frozen-requests.json")
    shutil.copyfile(OUT/"sampling-rules.json",OUT/"frozen-rules.json")
    policy_identity=dict(kind=RULES["kind"],rules_sha256=canonical_sha(RULES),
        rules_file="frozen-rules.json",rules_file_sha256=sha(OUT/"frozen-rules.json"),
        source_file="tdyssey/native_rgb_r2b_sampling.py",source_sha256=sha(ROOT/"tdyssey/native_rgb_r2b_sampling.py"))
    policy_identity["policy_sha256"]=hashlib.sha256(canonical({k:policy_identity[k] for k in ("kind","rules_sha256","source_sha256")})).hexdigest()
    config=dict(kind="R2B-selected-contact-following-1296.v1",max_controls=1296,max_slots=54,
        max_controls_per_slot=24,first_six_included_cap=144,prior_supervised_controls=40704,
        prior_all_native_controls=75264,selected_physics=selection,policy_identity=policy_identity,
        rules_sha256=canonical_sha(RULES),requests_sha256=sha(OUT/"frozen-requests.json"),
        physics_source=str((ROOT.parent/"Tdyssey/third_party/diffusion_policy").resolve()),
        candidate_comparisons_allowed=False,models=0,GPU_work=0,training_allowed=False,
        physics_substeps_per_control=10,control_dt=.1,physics_dt=.01,penetration_limit_world_px=2.,
        planned_H1=1134,planned_H4=972,planned_push_future_H4=378,planned_push_full_history_H4=216,
        stage_artifact_cap=128*1024**2,shared_cap=3*1024**3,atomic_reserve=256*1024**2,
        old_unused320_spent=False,replays_allowed=False)
    atomic_json(OUT/"frozen-config.json",config)
    # Only selected-physics configuration identity is read; old motions are not rerun.
    baseline_index=read(EXPERIMENT/"r2a2/index.json");baselines={}
    for row in baseline_index["slots"]:
        if row["protocol"]=="D-half-life-0.1s" and row["letter"] not in baselines:
            p=EXPERIMENT/"r2a2"/row["metadata_file"];meta=read(p)
            baselines[row["letter"]]=dict(metadata_file=str(p.resolve()),metadata_sha256=sha(p),
                initial_identity=meta["initial_identity"])
    if set(baselines)!=set(LETTERS):raise RuntimeError("Selected configuration reference missing a TRAIN representative")
    atomic_json(OUT/"selected-config-references.json",baselines)
    pairs=[(ROOT/name,name) for name in SOURCE_FILES]
    pairs.extend((p,p.relative_to(ROOT).as_posix()) for p in sorted((ROOT/"tests").glob("test_native_rgb_r2b*.py")))
    pairs.extend((p,p.relative_to(ROOT).as_posix()) for p in sorted((ROOT/"tests").glob("test_native_rgb_selected_physics.py")))
    for name in ("diffusion_policy/env/pusht/pusht_env.py","diffusion_policy/env/pusht/pymunk_override.py"):
        pairs.append((Path(config["physics_source"])/name,"official-source/"+name))
    for name in ("frozen-requests.json","frozen-rules.json","frozen-config.json","PARENT_REVIEW.md",
                 "prior-resource-identities.json","runtime-identity.json","preflight-checks.json",
                 "parent-review-identities.json","selected-config-references.json"):
        pairs.append((OUT/name,"R2B-inputs/"+name))
    identities=[]
    for source,name in pairs:
        target=OUT/"source-snapshot"/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as stream:stream.write(source.read_bytes())
        identities.append(dict(source=str(source.resolve()),snapshot=target.relative_to(OUT).as_posix(),sha256=sha(source),bytes=source.stat().st_size))
    atomic_json(OUT/"source-identity.json",dict(kind="R2B-frozen-before-first-control.v1",at_utc=utc(),files=identities))
    slots=[]
    for q in requests:
        slots.append(dict(slot_id=q["request_id"],request_id=q["request_id"],family_id=q["family_id"],
            letter=q["letter"],initial_space=q["initial_space"],role=q["role"],max_controls=24,
            dispatchable=q["dispatchable"],static_missing_reasons=q["static_missing_reasons"],
            blocked_reason=q["static_missing_reasons"] if not q["dispatchable"] else None,
            status="not_executed" if q["dispatchable"] else "static_blocked",attempted=False,
            charged_controls=0,completed_controls=0,actual_controls=0,partial_substeps=0,
            trajectory_file=f"trajectories/{q['request_id']}/trajectory.npz",
            metadata_file=f"trajectories/{q['request_id']}/metadata.json"))
    atomic_json(OUT/"index.json",dict(kind="R2B-all54-planned-slot-registry.v1",slots=slots,
        collection_closed=False,status="prepared",gate_passed=False,planned_slots=54,
        planned_controls=1296,planned_H1=1134,planned_H4=972,
        planned_push_future_H4=378,planned_push_full_history_H4=216))
    atomic_json(OUT/"resource-ledger.json",dict(kind="R2B-linked-physical-resource-ledger.v1",max_controls=1296,
        prior_supervised_controls=40704,prior_all_native_controls=75264,
        charged_controls=0,completed_controls=0,executed_substeps=0,engine_step_calls=0,
        engine_call_errors_with_unknown_advance=0,pending_control=None,closed=False,started_at_utc=None,
        failures=[],model_calls=0,optimizer_attempts=0,GPU_work=0,old_unused320_spent=False,
        static_blocked_slots=sum(not q["dispatchable"] for q in requests)))
    atomic_json(OUT/"pre-execution-storage.json",storage_check("R2B_prepared",2*1024**2))
    print(json.dumps(dict(prepared=True,planned_slots=54,dispatchable=sum(q["dispatchable"] for q in requests),
        max_controls=1296,selected_protocol_sha256=SELECTED_SHA,policy_sha256=policy_identity["policy_sha256"],new_physics_controls=0)))


def guard_charge(ledger,row,control):
    if (ledger["closed"] or ledger["pending_control"] is not None or ledger["charged_controls"]>=1296
            or row["charged_controls"]>=24 or control!=row["charged_controls"]):
        raise RuntimeError("No over-budget, replayed or unresolved control")
    ledger["charged_controls"]+=1;row["charged_controls"]+=1
    ledger["pending_control"]=dict(slot_id=row["slot_id"],control_index=control)


def verify_selected_parameters(identity,letter):
    from tdyssey.native_rgb_selected_physics import verify_selected_identity
    verify_selected_identity(identity)
    ref=read(OUT/"selected-config-references.json")[letter]
    if sha(ref["metadata_file"])!=ref["metadata_sha256"]:raise RuntimeError("Saved selected reference changed")
    if identity["physics"]!=ref["initial_identity"]["physics"]:
        raise RuntimeError("Non-state selected physical parameters changed")


def normalize_decision(decision,action,before,control,policy_identity,previous_action=None):
    import numpy as np
    from tdyssey.native_rgb_r2b_sampling import planned_phase,contact_frame
    a=np.asarray(action,dtype=np.float64);s=np.asarray(before,dtype=np.float64)
    if a.shape!=(2,) or not np.isfinite(a).all() or (a<0).any() or (a>511).any():raise RuntimeError("Invalid absolute target")
    if decision["control_index"]!=control or decision["planned_phase"]!=planned_phase(control):raise RuntimeError("Frozen phase/time rule mismatch")
    if not np.array_equal(decision["current_state"],s) or decision["state_before_sha256"]!=hashlib.sha256(s.tobytes()).hexdigest():
        raise RuntimeError("Decision used an unlogged current state")
    for key in ("proposed_action","action_absolute","bounded_target"):
        if not np.array_equal(decision[key],a):raise RuntimeError("Logged/applied feedback target mismatch")
    if decision["policy_rules_sha256"]!=policy_identity["rules_sha256"] or decision["lookahead_physics_steps"]!=0 or decision["used_future_state"] is not False:
        raise RuntimeError("Frozen rule identity or no-lookahead constraint violated")
    if np.linalg.norm(a-s[:2])>64+1e-8 or previous_action is not None and np.linalg.norm(a-previous_action)>96+1e-8:
        raise RuntimeError("Actual target exceeds frozen64/96worldpx bounds")
    if decision["command_mode"]!="hold" and not decision["command_geometry_verified"]:
        raise RuntimeError("Unconfirmed moving command issued")
    contact=dict(point_local=decision["local_contact_point"],outward_normal_local=decision["local_contact_normal"])
    point,normal,_=contact_frame(s,contact)
    if not np.allclose(point,decision["contact_point_world"],rtol=0,atol=1e-8) or not np.allclose(normal,decision["outward_normal_world"],rtol=0,atol=1e-8):
        raise RuntimeError("World contact/normal did not follow current object pose")
    decision=dict(decision,policy_sha256=policy_identity["policy_sha256"])
    return decision


def control_record(info,substeps,action,decision,control,completed,cog):
    import numpy as np
    kinds={"agent_object":set(),"object_wall":set(),"agent_wall":set()};arcs=set();points=[]
    for i,sub in enumerate(substeps):
        for event in sub.get("contacts",[]):
            kind=event.get("kind")
            if kind in kinds:kinds[kind].add(i)
            if kind!="agent_object":continue
            arcs.update(str(a) for a in event.get("arc_ids",[]))
            if event.get("phase")!="post_solve":continue
            normal=np.asarray(event["normal"],dtype=float)
            if event["shapes_order"][0]!="object":normal=-normal
            post=sub.get("post_state");com=None
            if post is not None and len(post)==15 and np.isfinite(post).all():
                state=np.asarray(post,dtype=float);c,s=math.cos(state[4]),math.sin(state[4]);co=np.asarray(cog,dtype=float)
                com=state[2:4]+np.array((c*co[0]-s*co[1],s*co[0]+c*co[1]))
            for p in event.get("points",[]):
                if p.get("object_world") is None:continue
                r=None if com is None else np.asarray(p["object_world"])-com;inward=-normal
                points.append(dict(substep_index=i,phase="post_solve",object_world=p["object_world"],object_local=p.get("object_local"),
                    arc_ids=p.get("arc_ids",[]),outward_normal_world=normal.tolist(),signed_normal_force_lever_world_px=None if r is None else float(r[0]*inward[1]-r[1]*inward[0])))
    p=[s.get("penetrations") for s in substeps];unknown=any(v is None or len(v)!=3 or any(x is None for x in v) for v in p)
    maxima=[None,None,None] if unknown else np.max(np.asarray(p),axis=0).tolist() if p else [0.,0.,0.]
    return dict(control_index=control,completed=completed,valid_transition=bool(completed and info.get("valid",False)),
        valid=bool(completed and info.get("valid",False)),substep_count=len(substeps),max_penetrations=maxima,
        max_substep_penetrations=maxima,penetration=maxima,unknown_penetrations=unknown,
        contact=len(kinds["agent_object"]),agent_object_contact_substeps=len(kinds["agent_object"]),
        object_wall_contact_substeps=len(kinds["object_wall"]),agent_wall_contact_substeps=len(kinds["agent_wall"]),
        actual_contact_arc_ids=sorted(arcs),contact_arc_bins=sorted(arcs),actual_contact_point_samples=points,
        action_absolute=list(map(float,action)),action_phase=decision["actual_phase"],planned_phase=decision["planned_phase"],decision=decision)


def save_dynamic_slot(row,meta,states,actions,images,attempted,telemetry,decisions,partial_states=None,partial_rgb=None):
    directory=OUT/"trajectories"/row["slot_id"];directory.mkdir(parents=True,exist_ok=True)
    target=directory/"decisionlogs.json.gz"
    with gzip.open(target.with_suffix(".gz.tmp"),"wt",encoding="utf-8") as stream:
        json.dump(json_safe(decisions),stream,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    os.replace(target.with_suffix(".gz.tmp"),target)
    meta.update(decisionlogs_file=target.relative_to(OUT).as_posix(),decisionlogs_sha256=sha(target))
    save_slot(row,meta,states,actions,images,attempted,telemetry,partial_states,partial_rgb)


def check_saved(row):
    """Replay only the deterministic sampler on a saved-state view, never physics."""
    import numpy as np
    from tdyssey.native_rgb_r2b_sampling import FeedbackPolicy
    meta=read(OUT/row["metadata_file"]);config=read(OUT/"frozen-config.json")
    if meta["policy_identity"]!=config["policy_identity"]:raise RuntimeError("Runtime policy bundle differs from frozen config")
    with np.load(OUT/row["trajectory_file"],allow_pickle=False) as z:
        states=z["states"].copy();actions=z["actions"].copy();images=z["rgb"].copy();attempted=z["attempted_actions"].copy()
    n=len(actions)
    if states.shape!=(n+1,15) or images.shape!=(n+1,3,96,96) or actions.dtype!=np.float64:
        raise RuntimeError("Ncontrols/N+1 saved timeline mismatch")
    if len(meta["frame_sha256"])!=n+1 or any(hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()!=h for im,h in zip(images,meta["frame_sha256"])):
        raise RuntimeError("Raw saved frame SHA mismatch")
    with gzip.open(OUT/meta["telemetry_file"],"rt",encoding="utf-8") as f:logs=json.load(f)
    with gzip.open(OUT/meta["decisionlogs_file"],"rt",encoding="utf-8") as f:decisions=json.load(f)
    if len(logs)!=row["charged_controls"] or len(decisions)!=len(logs):raise RuntimeError("A charged control has no decision/telemetry evidence")
    class StateView:
        def state(self):return self.value.copy()
    view=StateView();policy=FeedbackPolicy(view,meta["request"],read(OUT/"frozen-rules.json"));previous=None
    for i,log in enumerate(logs):
        before=states[i];d=decisions[i]
        if d!=meta["control_records"][i]["decision"]:raise RuntimeError("Independent decisionlog differs from control record")
        view.value=before;target,replayed=policy.next_action(i)
        replayed=normalize_decision(replayed,target,before,i,config["policy_identity"],previous)
        if not np.allclose(target,attempted[i],rtol=0,atol=1e-8) or not np.array_equal(d["proposed_action"],attempted[i]):
            raise RuntimeError("Feedback rule cannot reproduce actually applied target")
        if replayed["actual_phase"]!=d["actual_phase"] or replayed["command_mode"]!=d["command_mode"]:
            raise RuntimeError("Feedback branch is not replayable")
        if log["completed"]:
            if len(log["substeps"])!=10 or not np.array_equal(log["substeps"][0]["pre_state"],before) or not np.array_equal(log["substeps"][-1]["post_state"],states[i+1]):
                raise RuntimeError("100Hz timing/state chain mismatch")
            policy.observe(before,actions[i],states[i+1],meta["control_records"][i]["feedback_input"])
        previous=np.asarray(d["proposed_action"],dtype=float)
    return dict(passed=True,slot_id=row["slot_id"],completed_controls=n,raw_frames=n+1,
        sampler_replayed_controls=len(decisions),physics_steps_in_replay=0)


def run(mode):
    import numpy as np
    from tdyssey.native_rgb_selected_physics import make_selected_env,selected_metadata
    from tdyssey.native_rgb_r2b_sampling import FeedbackPolicy
    verify_sources();config=read(OUT/"frozen-config.json");index=read(OUT/"index.json");ledger=read(OUT/"resource-ledger.json")
    state=read(EXPERIMENT/"stage-state.json")
    if state["current_stage"]!="R2B" or not state["r2b_execution_authorized"]:raise RuntimeError("R2B no longer authorized")
    if ledger["closed"] or ledger["pending_control"] is not None or index["collection_closed"]:raise RuntimeError("No closed/interrupted launch replay")
    if mode=="first-six" and (ledger["charged_controls"]!=0 or index["status"]!="prepared"):raise RuntimeError("First6 can execute only once")
    if mode=="remaining" and not (index["gate_passed"] and index["status"]=="first_six_passed" and read(OUT/"first-six-once-claim.json")["finished_successfully"]):
        raise RuntimeError("Frozen remaining48 require completed first6 implementation gate")
    claim_path=OUT/(mode+"-once-claim.json");claim=dict(mode=mode,at_utc=utc(),pid=os.getpid(),finished_successfully=False,no_replays=True)
    with claim_path.open("x",encoding="utf-8") as stream:json.dump(claim,stream,indent=2)
    requests={q["request_id"]:q for q in read(OUT/"frozen-requests.json")["requests"]}
    rules=read(OUT/"frozen-rules.json");ledger["started_at_utc"]=ledger["started_at_utc"] or utc()
    state["status"]="running_R2B_first_six" if mode=="first-six" else "running_R2B_remaining";atomic_json(EXPERIMENT/"stage-state.json",state)
    selected_rows=index["slots"][:6] if mode=="first-six" else index["slots"][6:]
    current=None;halt=None
    try:
        for row in selected_rows:
            current=row
            if not row["dispatchable"]:
                if row["status"]!="static_blocked":raise RuntimeError("Static gap cannot be replaced or relabelled")
                continue
            if row["status"]!="not_executed":raise RuntimeError("Slot already initialized/executed; no replay")
            verify_sources();storage=storage_check("before-"+row["slot_id"],2*1024**2)
            q=requests[row["request_id"]];row.update(attempted=True,status="initializing");atomic_json(OUT/"index.json",index)
            env=make_selected_env(config["physics_source"],q["letter"]);env.seed(q["seed"])
            rgb=env.reset_scene(**q["initialization"]["reset_scene_arguments"]);before=env.state().copy();identity=env.initial_identity()
            verify_selected_parameters(identity,q["letter"])
            if (not np.isfinite(before).all() or not np.allclose(before,q["initialization"]["expected_initial_state15"],rtol=0,atol=1e-9)
                    or np.any(before[5:]!=0) or env.agent.angular_velocity!=0 or env.agent.torque!=0 or np.max(env.penetrations())>.1):
                raise RuntimeError("Actual initial scene/zero state differs from static frozen request")
            meta=dict(request=q,family=dict(family_id=q["family_id"],initialization=q["initialization"]),
                initial_identity=identity,actual_initial_state=before.tolist(),protocol_sha256=SELECTED_SHA,
                selected_physics=selected_metadata(),physics_display_name="0.1秒半衰期（已选）",
                policy_identity=config["policy_identity"],control_records=[],initial_storage=storage,
                actual_initial_parameters_match=True,geometry_state_is_privileged_collection_and_audit_only=True)
            states=[before];images=[rgb];actions=[];attempted=[];telemetry=[];decisions=[];partial_states=[];partial_rgb=None
            policy=FeedbackPolicy(env,q,rules);row["status"]="running";previous=None
            save_dynamic_slot(row,meta,states,actions,images,attempted,telemetry,decisions)
            for control in range(24):
                before=env.state().copy();action,decision=policy.next_action(control)
                decision=normalize_decision(decision,action,before,control,config["policy_identity"],previous)
                guard_charge(ledger,row,control);atomic_json(OUT/"resource-ledger.json",ledger);atomic_json(OUT/"index.json",index)
                attempted.append(np.asarray(action,dtype=np.float64).tolist());decisions.append(decision)
                info={};full=False;feedback=None;observed=None
                try:
                    rgb,info=env.step(np.asarray(action,dtype=np.float64))
                    if not info["valid"] or info["physics_substeps"]!=10 or env.control_step!=control+1:raise RuntimeError("Actual physics quality/timing failed")
                    full=True
                    feedback=dict(contact=int(info["contact"]),valid=True,physics_substeps=10,control_step=control+1,
                        penetration=np.asarray(info["penetration"]).tolist(),agent_object_contact_substeps=info["agent_object_contact_substeps"],
                        actual_contact_arc_ids=info["actual_contact_arc_ids"])
                    observed=policy.observe(before,action,env.state().copy(),feedback)
                except BaseException as exc:
                    row.update(status="failed_after_full_control" if full else "failed_partial_control",error=repr(exc))
                    meta["failure"]=dict(control_index=control,reason=repr(exc),at_utc=utc(),physical_control_complete=full)
                    if not full:
                        steps=env.last_substeps;row["partial_substeps"]=len(steps)
                        partial_states=[s["post_state"] for s in steps if s.get("post_state") is not None]
                        meta["partial_state_substep_indices"]=[i for i,s in enumerate(steps) if s.get("post_state") is not None]
                        meta["unavailable_partial_state_substeps"]=[i for i,s in enumerate(steps) if s.get("post_state") is None]
                        meta["partial_time_exact"]=not any(s.get("engine_exception") for s in steps)
                        if np.isfinite(env.state()).all():
                            try:partial_rgb=env.render_clean()
                            except Exception as capture:meta["failure"]["partial_render_error"]=repr(capture)
                    raise
                finally:
                    steps=env.last_substeps;row["actual_controls"]+=int(bool(steps));ledger["engine_step_calls"]+=len(steps)
                    ledger["executed_substeps"]+=sum(bool(s.get("executed")) and not s.get("engine_exception") for s in steps)
                    ledger["engine_call_errors_with_unknown_advance"]+=sum(bool(s.get("engine_exception")) for s in steps)
                    record=control_record(info,steps,action,decision,control,full,identity["physics"]["object"]["center_of_gravity"])
                    if feedback is not None:record["feedback_input"]=feedback
                    if full:
                        record["feedback_observed"]=observed
                        states.append(env.state().copy());images.append(rgb);actions.append(np.asarray(action,dtype=np.float64).tolist())
                        row["completed_controls"]+=1;ledger["completed_controls"]+=1;ledger["pending_control"]=None
                    meta["control_records"].append(record)
                    telemetry.append(dict(control_index=control,action=list(map(float,action)),completed=full,substeps=steps))
                    save_dynamic_slot(row,meta,states,actions,images,attempted,telemetry,decisions,partial_states,partial_rgb)
                    atomic_json(OUT/"resource-ledger.json",ledger);atomic_json(OUT/"index.json",index)
                previous=np.asarray(action,dtype=float)
            row["status"]="completed";save_dynamic_slot(row,meta,states,actions,images,attempted,telemetry,decisions)
            review=check_saved(row);atomic_json(OUT/"checks"/(row["slot_id"]+".json"),review)
            row["post_storage"]=storage_check("after-"+row["slot_id"]);atomic_json(OUT/"index.json",index)
            print(json.dumps(dict(slot=row["slot_id"],status=row["status"],charged=ledger["charged_controls"],substeps=ledger["executed_substeps"])),flush=True)
        if mode=="first-six":
            reviews=[check_saved(row) for row in selected_rows if row["dispatchable"]]
            if ledger["charged_controls"]>144:raise RuntimeError("First6 exceeded its included cap")
            atomic_json(OUT/"first-six-integrity-check.json",dict(passed=True,slot_reviews=reviews,
                static_blocked=[row["slot_id"] for row in selected_rows if not row["dispatchable"]],charged_controls=ledger["charged_controls"],
                movement_threshold_used_as_gate=False,new_physics_in_saved_rule_replay=0))
            index.update(status="first_six_passed",gate_passed=True)
        else:index.update(status="completed_frozen54_with_static_gaps",collection_closed=True);ledger.update(closed=True,stopped_at_utc=utc())
    except BaseException as exc:
        halt=dict(at_utc=utc(),slot_id=None if current is None else current["slot_id"],reason=repr(exc),no_replay_or_replacement=True)
        ledger["failures"].append(halt);ledger.update(closed=True,stopped_at_utc=utc())
        if current is not None and current["status"] not in ("completed","failed_partial_control","failed_after_full_control"):
            current.update(status="initialization_or_policy_or_IO_failure",error=repr(exc))
        if current is not None:
            p=OUT/current["metadata_file"]
            if p.exists():
                try:
                    saved=read(p);saved.update(status=current["status"])
                    saved.setdefault("failure",halt);atomic_json(p,saved);current["metadata_sha256"]=sha(p)
                except Exception as capture:halt["metadata_preservation_error"]=repr(capture)
        index.update(status="stopped_on_first_error",collection_closed=True,halt=halt);atomic_json(OUT/"halt.json",halt)
    finally:
        ledger.update(supervised_plus_charged_controls=40704+ledger["charged_controls"],all_native_plus_charged_controls=75264+ledger["charged_controls"],
            supervised_plus_completed_controls=40704+ledger["completed_controls"],all_native_plus_completed_controls=75264+ledger["completed_controls"])
        atomic_json(OUT/"resource-ledger.json",ledger);atomic_json(OUT/"index.json",index)
        state=read(EXPERIMENT/"stage-state.json");state.update(status="R2B_cold_ready_for_report" if index["collection_closed"] else "R2B_first_six_cold_pending_review",
            r2b_collection_closed=index["collection_closed"],r2b_first_six_gate_passed=index["gate_passed"],
            next_action="Only cold offline report/media then actual callback andSTOP" if index["collection_closed"] else "Inspect first6 saved rule/timing/quality, then frozenremaining48only; not a Dvalidation or motion-success gate")
        state["resources"].update(r2b_charged_controls=ledger["charged_controls"],r2b_completed_controls=ledger["completed_controls"],r2b_executed_substeps=ledger["executed_substeps"])
        atomic_json(EXPERIMENT/"stage-state.json",state)
    verify_sources();claim.update(finished_at_utc=utc(),finished_successfully=halt is None,collection_status=index["status"],charged_controls=ledger["charged_controls"]);atomic_json(claim_path,claim)
    print(json.dumps(dict(status=index["status"],collection_closed=index["collection_closed"],charged=ledger["charged_controls"],completed=ledger["completed_controls"],
        substeps=ledger["executed_substeps"],torch_imported="torch" in sys.modules,models=0,GPU_work=0)),flush=True)
    return 2 if halt else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=["prepare","first-six","remaining"]);args=parser.parse_args()
    if args.mode=="prepare":prepare();return 0
    return run(args.mode)


if __name__=="__main__":raise SystemExit(main())
