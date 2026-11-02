"""Only the parent-authorized, finite stage-one native RGB physical probe.

No model construction, optimizer, server execution or later-stage chaining.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdyssey.native_rgb_training import atomic_json, disk_bytes, sha256


class StageLedger:
    """Persist one charged control before entering the unchanged physics step."""
    def __init__(self, root, cap=4000):
        self.path = Path(root) / "resource-ledger.json"
        if self.path.exists():
            self.value = json.loads(self.path.read_text(encoding="utf-8"))
            if self.value["max_attempted_controls"] != cap:
                raise ValueError("Do not reset or enlarge the existing stage-one budget")
        else:
            self.value = dict(kind="native-rgb-supervised-resource-ledger.v1",
                max_attempted_controls=cap, charged_controls=0, completed_controls=0,
                physics_substeps=0, recorded_partial_substeps_lower_bound=0,
                optimizer_attempts=0, model_inference_calls=0,
                unseen_model_inference_calls=0, server_jobs=0, trajectories=[],
                reset_rejections=[], started_at_utc=datetime.now(timezone.utc).isoformat())
            self.save()

    def save(self):
        atomic_json(self.path, self.value)

    def begin_trajectory(self, trajectory_id, maximum_controls=64):
        if len(self.value["trajectories"]) >= 54:
            raise RuntimeError("Stage one permits at most54 trajectory attempts")
        if any(r["status"] == "running" for r in self.value["trajectories"]):
            raise RuntimeError("Reconcile an interrupted trajectory before starting another")
        row = dict(trajectory_id=trajectory_id, maximum_controls=maximum_controls,
                   charged_controls=0, completed_controls=0, status="running", pending_control=None)
        self.value["trajectories"].append(row)
        self.save()
        return len(self.value["trajectories"]) - 1

    def charge(self, index):
        row = self.value["trajectories"][index]
        if row["status"] != "running" or row["pending_control"] is not None:
            raise RuntimeError("Do not execute or repeat an unresolved physics attempt")
        if (self.value["charged_controls"] >= self.value["max_attempted_controls"]
                or row["charged_controls"] >= row["maximum_controls"]):
            raise RuntimeError("Stage-one control budget exhausted")
        row["charged_controls"] += 1
        row["pending_control"] = row["charged_controls"] - 1
        self.value["charged_controls"] += 1
        self.save()

    def complete(self, index, physics_substeps=10):
        row = self.value["trajectories"][index]
        if row["status"] != "running" or row["pending_control"] is None:
            raise RuntimeError("No pending control to complete")
        row["completed_controls"] += 1
        row["pending_control"] = None
        self.value["completed_controls"] += 1
        self.value["physics_substeps"] += int(physics_substeps)
        self.save()

    def finish(self, index, status="completed", **details):
        row = self.value["trajectories"][index]
        if row["status"] != "running":
            raise RuntimeError("A trajectory cannot finish twice")
        row.update(status=status, **details)
        self.save()

    def reject_reset(self, trajectory_id, reason):
        self.value["reset_rejections"].append(dict(trajectory_id=trajectory_id,
            reason=str(reason), physics_controls=0, hidden_physics_substeps=0))
        self.save()


def read_supervision(output):
    output = Path(output)
    memo = (output / "SUPERVISION.md").read_text(encoding="utf-8")
    state = json.loads((output / "stage-state.json").read_text(encoding="utf-8"))
    if (not memo or state["current_stage"] != 1 or state["authorized_stages"] != [1]
            or not state["authorization"]["verified"] or state["callback"]["sent"]
            or state["status"] in ("waiting_parent", "ready_to_report")):
        raise RuntimeError("Only the explicitly authorized current stage may run; read supervision")
    return state


def storage_check(config, output):
    old = (ROOT / config["old_artifact_root"]).resolve()
    output = Path(output).resolve()
    if old == output or not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Old evidence and the new experiment must remain separate")
    source_paths=set()
    for pattern in ("tdyssey/native_rgb*.py","scripts/native_rgb*.py","tests/test_native_rgb*.py",
                    "configs/native_rgb*.json","run_native_rgb*","requirements/native-rgb*",
                    "docs/NATIVE_RGB_WORLD_MODEL.md"):
        source_paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    native_sources=sum(p.stat().st_size for p in source_paths)
    current = disk_bytes(old) + disk_bytes(output) + native_sources
    if current >= config["max_combined_artifact_bytes"]:
        raise RuntimeError("Combined old/new native artifacts reached3GiB")
    free = shutil.disk_usage(output).free
    if free <= config["disk_reserve_bytes"]:
        raise RuntimeError("Preserve the declared free-disk reserve")
    return dict(old_artifact_bytes=disk_bytes(old), new_artifact_bytes=disk_bytes(output),native_source_bytes=native_sources,
                combined_bytes=current, max_combined_bytes=config["max_combined_artifact_bytes"],
                disk_free_bytes=free, disk_reserve_bytes=config["disk_reserve_bytes"])


def freeze_source(config_path, output, physics_root):
    output = Path(output)
    identity_path = output / "source-identity.json"
    if identity_path.exists():
        raise FileExistsError("A launch snapshot already exists; never overwrite an inspected launch")
    paths = ["scripts/native_rgb_supervised_stage1.py", "tdyssey/native_rgb_supervised_sampling.py",
        "tdyssey/native_rgb_motion_statistics.py", "tdyssey/native_rgb_env.py", "tdyssey/native_rgb_data.py",
        "tdyssey/native_rgb_training.py", "tdyssey/geometry.py", "tdyssey/letters.py", "tdyssey/envs/official.py",
        "tdyssey/official.py", "tdyssey/source_export.py", "manifests/official_dp.json"]
    pairs = [(ROOT / name, name) for name in paths]
    pairs.append((Path(config_path), "frozen-config.json"))
    launch_config=json.loads(Path(config_path).read_text(encoding="utf-8"))
    pairs.append((ROOT/launch_config["sampling"]["prior_coverage"],"historical-contact-coverage.json"))
    for name in ("diffusion_policy/env/pusht/pusht_env.py", "diffusion_policy/env/pusht/pymunk_override.py"):
        pairs.append((Path(physics_root) / name, "official-source/" + name))
    rows = []
    for source, name in pairs:
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        rows.append(dict(source=str(source.resolve()), snapshot=target.relative_to(output).as_posix(),
                         bytes=target.stat().st_size, sha256=sha256(target)))
    identity = dict(kind="native-rgb-supervised-source-identity.v1", capture="before any stage1 physics controls",
        files=rows, timestamp_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(identity_path, identity)
    return identity


def verify_source(output):
    output = Path(output)
    identity = json.loads((output / "source-identity.json").read_text(encoding="utf-8"))
    for row in identity["files"]:
        if sha256(output / row["snapshot"]) != row["sha256"] or sha256(row["source"]) != row["sha256"]:
            raise RuntimeError("Collection source/config changed after launch: " + row["source"])
    return True


def contacted_bin_ids(letter, segments, points):
    """Deduplicate substep collision points to one actual contact-control per bin."""
    from shapely.geometry import Point
    from tdyssey.native_rgb_env import shape_geometry
    geometry=shape_geometry(letter);rings=[geometry.exterior,*geometry.interiors];hit=set()
    for xy in points:
        point=Point(*xy);ring_index=min(range(len(rings)),key=lambda i:rings[i].distance(point))
        ring=rings[ring_index]
        if ring.distance(point)>2:continue
        arc=ring.project(point)
        for row in segments:
            if row["ring"]==ring_index and row["arc_start"]<=arc<=row["arc_end"]+1e-6:
                hit.add(row["id"]);break
    return hit


def collect(config_path, output, physics_root):
    """The owned lock covers setup failures as well as the physical loop."""
    output=Path(output).resolve()
    read_supervision(output)
    lock=output/"COLLECTION_RUNNING.lock"
    with lock.open("x",encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    try:
        return _collect_impl(config_path,output,physics_root)
    finally:
        lock.unlink(missing_ok=True)


def _collect_impl(config_path, output, physics_root):
    import numpy as np
    from tdyssey.native_rgb_env import make_env, PROTOCOL_SHA256, CONTROLLER, STATE_COLUMNS
    from tdyssey.native_rgb_data import contour_segments, _contact_coverage, _write_coverage
    from tdyssey.native_rgb_supervised_sampling import plan_scenes, initialize_scene, ContinuousContactSampler

    output = Path(output).resolve()
    state = read_supervision(output)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if (config["stage"] != 1 or config["max_attempted_controls"] > 4000
            or config["max_trajectories"] > 54 or config["controls_per_trajectory"] > 64
            or config["environment"]["protocol_sha256"] != PROTOCOL_SHA256
            or config["environment"]["controller"] != CONTROLLER
            or config["stage1_optimizer_updates_allowed"] != 0):
        raise ValueError("Stage-one fixed scope/protocol differs")
    if tuple(config["letters"]) != tuple("TLOCXB"):
        raise ValueError("Only the six authorized training letters belong in this probe")
    data = output / "probe"
    if (data / "manifest.json").exists():
        raise FileExistsError("An inspected stage-one probe already exists; never silently recollect")
    started = time.monotonic()
    entries, reset_failures = [], []
    ledger = StageLedger(output, config["max_attempted_controls"])
    if ledger.value["charged_controls"]:
        raise RuntimeError("Existing spent probe requires parent review, not automatic replay")
    storage_before = storage_check(config, output)
    old_ledger = ROOT / config["old_artifact_root"] / "native-rgb-cost-ledger.json"
    ledger.value["historical_native_training"] = json.loads(old_ledger.read_text(encoding="utf-8"))
    ledger.value["historical_native_controls"] = 34560
    ledger.value["storage_before"] = storage_before
    ledger.save()
    coverage = {letter: contour_segments(letter, 8) for letter in config["letters"]}
    for rows in coverage.values():
        for row in rows:
            row.update(contact_control_count=0,moving_contact_control_count=0,trajectory_contact_controls={})
    counts = {letter: {row["id"]: 0 for row in rows} for letter, rows in coverage.items()}
    prior_path = ROOT / config["sampling"]["prior_coverage"]
    prior = json.loads(prior_path.read_text(encoding="utf-8"))["segments"]
    historical = {letter: {row["id"]: len(row["trajectory_ids"]) for row in prior[letter]} for letter in config["letters"]}
    reachability = {}
    requests = plan_scenes(config)
    if len(requests) != 54:
        raise ValueError("Probe must declare the fixed six-by-nine planned scene slots")
    data.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "scene-requests.json", requests)
    freeze_source(config_path, output, physics_root)
    failure = None
    try:
        for sequence, request in enumerate(requests):
            verify_source(output)
            storage_check(config, output)
            letter = request.get("letter", request.get("shape"))
            location = request.get("spatial_class", request.get("location"))
            trajectory_id = f"probe-{sequence:02d}-{letter}-{location}"
            env = make_env(physics_root, letter, controller=CONTROLLER, render_size=96)
            scene = initialize_scene(env, request, coverage_counts=counts[letter], historical_counts=historical[letter])
            if env.control_step != 0:
                raise RuntimeError("Scene initialization consumed hidden physics controls")
            if not scene.get("legal", False) or np.max(env.penetrations()) > .1:
                reason = dict(request=request, scene=scene, initial_penetrations=env.penetrations().tolist())
                ledger.reject_reset(trajectory_id, reason)
                reset_failures.append(reason)
                atomic_json(output / "reset-failures.json", reset_failures)
                continue
            reachability[letter] = scene.get("reachability", reachability.get(letter, []))
            policy = ContinuousContactSampler(env, coverage[letter], counts[letter], scene)
            attempt = ledger.begin_trajectory(trajectory_id, config["controls_per_trajectory"])
            states=[env.state().copy()];actions=[];contacts=[];valid=[]
            penetrations=[env.penetrations().copy()];substeps=[];points=[];offsets=[0];angles=[];step_records=[]
            trajectory_error = None
            exception_state = None
            try:
                for step in range(config["controls_per_trajectory"]):
                    if env.control_step != step:
                        raise RuntimeError("Sampler reset or stepped physics outside the charged runner")
                    if not np.array_equal(env.state(), states[-1]):
                        raise RuntimeError("Physical state changed without a charged physics step")
                    action, metadata = policy.next_action(step)
                    if not np.array_equal(env.state(), states[-1]):
                        raise RuntimeError("Action planning mutated physical state")
                    metadata=dict(metadata)
                    metadata.setdefault("target_bin",metadata.get("active_arc","unknown"))
                    metadata.setdefault("target_distance_class",metadata.get("requested_distance_category","unknown"))
                    action=np.asarray(action,dtype=np.float64)
                    if action.shape != (2,) or not np.isfinite(action).all() or (action<0).any() or (action>511).any():
                        raise ValueError("Sampler proposed an invalid absolute XY action")
                    before=env.state().copy()
                    centroid_before=np.asarray(env.world_geometry().centroid.coords[0])
                    ledger.charge(attempt)
                    _, info = env.step(action)
                    after=env.state().copy()
                    substep=np.asarray(env.substep_penetrations,dtype=np.float64)
                    if substep.shape != (10,3) or info["physics_substeps"] != 10 or env.control_step != step+1:
                        raise RuntimeError("Control must execute exactly ten audited100Hz substeps")
                    states.append(after);actions.append(action);contacts.append(info["contact"])
                    valid.append(info["valid"]);penetrations.append(info["penetration"]);substeps.append(substep)
                    points.extend(info["contact_points"].tolist());offsets.append(len(points))
                    angles.extend(info["contact_angles"].tolist())
                    movement=float(np.linalg.norm(np.asarray(env.world_geometry().centroid.coords[0])-centroid_before))
                    rotation=abs(float(after[4]-before[4]))
                    step_records.append(dict(metadata,step=step,actual_translation_world_px=movement,
                        actual_rotation_rad=rotation,valid=bool(info["valid"]),physics_substeps=10))
                    ledger.complete(attempt,10)
                    transition = policy.observe(before,action,after,info) or {}
                    step_records[-1]["observed"] = transition
                    if env.control_step != step+1:
                        raise RuntimeError("Sampler observe altered the physical timeline")
                    if not np.array_equal(env.state(), after):
                        raise RuntimeError("Sampler observe mutated physical state")
                    moving=movement>=2 or rotation>=np.pi/180
                    _contact_coverage(letter,coverage[letter],info["contact_points"],trajectory_id,
                        moving=moving,direction=str(metadata.get("approach_kind",metadata.get("intent","unknown"))),
                        angles=info["contact_angles"])
                    for bin_id in contacted_bin_ids(letter,coverage[letter],info["contact_points"]):
                        bin_row=coverage[letter][bin_id]
                        bin_row["contact_control_count"]+=1
                        bin_row["moving_contact_control_count"]+=int(moving)
                        per_trajectory=bin_row["trajectory_contact_controls"]
                        per_trajectory[trajectory_id]=per_trajectory.get(trajectory_id,0)+1
                    counts[letter].update({row["id"]:row["contact_control_count"] for row in coverage[letter]})
                    if not info["valid"]:
                        trajectory_error="invalid_physical_transition_fixed_2px_threshold"
                        break
            except BaseException as error:
                trajectory_error=type(error).__name__+": "+str(error)
                failure=trajectory_error
                exception_state=dict(state=env.state().tolist(),control_step=env.control_step,
                    substep_penetrations=np.asarray(env.substep_penetrations).tolist(),
                    pending_control=ledger.value["trajectories"][attempt]["pending_control"])
                if exception_state["pending_control"] is not None:
                    partial=len(env.substep_penetrations)
                    ledger.value["recorded_partial_substeps_lower_bound"]+=partial
                    exception_state.update(recorded_partial_substeps_lower_bound=partial,
                        exact_partial_substep_execution_unconfirmed=True,
                        charged_control_remains_full=True)
                    ledger.save()
            relative = f"trajectories/{trajectory_id}.npz"
            path=data/relative;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("xb") as stream:
                np.savez_compressed(stream,states=np.asarray(states,dtype=np.float64),
                    actions=np.asarray(actions,dtype=np.float32).reshape(-1,2),contact=np.asarray(contacts,dtype=np.int16),
                    valid_transition=np.asarray(valid,dtype=np.bool_),penetrations=np.asarray(penetrations,dtype=np.float64).reshape(-1,3),
                    substep_penetrations=np.asarray(substeps,dtype=np.float64).reshape(-1,10,3),
                    contact_points=np.asarray(points,dtype=np.float32).reshape(-1,2),contact_offsets=np.asarray(offsets,dtype=np.int32),
                    contact_angles=np.asarray(angles,dtype=np.float32))
            metadata_relative=relative.removesuffix(".npz")+".json"
            atomic_json(data/metadata_relative,dict(trajectory_id=trajectory_id,request=request,scene=scene,
                steps=step_records,error=trajectory_error,exception_state=exception_state,hidden_physics_controls=0))
            entry=dict(shape=letter,letter=letter,trajectory_id=trajectory_id,file=relative,sha256=sha256(path),
                metadata_file=metadata_relative,metadata_sha256=sha256(data/metadata_relative),split="stage1_probe",
                seed=request["seed"],location=location,spatial_class=location,controls=len(actions),
                gap_band_requested=request.get("gap_band"),gap_requested_world_px=request.get("gap_world_px"),
                initial_scene=states[0].tolist(),initial_scene_hash=hashlib.sha256(states[0].tobytes()).hexdigest(),
                invalid_controls=sum(not x for x in valid),error=trajectory_error,
                parent_approval_required_before_training=True)
            entries.append(entry)
            ledger.finish(attempt,"failed" if trajectory_error else "completed",saved_controls=len(actions),error=trajectory_error)
            atomic_json(output/"collection-progress.json",dict(completed_scene_slots=sequence+1,planned_scene_slots=54,
                saved_trajectories=len(entries),charged_controls=ledger.value["charged_controls"],last=entry))
            print(json.dumps(dict(scene=sequence+1,letter=letter,location=location,
                charged_controls=ledger.value["charged_controls"],saved_controls=len(actions),error=trajectory_error)),flush=True)
            if failure:
                break
    except BaseException as error:
        failure=type(error).__name__+": "+str(error)
    finally:
        manifest=dict(kind="native-rgb-supervised-probe.v1",stage=1,source_root=str(Path(physics_root).resolve()),
            controller=CONTROLLER,protocol_sha256=PROTOCOL_SHA256,state_columns=list(STATE_COLUMNS),config=config,
            config_sha256=sha256(config_path),source_identity_sha256=sha256(output/"source-identity.json"),
            entries=entries,planned_scene_slots=54,reset_failures=reset_failures,failure=failure,
            charged_controls=ledger.value["charged_controls"],actual_controls=ledger.value["completed_controls"],
            saved_controls=sum(row["controls"] for row in entries),wall_clock_seconds=time.monotonic()-started,
            model_inference_calls=0,optimizer_updates=0,unseen_predictions=0)
        atomic_json(data/"manifest.json",manifest)
        atomic_json(output/"contact-coverage.json",dict(segments=coverage,scope="new stage1 actual collision points only"))
        # Canonical reachability is kept distinct from actual new scene reachability.
        from tdyssey.native_rgb_data import _reachable_segments
        canonical={}
        for letter in config["letters"]:
            env=make_env(physics_root,letter);env.reset_scene()
            canonical[letter]=_reachable_segments(env,coverage[letter])
        _write_coverage(output/"coverage",coverage,canonical)
        ledger.value["collection_wall_seconds"]=time.monotonic()-started
        ledger.value["storage_after_collection"]=storage_check(config,output)
        ledger.value["historical_plus_new_charged_controls"]=34560+ledger.value["charged_controls"]
        ledger.value["physics_substeps_lower_bound"]=ledger.value["physics_substeps"]+ledger.value["recorded_partial_substeps_lower_bound"]
        ledger.value["charged_physics_substep_allowance"]=ledger.value["charged_controls"]*10
        ledger.value["physics_substeps_scope"]="physics_substeps counts fully completed controls; partial recorded substeps are separately a lower bound; pending controls remain fully charged"
        ledger.save()
        state["resources"].update(controls_reserved=ledger.value["charged_controls"],
            controls_actual=ledger.value["completed_controls"],physics_substeps=ledger.value["physics_substeps"],
            partial_substeps_lower_bound=ledger.value["recorded_partial_substeps_lower_bound"])
        state["collection_complete"]=True
        state["collection_failure"]=failure
        atomic_json(output/"stage-state.json",state)
    return manifest


def audit_probe(output):
    import numpy as np
    from tdyssey.native_rgb_env import make_env, PROTOCOL_SHA256
    output=Path(output);data=output/"probe"
    manifest=json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    verify_source(output)
    issues=[];rows=[];maxima=np.zeros(3);invalid=0;substep_count=0;seen=set();environments={}
    for entry in manifest["entries"]:
        trajectory_issues=[];p=data/entry["file"]
        if sha256(p)!=entry["sha256"] or sha256(data/entry["metadata_file"])!=entry["metadata_sha256"]:
            trajectory_issues.append("file_identity")
        if entry["trajectory_id"] in seen:trajectory_issues.append("duplicate_trajectory")
        seen.add(entry["trajectory_id"])
        with np.load(p,allow_pickle=False) as a:
            n=len(a["actions"]);states=a["states"];sub=a["substep_penetrations"];penetration=a["penetrations"]
            if states.shape!=(n+1,15) or sub.shape!=(n,10,3):trajectory_issues.append("timeline_alignment")
            if not np.isfinite(states).all() or not np.isfinite(sub).all():trajectory_issues.append("nonfinite")
            if a["actions"].shape!=(n,2) or not np.isfinite(a["actions"]).all() or (a["actions"]<0).any() or (a["actions"]>511).any():trajectory_issues.append("absolute_action_contract")
            if n and not np.allclose(np.max(sub,axis=1),penetration[1:],rtol=0,atol=1e-6):trajectory_issues.append("substep_max_mismatch")
            computed=(np.max(sub,axis=(1,2))<=2) if n else np.ones(0,dtype=bool)
            if not np.array_equal(computed,a["valid_transition"]):trajectory_issues.append("valid_flag_mismatch")
            bad=int((~computed).sum());invalid+=bad;substep_count+=n*10
            if n:maxima=np.maximum(maxima,np.max(sub,axis=(0,1)))
            env=environments.get(entry["shape"])
            if env is None:env=make_env(manifest["source_root"],entry["shape"]);env.reset_scene();environments[entry["shape"]]=env
            for index,s in enumerate(states):
                env.set_render_state(s);pnow=env.penetrations()
                if (pnow>penetration[index]+1e-3).any():trajectory_issues.append("underreported_endpoint_penetration")
            offsets=a["contact_offsets"]
            if offsets.shape!=(n+1,) or offsets[-1]!=len(a["contact_points"]) or len(a["contact_angles"])!=len(a["contact_points"]):trajectory_issues.append("contact_alignment")
            if hashlib.sha256(states[0].tobytes()).hexdigest()!=entry["initial_scene_hash"]:trajectory_issues.append("scene_identity")
        if trajectory_issues:issues.extend(trajectory_issues)
        rows.append(dict(trajectory_id=entry["trajectory_id"],invalid_transitions=bad,issues=sorted(set(trajectory_issues))))
    if manifest["protocol_sha256"]!=PROTOCOL_SHA256:issues.append("protocol_identity")
    if manifest["failure"]:issues.append("collection_failure")
    if invalid:issues.append("invalid_real_transitions_retained_and_reported")
    report=dict(kind="native-rgb-supervised-physics-audit.v1",passed=not issues,
        manifest_sha256=sha256(data/"manifest.json"),issues=sorted(set(issues)),invalid_real_transitions=invalid,
        max_substep_penetration_world_px=maxima.tolist(),audited_physics_substeps=substep_count,
        controls=sum(e["controls"] for e in manifest["entries"]),trajectories=rows,
        threshold_world_px=2,threshold_relaxed=False,extra_physics_controls=0,model_calls=0)
    atomic_json(output/"physics-audit.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["plan","collect","audit"])
    parser.add_argument("--config",type=Path,default=ROOT/"configs/native_rgb_supervised_stage1.json")
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/native-rgb-supervised-20261029")
    parser.add_argument("--physics-root",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    read_supervision(args.output)
    if args.stage=="plan":
        from tdyssey.native_rgb_supervised_sampling import plan_scenes
        cfg=json.loads(args.config.read_text(encoding="utf-8"));requests=plan_scenes(cfg)
        result=dict(requests=requests,storage=storage_check(cfg,args.output),controls_executed=0,
            planned_controls=len(requests)*cfg["controls_per_trajectory"])
        atomic_json(args.output/"pre-collection-plan.json",result)
    elif args.stage=="collect":
        if not args.execute:raise RuntimeError("Real collection requires explicit --execute")
        result=collect(args.config,args.output,args.physics_root)
    else:result=audit_probe(args.output)
    print(json.dumps({k:v for k,v in result.items() if k in ("failure","passed","charged_controls","actual_controls","controls_executed","planned_controls","issues")},indent=2))
    return 1 if result.get("failure") or result.get("passed") is False else 0


if __name__=="__main__":raise SystemExit(main())
