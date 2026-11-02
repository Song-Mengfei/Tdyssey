"""Bounded server CPU collection in independent, deterministic letter workers.

Only the parent writes the shared control-budget ledger. Each child owns a
letter/split, its contact-coverage counters and its files. This module never loads
a world model or imports Torch. Importing it does not launch any process.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
import json
import math
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import shutil
import sys
import time
import traceback

from .native_rgb_data import (BudgetLedger, TRAIN_LETTERS, UNSEEN_LETTERS,
    _validate_collection_config, _wall_scene, _reachable_segments, _run_actions,
    _contact_coverage, _write_coverage, contour_segments, scene_seed,
    capture_collection_source, file_sha256, audit_dataset)
from .native_rgb_env import (CONTROLLER, CONTROL_PROTOCOL, PROTOCOL_SHA256,
    STATE_COLUMNS, make_env, rotational_split_audit)


def write_json(path,value):
    """Retry only an atomic IO sink, never physical work or ledger mutations."""
    from .native_rgb_training import atomic_json
    return atomic_json(path,value)


def _reserve_budget(ledger,split,controls,reason):
    before=deepcopy(ledger.value);attempt_id=len(before["attempts"])
    try:return ledger.reserve(split,controls,reason)
    except PermissionError:
        expected=deepcopy(before);expected["charged_controls"][split]+=controls
        expected["attempts"].append(dict(id=attempt_id,split=split,reserved_controls=controls,
            actual_controls=0,reason=reason,status="reserved"))
        if ledger.value!=expected:raise
        # The base method already completed its one mutation. Re-serialize that
        # exact value through the bounded IO sink; do not call reserve again.
        write_json(ledger.path,ledger.value)
        return attempt_id


def _finish_budget(ledger,attempt_id,actual,**extra):
    before=deepcopy(ledger.value)
    try:return ledger.finish(attempt_id,actual,**extra)
    except PermissionError:
        expected=deepcopy(before);status=extra.get("status","completed")
        fields={key:value for key,value in extra.items() if key!="status"}
        expected["attempts"][attempt_id].update(actual_controls=int(actual),status=status,**fields)
        expected["actual_controls"]+=actual
        if ledger.value!=expected:raise
        # As above, preserve exactly one accounting update across transient IO.
        write_json(ledger.path,ledger.value)


def sampling_policy(config):
    """Freeze sampler choices before collection; controller physics is unchanged."""
    collection=config.get("collection",{})
    distances=collection.get("staging_distances",[18,16])
    if not isinstance(distances,(list,tuple)) or not distances or any(
            isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
            or value<15.5 or value>32 for value in distances):
        raise ValueError("Staging distances must be finite, unique world-pixel values in [15.5,32]")
    distances=[float(value) for value in distances]
    if len(set(distances))!=len(distances):raise ValueError("Duplicate staging distances")
    misses=collection.get("max_same_approach_misses",2)
    if type(misses)is not int or not 1<=misses<=2:raise ValueError("Same-approach miss limit must be integer 1 or 2")
    return dict(version="fair-arc-approach-v2",arc_segment_px=8,staging_distances=distances,
        max_same_approach_misses=misses,initial_approach_modes=[0,1,2],
        initial_approach_names=["normal","oblique_left_45deg","oblique_right_45deg"],
        fairness="planned-attempt tiers of miss-limit width, then low actual hit count; no permanent exclusion",
        approach_rotation="after miss limit rotate direction/staging; exhausted variants restart only at an independent pose",
        closed_holes="always excluded",pusher_radius=15,clearance_margin=.5,
        commanded_target_distances=[24,96,256],command_hold_controls=[2,4,8,16],
        physics_protocol_sha256=PROTOCOL_SHA256,actual_hit_set="real collision mappings only; planned targets never add hits")


def staging_candidates(env,segments,distances):
    """Geometry-only candidates, including shorter legal staging in tight spaces."""
    import numpy as np
    from shapely.geometry import Point,LineString,box
    free=box(22.5,22.5,488.5,488.5).difference(env.world_geometry().buffer(15.5,resolution=32))
    pieces=list(free.geoms) if hasattr(free,"geoms") else [free]
    pieces=[piece for piece in pieces if piece.area>1e-8]
    if not pieces:return []
    exterior=max(pieces,key=lambda piece:piece.area).buffer(1e-7)
    c,s=math.cos(env.block.angle),math.sin(env.block.angle)
    rotation=np.asarray(((c,-s),(s,c)))
    rows=[]
    for segment in segments:
        if segment["ring"]:continue
        point=np.asarray(env.block.local_to_world(segment["point"]))
        normal=rotation@np.asarray(segment["normal"]);contact_center=point+normal*15.5
        if not exterior.covers(Point(*contact_center)):continue
        for distance in distances:
            staging=point+normal*distance;path=LineString((staging,contact_center))
            if not exterior.covers(Point(*staging)) or not exterior.covers(path):continue
            rows.append(dict(segment_id=segment["id"],status="reachable_exterior_component",
                staging=staging.tolist(),staging_distance=float(distance),point=point.tolist(),normal=normal.tolist(),
                contact_center=contact_center.tolist(),normal_contact_path=[staging.tolist(),contact_center.tolist()],
                path_evidence="staging/contact-center/segment in conservative exterior-connected free space; no physical step",
                pusher_radius=15,conservative_clearance_margin=.5))
    return rows


class AdaptiveArcSampler:
    """Fair planned exposure and bounded misses; actual collision coverage is separate."""
    def __init__(self,segments,policy):
        self.segments=segments;self.policy=policy;self.serial=0;self.pending={}
        self.variants=[(distance,mode) for distance in policy["staging_distances"] for mode in policy["initial_approach_modes"]]
        self.stats={row["id"]:dict(segment_id=row["id"],planned_attempts=0,actual_target_hits=0,
            approach_cycle=0,preferred_variant=0,consecutive_misses={},variant_attempts={},variant_target_hits={},
            last_pose_token=None) for row in segments}
    def choose(self,candidates,rng,pose_token):
        if not candidates:return None
        grouped={}
        for candidate in candidates:
            if self.segments[candidate["segment_id"]]["ring"]:continue
            grouped.setdefault(candidate["segment_id"],[]).append(candidate)
        if not grouped:return None
        limit=self.policy["max_same_approach_misses"]
        def priority(segment_id):
            stats=self.stats[segment_id];segment=self.segments[segment_id]
            return (stats["planned_attempts"]//limit,int(segment["contact_count"]>0),
                stats["actual_target_hits"],segment["contact_count"],stats["planned_attempts"],
                -len(segment.get("features",())),float(rng.random()))
        selected=min(grouped,key=priority);stats=self.stats[selected]
        available={candidate["staging_distance"]:candidate for candidate in grouped[selected]}
        ordered=[(stats["preferred_variant"]+offset)%len(self.variants) for offset in range(len(self.variants))]
        def permitted(index):
            distance,mode=self.variants[index]
            return distance in available and stats["consecutive_misses"].get(str(index),0)<limit
        possible=[index for index in ordered if permitted(index)]
        if not possible:
            if stats["last_pose_token"]==pose_token:
                raise RuntimeError("Exhausted approaches require a new independent pose, not a retry at the same pose")
            stats["approach_cycle"]+=1;stats["consecutive_misses"]={}
            possible=[index for index in ordered if self.variants[index][0] in available]
        index=possible[0];distance,mode=self.variants[index]
        stats["planned_attempts"]+=1;stats["last_pose_token"]=pose_token
        stats["variant_attempts"][str(index)]=stats["variant_attempts"].get(str(index),0)+1
        self.serial+=1
        token=dict(choice_id=self.serial,segment_id=selected,variant=index,initial_approach_mode=mode,
            staging_distance=distance,pose_token=pose_token,approach_cycle=stats["approach_cycle"],
            planned_attempt_number=stats["planned_attempts"],fairness_tier=(stats["planned_attempts"]-1)//limit)
        self.pending[self.serial]=token
        self.segments[selected]["planned_attempts"]=stats["planned_attempts"]
        self.segments[selected]["actual_target_hits"]=stats["actual_target_hits"]
        return dict(available[distance],initial_approach_mode=mode,sampler_choice=dict(token))
    def record_outcome(self,choice_id,target_hit):
        token=self.pending.pop(choice_id);stats=self.stats[token["segment_id"]];key=str(token["variant"])
        if target_hit:
            stats["actual_target_hits"]+=1;stats["consecutive_misses"][key]=0
            stats["variant_target_hits"][key]=stats["variant_target_hits"].get(key,0)+1
        else:
            stats["consecutive_misses"][key]=stats["consecutive_misses"].get(key,0)+1
            if stats["consecutive_misses"][key]>=self.policy["max_same_approach_misses"]:
                stats["preferred_variant"]=(token["variant"]+1)%len(self.variants)
        self.segments[token["segment_id"]]["actual_target_hits"]=stats["actual_target_hits"]
    def snapshot(self):return [self.stats[index] for index in sorted(self.stats)]


def allocate_jobs(config,caps,charged,splits):
    """Allocate remaining controls in trajectory/letter round-robin order."""
    collection=config.get("collection",{});controls=int(collection.get("controls",64))
    if not 4<=controls<=256:raise ValueError("Trajectory controls must be in [4,256]")
    requested=dict(train=int(collection.get("train_trajectories",24)),
        seen_val=int(collection.get("val_trajectories",6)),
        seen_test=int(collection.get("seen_test_trajectories",0)),
        unseen_test=int(collection.get("unseen_test_trajectories",0)))
    jobs=[]
    for split in splits:
        if split not in caps or not 1<=requested.get(split,0)<1000:raise ValueError("Split is not explicitly budgeted")
        letters=UNSEEN_LETTERS if split=="unseen_test" else TRAIN_LETTERS
        rows={letter:[] for letter in letters};remaining=caps[split]-charged.get(split,0)
        if remaining<0:raise RuntimeError("Existing control ledger exceeds split budget")
        for repetition in range(requested[split]):
            for letter_index,letter in enumerate(letters):
                if remaining<4:continue
                count=min(controls,remaining);remaining-=count
                rows[letter].append(dict(repetition=repetition,controls=count,
                    seed=scene_seed(config.get("data_seed",20261028),split,repetition,letter_index)))
        for letter in letters:
            if rows[letter]:jobs.append(dict(id=f"{split}-{letter}",shape=letter,split=split,
                scenes=rows[letter],reserved_controls=sum(row["controls"] for row in rows[letter])))
    return jobs


def _worker_entry(worker,job,queue):
    try:
        result=worker(job)
        queue.put(dict(status="completed",result=result))
    except BaseException as exc:
        queue.put(dict(status="failed",error=dict(type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc()[-4000:])))


def _known_progress(job):
    path=job.get("progress_file")
    if path and Path(path).exists():
        try:return json.loads(Path(path).read_text(encoding="utf8"))
        except (OSError,ValueError):pass
    return dict(actual_controls=0,rejected_resets=0)


def run_budgeted_jobs(jobs,worker,ledger,*,workers=2,job_timeout_seconds=1800,total_timeout_seconds=21600):
    """Spawn at most N children; timeout/failure cancels only owned children.

    Reservations precede process.start(). Interrupted work remains fully charged;
    actual_controls on a failure is explicitly a persisted lower bound. No retry
    or budget reset occurs. Pending, never-started jobs are not charged.
    """
    if type(workers)is not int or not 1<=workers<=16:raise ValueError("Worker count must be an integer in [1,16]")
    if any(not math.isfinite(value) or value<=0 for value in (job_timeout_seconds,total_timeout_seconds)):
        raise ValueError("Invalid finite deadline")
    identifiers=[job["id"] for job in jobs]
    if any(type(key)is not str or not key for key in identifiers) or len(set(identifiers))!=len(identifiers):
        raise ValueError("Every owned job must have a distinct nonempty ID")
    if any(type(job["reserved_controls"])is not int or job["reserved_controls"]<0 for job in jobs):
        raise ValueError("Every owned job needs a finite integer control reservation")
    context=multiprocessing.get_context("spawn")
    pending=list(jobs);active={};results=[];peak=0;started=time.monotonic();fatal=None
    def finish(key,payload):
        owned=active.pop(key);process=owned["process"];queue=owned["queue"]
        if process.pid is not None:
            process.join(timeout=3)
            if process.is_alive():process.terminate();process.join(timeout=3)
            if process.is_alive():process.kill();process.join(timeout=3)
        queue.close();queue.cancel_join_thread()
        progress=payload.get("result",{}) if payload["status"]=="completed" else _known_progress(owned["job"])
        actual=progress.get("actual_controls",0)
        if type(actual)is not int or not 0<=actual<=owned["job"]["reserved_controls"]:
            payload=dict(status="failed",error=dict(type="InvalidWorkerAccounting",message="Child exceeded its reservation"))
            actual=0
        _finish_budget(ledger,owned["reservation"],actual,status=payload["status"],
            worker_pid=process.pid,job_id=key,actual_is_lower_bound=payload["status"]!="completed")
        ledger.value["rejected_resets"]+=int(progress.get("rejected_resets",0))
        write_json(ledger.path,ledger.value)
        results.append(dict(job_id=key,pid=process.pid,exitcode=process.exitcode,
            reservation=owned["reservation"],progress_file=owned["job"].get("progress_file"),
            known_actual_controls=actual,actual_is_lower_bound=payload["status"]!="completed",**payload))
        return payload
    try:
        while pending or active:
            while pending and len(active)<workers and fatal is None:
                job=pending.pop(0)
                reservation=_reserve_budget(ledger,job["split"],job["reserved_controls"],f"parallel letter job {job['id']}; child scene attempts retained")
                queue=context.Queue(maxsize=1)
                process=context.Process(target=_worker_entry,args=(worker,job,queue),name=f"native-rgb-{job['id']}")
                active[job["id"]]=dict(job=job,reservation=reservation,process=process,queue=queue,start=time.monotonic())
                process.start();peak=max(peak,len(active))
            for key in list(active):
                owned=active[key]
                try:payload=owned["queue"].get_nowait()
                except Empty:payload=None
                if payload is not None:
                    payload=finish(key,payload)
                    if payload["status"]!="completed":fatal=payload.get("error",{"message":"worker failure"})
                elif time.monotonic()-owned["start"]>job_timeout_seconds:
                    fatal=dict(type="WorkerTimeout",message=f"Owned job {key} exceeded deadline")
                    finish(key,dict(status="failed",error=fatal))
                elif owned["process"].exitcode is not None:
                    try:payload=owned["queue"].get(timeout=.1)
                    except Empty:payload=None
                    if payload is not None:
                        payload=finish(key,payload)
                        if payload["status"]!="completed":fatal=payload.get("error",{"message":"worker failure"})
                    else:
                        fatal=dict(type="WorkerExited",message=f"Owned job {key} exited without a result")
                        finish(key,dict(status="failed",error=fatal))
            if time.monotonic()-started>total_timeout_seconds:
                fatal=dict(type="CollectionTimeout",message="Overall collection deadline exceeded")
            if fatal:
                for key in list(active):finish(key,dict(status="cancelled",error=fatal))
                break
            if active:time.sleep(.02)
    except BaseException as exc:
        fatal=dict(type=type(exc).__name__,message=str(exc))
        for key in list(active):finish(key,dict(status="cancelled",error=fatal))
    return dict(status="failed" if fatal else "completed",error=fatal,results=results,
        pending_job_ids=[job["id"] for job in pending],requested_workers=int(workers),
        max_active_workers=peak,seconds=time.monotonic()-started,
        budget=ledger.value,no_automatic_retry=True,only_owned_processes_terminated=True)


def collect_letter_job(job):
    """Child-only implementation. It never reads/writes the shared BudgetLedger."""
    import numpy as np
    import cv2
    from shapely import affinity
    cv2.setNumThreads(1)
    output=Path(job["output"]);config=job["config"];collection=config.get("collection",{})
    policy=job.get("sampling_policy",sampling_policy(config))
    if policy!=sampling_policy(config):raise ValueError("Worker sampling policy differs from the parent-frozen policy")
    letter=job["shape"];split=job["split"];job_dir=Path(job["job_dir"])
    job_dir.mkdir(parents=True,exist_ok=True)
    env=make_env(job["source_root"],letter,CONTROLLER);env.reset_scene()
    segments=contour_segments(letter,8);canonical=_reachable_segments(env,segments)
    sampler=AdaptiveArcSampler(segments,policy)
    entries=[];attempts=[];scene_hashes=set();actual=0;rejected=0
    locations=("center","left","right","top","bottom","left-top","right-top","left-bottom","right-bottom")
    def save_progress():
        write_json(job["progress_file"],dict(actual_controls=actual,rejected_resets=rejected,
            completed_trajectories=len(entries),worker_pid=os.getpid(),attempts=attempts,
            sampling_policy=policy,sampler_bin_stats=sampler.snapshot(),pending_choices=list(sampler.pending.values())))
    save_progress()
    for scene in job["scenes"]:
        repetition=scene["repetition"];count=scene["controls"];seed=scene["seed"]
        rng=np.random.default_rng(seed);location=locations[repetition%len(locations)]
        chosen=None;reset_records=[]
        for attempt in range(int(collection.get("max_reset_attempts",4))):
            _wall_scene(env,location);env.block.angle=float(rng.uniform(-math.pi,math.pi))
            xmin,ymin,xmax,ymax=env.world_geometry().bounds
            dx=(10-xmin) if "left" in location else (501-xmax) if "right" in location else max(0,10-xmin)-max(0,xmax-501)
            dy=(10-ymin) if "top" in location else (501-ymax) if "bottom" in location else max(0,10-ymin)-max(0,ymax-501)
            env.block.position=tuple(np.asarray(env.block.position)+(dx,dy))
            candidates=staging_candidates(env,segments,policy["staging_distances"])
            if candidates:
                chosen=sampler.choose(candidates,rng,pose_token=f"{seed}:{attempt}")
                env.agent.position=tuple(chosen["staging"])
                env.space.reindex_shapes_for_body(env.agent);env.space.reindex_shapes_for_body(env.block)
            issues=[]
            if chosen is None:issues.append("no_exterior_connected_contact_staging")
            if env.penetrations().max()>.1:issues.append("initial_penetration")
            identity=hashlib.sha256(env.state().tobytes()).hexdigest()
            if identity in scene_hashes:issues.append("duplicate_scene")
            reset_records.append(dict(attempt=attempt,issues=issues,initial_state=env.state().tolist(),reachability=chosen))
            if not issues:break
            if chosen is not None:sampler.record_outcome(chosen["sampler_choice"]["choice_id"],False)
            rejected+=1;chosen=None;attempts.append(dict(seed=seed,status="rejected_reset",details=reset_records[-1]));save_progress()
        if chosen is None:continue
        scene_hashes.add(identity);trajectory_id=f"{split}-{letter}-{repetition:04d}-{seed}"
        path=output/"trajectories"/f"{trajectory_id}.npz"
        if path.exists():raise FileExistsError("Never overwrite a child trajectory")
        local_normal=np.asarray(segments[chosen["segment_id"]]["normal"]);metadata=[]
        save_progress()
        def actions():
            phase=0;done=0
            while done<count:
                c,s=math.cos(env.block.angle),math.sin(env.block.angle)
                normal=np.asarray(((c,-s),(s,c)))@local_normal;tangent=np.asarray((-normal[1],normal[0]))
                mode=(repetition+phase)%8 if phase else chosen["initial_approach_mode"]
                direction=(-normal,(-normal+tangent)/math.sqrt(2),(-normal-tangent)/math.sqrt(2),
                    tangent,-tangent,normal,(normal+tangent)/math.sqrt(2),(normal-tangent)/math.sqrt(2))[mode]
                distance=(24,96,256)[(repetition+phase)%3];duration=(2,4,8,16)[(repetition+phase)%4]
                target=np.clip(np.asarray(env.agent.position)+direction*distance,0,511)
                for _ in range(min(duration,count-done)):
                    state=env.state();metadata.append(dict(mode=mode,distance_category=("near","mid","far")[(repetition+phase)%3],
                        requested_distance=distance,duration=duration,moving=bool(np.linalg.norm(state[7:9])>.5 or abs(state[9])>.02)))
                    yield target;done+=1
                phase+=1
        stats=_run_actions(env,actions(),path,stop_invalid=True);actual+=stats["controls"]
        with np.load(path,allow_pickle=False) as arrays:
            for t in range(stats["controls"]):
                lo,hi=arrays["contact_offsets"][t:t+2]
                _contact_coverage(letter,segments,arrays["contact_points"][lo:hi],trajectory_id,
                    metadata[t]["moving"],str(metadata[t]["mode"]),arrays["contact_angles"][lo:hi])
            near_walls={name:0 for name in ("left","right","top","bottom")};corners=0
            for state in arrays["states"]:
                geom=affinity.rotate(env.local_geometry,float(state[4]),origin=(0,0),use_radians=True)
                geom=affinity.translate(geom,*state[2:4]);xmin,ymin,xmax,ymax=geom.bounds
                close=(xmin<23,xmax>488,ymin<23,ymax>488)
                for name,flag in zip(near_walls,close):near_walls[name]+=int(flag)
                corners+=int((close[0] or close[1]) and (close[2] or close[3]))
        target_hit=trajectory_id in segments[chosen["segment_id"]]["trajectory_ids"]
        sampler.record_outcome(chosen["sampler_choice"]["choice_id"],target_hit)
        entries.append(dict(trajectory_id=trajectory_id,shape=letter,split=split,seed=seed,
            file=str(path.relative_to(output)).replace("\\","/"),sha256=file_sha256(path),
            initial_scene_hash=identity,location=location,staging_segment=chosen["segment_id"],
            staging_distance=chosen["staging_distance"],initial_approach_mode=chosen["initial_approach_mode"],
            sampler_choice=chosen["sampler_choice"],planned_arc_actually_hit=target_hit,
            reset_records=reset_records,action_metadata=metadata[:stats["controls"]],
            actual_wall_frame_counts=near_walls,actual_corner_frames=corners,**stats))
        attempts.append(dict(seed=seed,trajectory_id=trajectory_id,reserved_controls=count,status="completed",**stats));save_progress()
        if actual>job["reserved_controls"]:raise RuntimeError("Child exceeded its finite control reservation")
    result=job_dir/"result.json"
    write_json(result,dict(entries=entries,segments=segments,canonical_reachability=canonical,
        actual_controls=actual,rejected_resets=rejected,worker_pid=os.getpid(),attempts=attempts,
        sampling_policy=policy,sampler_bin_stats=sampler.snapshot()))
    return dict(result_file=str(result),actual_controls=actual,rejected_resets=rejected,
        worker_pid=os.getpid(),torch_imported="torch" in sys.modules)


def collect_dataset_parallel(config,output,source_root,splits=("train","seen_val","seen_test","unseen_test")):
    """Server entry point. Call only under the CLI's __main__ process guard."""
    output=Path(output).resolve();caps=_validate_collection_config(config);experiment=output.parent
    policy=sampling_policy(config)
    if (output/"manifest.json").exists():raise FileExistsError("Never overwrite an existing dataset")
    probe=json.loads((experiment/"physics-probe/report.json").read_text(encoding="utf8"))
    if not probe.get("physics_probe_gate_passed") or probe.get("protocol_sha256")!=PROTOCOL_SHA256:
        raise RuntimeError("The matching physical probe must pass before parallel collection")
    ledger=BudgetLedger(experiment,caps)
    if any(row.get("reason","").startswith("parallel letter job") for row in ledger.value["attempts"]):
        raise RuntimeError("A prior parallel attempt exists; inspect its retained partial files before a new authorized run")
    existing_capture=(experiment/"collection-source-start.json").exists()
    source_identity=capture_collection_source(experiment,source_root)
    source_record=experiment/source_identity["file"]
    source=json.loads(source_record.read_text(encoding="utf8"))
    if existing_capture:
        raise RuntimeError("Parallel collection requires a fresh source snapshot; preserve and inspect the earlier run")
    for relative in ("tdyssey/native_rgb_parallel.py","tdyssey/native_rgb_training.py"):
        source_path=Path(__file__).resolve().parents[1]/relative
        snapshot=experiment/"collection-source"/relative
        snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source_path,snapshot)
        source["files"].append(dict(snapshot=str(snapshot.relative_to(experiment)).replace("\\","/"),
            bytes=snapshot.stat().st_size,sha256=file_sha256(snapshot)))
    write_json(source_record,source);source_identity["sha256"]=file_sha256(source_record)
    jobs=allocate_jobs(config,caps,ledger.value["charged_controls"],splits)
    if not jobs:raise RuntimeError("No collection budget remains")
    configured_workers=config.get("collection",{}).get("workers",1)
    if type(configured_workers)is not int:raise ValueError("Configured workers must be an integer")
    workers=min(configured_workers,len(jobs))
    if not 1<=workers<=16:raise ValueError("Server collection workers must be in [1,16]")
    policy_record=experiment/"sampling-policy.json"
    if policy_record.exists():raise FileExistsError("Never overwrite an existing pre-collection sampling policy")
    write_json(policy_record,dict(kind="native-rgb-server-sampling-policy-v2",policy=policy,
        written_before_workers=True,collector_source_sha256=file_sha256(Path(__file__)),
        physics_protocol_sha256=PROTOCOL_SHA256))
    policy_identity=dict(file=policy_record.name,sha256=file_sha256(policy_record))
    for job in jobs:
        directory=output/"parallel-jobs"/job["id"]
        job.update(config=config,output=str(output),source_root=str(Path(source_root).resolve()),
            job_dir=str(directory),progress_file=str(directory/"progress.json"),sampling_policy=policy)
    run=run_budgeted_jobs(jobs,collect_letter_job,ledger,workers=workers,
        job_timeout_seconds=float(config.get("collection",{}).get("job_timeout_seconds",1800)),
        total_timeout_seconds=float(config.get("collection",{}).get("total_timeout_seconds",21600)))
    write_json(experiment/"parallel-collection.json",run)
    entries=[];by_split={split:{} for split in splits};canonical={split:{} for split in splits}
    for row in run["results"]:
        result=row.get("result",{});path=result.get("result_file")
        if not path:continue
        result=json.loads(Path(path).read_text(encoding="utf8"))
        selected=next(job for job in jobs if job["id"]==row["job_id"])
        split,letter=selected["split"],selected["shape"]
        entries.extend(result["entries"]);by_split[split][letter]=result["segments"]
        canonical[split][letter]=result["canonical_reachability"]
    summary={split:_write_coverage(output/"coverage"/split,rows,canonical[split]) for split,rows in by_split.items()}
    write_json(output/"coverage/summary.json",summary)
    manifest=dict(kind="native-rgb-compact-data-v1",controller=CONTROLLER,controller_protocol=CONTROL_PROTOCOL,
        protocol_sha256=PROTOCOL_SHA256,source_root=str(Path(source_root).resolve()),collector_source=source_identity,
        train_letters=list(TRAIN_LETTERS),unseen_letters=list(UNSEEN_LETTERS),entries=entries,config=config,
        state_columns=STATE_COLUMNS,render_size=96,control_hz=10,physics_hz=100,
        data_seed=int(config.get("data_seed",20261028)),actual_controls=sum(row["controls"] for row in entries),
        actual_trajectories=len(entries),requested_workers=int(config.get("collection",{}).get("workers",1)),
        actual_workers=run["max_active_workers"],worker_pids=[row["pid"] for row in run["results"]],
        collection_seconds=run["seconds"],coverage_summary=summary,budget=ledger.value,
        failure=run["error"],split_geometry_audit=rotational_split_audit(TRAIN_LETTERS,UNSEEN_LETTERS),
        sampling_policy=policy,sampling_policy_record=policy_identity,
        material_change="finite-mass force-limited pusher; unchanged letter geometry, object inertia and walls",
        privileged_sampling="exact contour, reachability, and real contact coverage; never model inputs/targets",
        parallel_policy="parent-only ledger reservations; independent letter/split children; no retry; fixed seed coordinates")
    write_json(output/"manifest.json",manifest)
    audit=audit_dataset(output)
    if run["status"]!="completed":raise RuntimeError("Parallel collection failed; all partial files and conservative reservations retained")
    return dict(manifest=manifest,audit=audit,parallel=run)
