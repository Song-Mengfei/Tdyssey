"""Finite native-RGB scene collection, physical audit, and bounded image windows.

Simulator geometry is used only for sampling and offline diagnostics.  Dataset
items expose only RGB/actions to the denoiser, never state or letter identifiers.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from .native_rgb_env import (CONTROLLER, CONTROL_PROTOCOL, PROTOCOL_SHA256,
    STATE_COLUMNS, make_env, rotational_split_audit, shape_geometry)

TRAIN_LETTERS=tuple("ABCE FHIJLMOPSTUVXY".replace(" ",""))
UNSEEN_LETTERS=tuple("DGKNQRWZ")
SPLIT_CAPS={"train":27648,"seen_val":6912}
SPLIT_SEED_OFFSETS={"train":0,"seen_val":1000000,"seen_test":2000000,"unseen_test":3000000}


def scene_seed(data_seed,split,repetition,letter_index):
    """Stable scene seeds, independent of requested split order or worker count."""
    if split not in SPLIT_SEED_OFFSETS or not 0<=repetition<1000 or not 0<=letter_index<26:
        raise ValueError("Invalid fixed scene seed coordinates")
    return int(data_seed)+SPLIT_SEED_OFFSETS[split]+repetition*1000+letter_index


def collection_caps(config):
    collection=config.get("collection",{})
    controls=int(collection.get("controls",64))
    caps=dict(train=len(TRAIN_LETTERS)*int(collection.get("train_trajectories",24))*controls,
              seen_val=len(TRAIN_LETTERS)*int(collection.get("val_trajectories",6))*controls)
    for split,key,letters in (("seen_test","seen_test_trajectories",TRAIN_LETTERS),
                               ("unseen_test","unseen_test_trajectories",UNSEEN_LETTERS)):
        if collection.get(key,0):caps[split]=len(letters)*int(collection[key])*controls
    if sum(caps.values())>int(collection.get("max_attempted_controls",sum(caps.values()))):
        raise ValueError("Split collection budgets exceed the declared aggregate cap")
    return caps


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    temporary.replace(path)


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def capture_collection_source(experiment_root,source_root):
    """Capture small source files before new collection, preserving earlier runs."""
    output=Path(experiment_root);record=output/"collection-source-start.json"
    if record.exists():return dict(file=record.name,sha256=file_sha256(record))
    package=Path(__file__).resolve().parents[1];physics=Path(source_root).resolve()
    pairs=[(package/relative,relative) for relative in ("tdyssey/native_rgb_env.py","tdyssey/native_rgb_data.py",
        "tdyssey/geometry.py","tdyssey/letters.py","tdyssey/envs/official.py","tdyssey/collection.py",
        "tdyssey/state.py","manifests/official_dp.json")]
    pairs.extend((physics/relative,"official-source/"+relative) for relative in (
        "diffusion_policy/env/pusht/pusht_env.py","diffusion_policy/env/pusht/pymunk_override.py"))
    rows=[]
    for source,relative in pairs:
        target=output/"collection-source"/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        rows.append(dict(snapshot=str(target.relative_to(output)).replace("\\","/"),
            bytes=source.stat().st_size,sha256=file_sha256(source)))
    write_json(record,dict(kind="native-rgb-collection-source-v1",capture="before collection controls",files=rows))
    return dict(file=record.name,sha256=file_sha256(record))


def verify_collection_source(data_root,manifest):
    identity=manifest.get("collector_source")
    if not identity:
        if manifest.get("kind")=="native-rgb-compact-data-v1":raise ValueError("Missing collection source fingerprint")
        return
    root=Path(data_root).parent.resolve()
    for identity in (identity,manifest.get("offline_review_source")):
        if identity is None:continue
        record=(root/identity["file"]).resolve()
        if not record.is_relative_to(root) or file_sha256(record)!=identity["sha256"]:
            raise ValueError("Collection/review source record changed")
        for row in json.loads(record.read_text())["files"]:
            snapshot=(root/row["snapshot"]).resolve()
            if not snapshot.is_relative_to(root) or file_sha256(snapshot)!=row["sha256"]:
                raise ValueError("Collection/review source snapshot changed")


class BudgetLedger:
    """Persist charges before execution. Unused reservations remain charged."""
    def __init__(self,root,caps=None):
        self.path=Path(root)/"budget.json"
        caps=caps or SPLIT_CAPS
        self.value=json.loads(self.path.read_text()) if self.path.exists() else dict(
            kind="native-rgb-budget-v1",caps=caps,total_cap=sum(caps.values()),
            attempts=[],charged_controls={key:0 for key in caps},actual_controls=0,
            rejected_resets=0,reset_physics_substeps=0)
        if self.value["caps"]!=caps:raise ValueError("Existing collection budget differs")
    def reserve(self,split,controls,reason):
        if split not in self.value["caps"] or type(controls)is not int or controls<0:raise ValueError("Invalid budget request")
        if self.value["charged_controls"][split]+controls>self.value["caps"][split]:raise RuntimeError("Finite collection budget exhausted")
        self.value["charged_controls"][split]+=controls
        self.value["attempts"].append(dict(id=len(self.value["attempts"]),split=split,
            reserved_controls=controls,actual_controls=0,reason=reason,status="reserved"))
        write_json(self.path,self.value)
        return len(self.value["attempts"])-1
    def finish(self,attempt,actual,status="completed",**extra):
        row=self.value["attempts"][attempt]
        if row["status"]!="reserved" or not 0<=actual<=row["reserved_controls"]:raise ValueError("Invalid budget completion")
        row.update(actual_controls=int(actual),status=status,**extra)
        self.value["actual_controls"]+=actual
        write_json(self.path,self.value)
    def reject(self,reason):
        self.value["rejected_resets"]+=1
        self.value["attempts"].append(dict(id=len(self.value["attempts"]),split=None,
            reserved_controls=0,actual_controls=0,status="rejected_reset",reason=reason))
        write_json(self.path,self.value)


def _save_trajectory(path,states,actions,contacts,valid,penetrations,points,offsets,angles=None):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,states=np.asarray(states,dtype=np.float64),
        actions=np.asarray(actions,dtype=np.float32).reshape(-1,2),contact=np.asarray(contacts,dtype=np.int16),
        valid_transition=np.asarray(valid,dtype=np.bool_),penetrations=np.asarray(penetrations,dtype=np.float32).reshape(-1,3),
        contact_points=np.asarray(points,dtype=np.float32).reshape(-1,2),contact_offsets=np.asarray(offsets,dtype=np.int32),
        contact_angles=np.asarray(angles if angles is not None else [],dtype=np.float32))


def _run_actions(env,actions,path,stop_invalid=True):
    states=[env.state()];used=[];contacts=[];valid=[];penetrations=[env.penetrations()];points=[];offsets=[0];angles=[]
    for action in actions:
        _,info=env.step(action)
        states.append(env.state());used.append(action);contacts.append(info["contact"])
        valid.append(info["valid"]);penetrations.append(info["penetration"])
        points.extend(info["contact_points"].tolist());offsets.append(len(points))
        angles.extend(info["contact_angles"].tolist())
        if stop_invalid and not info["valid"]:break
    _save_trajectory(path,states,used,contacts,valid,penetrations,points,offsets,angles)
    return dict(controls=len(used),valid_controls=sum(valid),contact_controls=sum(c>0 for c in contacts),
        contact_points=len(points),invalid_controls=sum(not v for v in valid),
        max_penetration=[max(float(row[i]) for row in penetrations) for i in range(3)],
        terminal_invalid=bool(valid and not valid[-1]))


def _wall_scene(env,location):
    """Initial placement only, outside solids/holes; no in-trajectory relocation."""
    import numpy as np
    env.reset_scene()
    xmin,ymin,xmax,ymax=env.world_geometry().bounds
    shift=np.zeros(2)
    if "left" in location:shift[0]=10-xmin
    if "right" in location:shift[0]=501-xmax
    if "top" in location:shift[1]=10-ymin
    if "bottom" in location:shift[1]=501-ymax
    env.block.position=tuple(np.asarray(env.block.position)+shift)
    xmin,ymin,xmax,ymax=env.world_geometry().bounds
    center=np.asarray(((xmin+xmax)/2,(ymin+ymax)/2))
    dx=1 if "left" in location else -1 if "right" in location else 0
    dy=1 if "top" in location else -1 if "bottom" in location else 0
    if location=="center":dx=1
    pusher=center.copy()
    if dx:pusher[0]=xmax+20 if dx>0 else xmin-20
    if dy:pusher[1]=ymax+20 if dy>0 else ymin-20
    env.agent.position=tuple(pusher)
    env.space.reindex_shapes_for_body(env.agent);env.space.reindex_shapes_for_body(env.block)
    return np.asarray((-dx,-dy),dtype=float),pusher


def physics_probe(config,output,source_root):
    """Paired near/mid/far and wall/corner probe, all controls charged globally."""
    import numpy as np
    output=Path(output)
    probe_dir=output/"physics-probe"
    if (probe_dir/"report.json").exists():raise FileExistsError("Do not repeat an inspected physics probe")
    ledger=BudgetLedger(output,collection_caps(config))
    steps=int(config.get("collection",{}).get("probe_controls",32))
    if not 1<=steps<=64:raise ValueError("Probe controls must be in [1,64]")
    locations=("center","left","right","top","bottom","left-top","right-top","left-bottom","right-bottom")
    rows=[]
    for letter in ("T","L"):
        for controller in ("upstream-kinematic",CONTROLLER):
            env=make_env(source_root,letter,controller)
            for index,location in enumerate(locations):
                direction,pusher=_wall_scene(env,location)
                distance=(24,96,256)[index%3] if location=="center" else 400
                target=np.clip(pusher+direction*distance,0,511)
                reservation=ledger.reserve("train",steps,f"physical probe {letter} {controller} {location}")
                path=probe_dir/f"{letter}-{controller}-{location}.npz"
                stats=_run_actions(env,[target]*steps,path,stop_invalid=False)
                ledger.finish(reservation,stats["controls"],**stats)
                rows.append(dict(shape=letter,controller=controller,location=location,
                    initial_target_distance=float(np.linalg.norm(target-pusher)),target=target.tolist(),
                    file=str(path.relative_to(output)),**stats))
            # Each distance category also has an independent contact-rich center scene.
            for label,distance in (("mid",96),("far",256)):
                direction,pusher=_wall_scene(env,"center")
                target=np.clip(pusher+direction*distance,0,511)
                reservation=ledger.reserve("train",steps,f"physical probe {letter} {controller} center-{label}")
                path=probe_dir/f"{letter}-{controller}-center-{label}.npz"
                stats=_run_actions(env,[target]*steps,path,stop_invalid=False)
                ledger.finish(reservation,stats["controls"],**stats)
                rows.append(dict(shape=letter,controller=controller,location=f"center-{label}",
                    initial_target_distance=float(np.linalg.norm(target-pusher)),target=target.tolist(),
                    file=str(path.relative_to(output)),**stats))
    candidate=[row for row in rows if row["controller"]==CONTROLLER]
    report=dict(kind="native-rgb-physical-probe-v1",controller=CONTROL_PROTOCOL,
        protocol_sha256=PROTOCOL_SHA256,rows=rows,
        physics_probe_gate_passed=all(row["invalid_controls"]==0 for row in candidate),
        candidate_max_penetration=[max(row["max_penetration"][i] for row in candidate) for i in range(3)],
        original_max_penetration=[max(row["max_penetration"][i] for row in rows if row["controller"]=="upstream-kinematic") for i in range(3)],
        material_change="finite-mass dynamic pusher, force/speed caps, solver50; same target coordinates and 10Hz/100Hz; not upstream parity",
        split_geometry_audit=rotational_split_audit(TRAIN_LETTERS,UNSEEN_LETTERS),
        budget=ledger.value)
    write_json(probe_dir/"report.json",report)
    return report


def contour_segments(letter,segment_px=8):
    """Fixed arc-length bins of the exact polygon union, including hole rings."""
    import numpy as np
    from shapely.geometry import Point,LineString
    geom=shape_geometry(letter)
    rings=[geom.exterior,*geom.interiors]
    rows=[]
    for ring_index,ring in enumerate(rings):
        vertices=np.asarray(ring.coords[:-1])
        for index,start in enumerate(np.arange(0,float(ring.length),segment_px)):
            end=min(start+segment_px,float(ring.length));arc=(start+end)/2
            mid=ring.interpolate(arc)
            before=ring.interpolate(max(0,arc-.01));after=ring.interpolate(min(ring.length,arc+.01))
            tangent=np.asarray((after.x-before.x,after.y-before.y));tangent/=np.linalg.norm(tangent)
            normal=np.asarray((-tangent[1],tangent[0]))
            if geom.contains(Point(mid.x+normal[0]*.01,mid.y+normal[1]*.01)):normal=-normal
            near_corner=bool(np.min(np.linalg.norm(vertices-np.asarray((mid.x,mid.y)),axis=1))<=segment_px)
            features=[]
            if ring_index:features.append("hole")
            if near_corner:features.append("corner_or_concavity")
            ray=LineString(((mid.x,mid.y),(mid.x-normal[0]*200,mid.y-normal[1]*200)))
            cross_section=geom.intersection(ray)
            pieces=list(cross_section.geoms) if hasattr(cross_section,"geoms") else [cross_section]
            local_widths=[piece.length for piece in pieces if piece.distance(mid)<1e-6 and piece.length>1e-6]
            if local_widths and min(local_widths)<=30.0001:features.append("thin_end_or_stroke")
            rows.append(dict(id=len(rows),ring=ring_index,ring_bin=index,arc_start=float(start),arc_end=float(end),
                point=[mid.x,mid.y],normal=normal.tolist(),features=features,
                contact_count=0,trajectory_ids=[],requested_direction_counts={},actual_contact_angle_bins={},moving_contact_count=0))
    return rows


def _reachable_segments(env,segments):
    """Exact polygon configuration-space component membership (fixed scene).

    A conservative 0.5px safety margin proves reachable staging where connected.
    Disconnection under inflation is explicitly unconfirmed for the true radius;
    only a geometrically closed hole is marked proven inaccessible from outside.
    """
    import numpy as np
    from shapely.geometry import Point,box,LineString
    geom=env.world_geometry()
    free=box(22.5,22.5,488.5,488.5).difference(geom.buffer(15.5,resolution=32))
    components=list(free.geoms) if hasattr(free,"geoms") else [free]
    components=[piece for piece in components if piece.area>1e-8]
    exterior=max(components,key=lambda piece:piece.area) if components else None
    c,s=math.cos(env.block.angle),math.sin(env.block.angle)
    rotation=np.asarray(((c,-s),(s,c)))
    rows=[]
    for segment in segments:
        point=np.asarray(env.block.local_to_world(segment["point"]))
        normal=rotation@np.asarray(segment["normal"])
        staging=point+normal*18
        center=Point(*staging)
        contact_center=point+normal*15.5
        contact=Point(*contact_center)
        contact_path=LineString((staging,contact_center))
        outside=exterior.buffer(1e-7) if exterior is not None else None
        if segment["ring"]:
            status="proven_closed_hole_from_exterior"
        elif outside is not None and outside.covers(center) and outside.covers(contact) and outside.covers(contact_path):
            status="reachable_exterior_component"
        elif outside is not None and outside.covers(center):
            status="normal_contact_path_blocked_reachability_unconfirmed"
        elif any(piece.covers(center) for piece in components):
            status="unconfirmed_disconnected_conservative_fixed_pose_component"
        else:
            status="normal_staging_blocked_reachability_unconfirmed"
        rows.append(dict(segment_id=segment["id"],status=status,staging=staging.tolist(),
            contact_center=contact_center.tolist(),normal_contact_path=[staging.tolist(),contact_center.tolist()],
            point=point.tolist(),normal=normal.tolist(),pusher_radius=15,conservative_clearance_margin=.5,
            path_evidence="staging/contact center and connecting segment belong to exterior-connected conservative free space" if status.startswith("reachable") else None))
    return rows


def _contact_coverage(letter,segments,points,trajectory_id,moving=False,direction="unknown",angles=None):
    from shapely.geometry import Point
    geom=shape_geometry(letter);rings=[geom.exterior,*geom.interiors]
    for point_index,xy in enumerate(points):
        point=Point(*xy)
        distances=[ring.distance(point) for ring in rings]
        ring_index=min(range(len(rings)),key=distances.__getitem__)
        if distances[ring_index]>2:continue
        arc=rings[ring_index].project(point)
        candidates=[row for row in segments if row["ring"]==ring_index and row["arc_start"]<=arc<=row["arc_end"]+1e-6]
        if not candidates:continue
        row=candidates[0];row["contact_count"]+=1
        if trajectory_id not in row["trajectory_ids"]:row["trajectory_ids"].append(trajectory_id)
        row["moving_contact_count"]+=int(moving)
        row["requested_direction_counts"][direction]=row["requested_direction_counts"].get(direction,0)+1
        if angles is not None:
            angle=float(angles[point_index]);bucket=str(int(((angle+math.pi)%(2*math.pi))/(math.pi/4)))
            row["actual_contact_angle_bins"][bucket]=row["actual_contact_angle_bins"].get(bucket,0)+1


def _write_coverage(output,coverage,reachability):
    import html
    from shapely.ops import substring
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summary={}
    for letter,rows in coverage.items():
        hit=sum(row["contact_count"]>0 for row in rows)
        reachable={row["segment_id"] for row in reachability[letter] if row["status"].startswith("reachable")}
        reached=sum(rows[i]["contact_count"]>0 for i in reachable)
        summary[letter]=dict(total_segments=len(rows),contacted_segments=hit,
            contacted_fraction=hit/max(1,len(rows)),canonical_reachable_segments=len(reachable),
            contacted_canonical_reachable_fraction=reached/max(1,len(reachable)),
            independent_contact_trajectories=len({t for row in rows for t in row["trajectory_ids"]}),
            actual_contact_points=sum(row["contact_count"] for row in rows),
            no_contact_segments=[row["id"] for row in rows if not row["contact_count"]],
            reachability_status_counts={status:sum(row["status"]==status for row in reachability[letter])
                for status in sorted({row["status"] for row in reachability[letter]})})
        lines=['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="620" viewBox="-100 -125 200 250">',
            '<rect x="-100" y="-125" width="200" height="250" fill="white"/>',
            f'<text x="-95" y="-110" font-size="7">{html.escape(letter)}: actual contact arc bins, 8px; green=hit red=unhit</text>']
        geom=shape_geometry(letter);rings=[geom.exterior,*geom.interiors]
        for row in rows:
            ring=rings[row["ring"]]
            points=" ".join(f"{x:.3f},{y:.3f}" for x,y in substring(ring,row["arc_start"],row["arc_end"]).coords)
            color="#14853c" if row["contact_count"] else "#c93434"
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"><title>bin {row["id"]}: {row["contact_count"]} points, {len(row["trajectory_ids"])} trajectories</title></polyline>')
        lines.append('</svg>');(output/f"{letter}.svg").write_text("\n".join(lines),encoding="utf8")
    write_json(output/"coverage.json",dict(summary=summary,segments=coverage,canonical_reachability=reachability,
        reachability_version="normal-contact-path-v2",
        evidence_scope="closed holes are inaccessible from exterior; normal staging/contact center and connecting segment checked within conservative fixed-pose exterior component; blocked paths remain unconfirmed"))
    return summary


def _validate_collection_config(config):
    if tuple(config.get("train_letters",TRAIN_LETTERS))!=TRAIN_LETTERS or tuple(config.get("unseen_letters",UNSEEN_LETTERS))!=UNSEEN_LETTERS:
        raise ValueError("Letter allocation differs from the approved fixed 18/8 split")
    if config.get("environment",{}).get("controller",CONTROLLER)!=CONTROLLER:raise ValueError("Collection requires the audited controller")
    expected=config.get("environment",{}).get("protocol_sha256")
    if expected and expected!=PROTOCOL_SHA256:raise ValueError("Environment protocol hash differs")
    if config.get("collection",{}).get("arc_segment_px",8)!=8:raise ValueError("Arc coverage protocol is fixed at 8px")
    if not rotational_split_audit(TRAIN_LETTERS,UNSEEN_LETTERS)["passed"]:raise ValueError("Cross-split rotational duplicate geometry")
    return collection_caps(config)


def collect_dataset(config,output,source_root,splits=("train","seen_val")):
    """Round-robin whole-scene collection; no replacement of failed transitions.

    Runs one CPU process deliberately: renderer/solver state is isolated, and the
    shared persistent budget is charged before each trajectory. Worker count is
    reported, never inferred from the requested config value.
    """
    import numpy as np
    output=Path(output);caps=_validate_collection_config(config)
    if (output/"manifest.json").exists():raise FileExistsError("Never overwrite collected scenes")
    probe_path=output.parent/"physics-probe/report.json"
    if not probe_path.exists() or not json.loads(probe_path.read_text()).get("physics_probe_gate_passed"):
        raise RuntimeError("Run and pass the physical probe before collection")
    probe=json.loads(probe_path.read_text())
    if probe.get("protocol_sha256")!=PROTOCOL_SHA256:raise ValueError("Physical probe tested another controller protocol")
    ledger=BudgetLedger(output.parent,caps)
    source_identity=capture_collection_source(output.parent,source_root)
    collection=config.get("collection",{});controls=int(collection.get("controls",64))
    if not 4<=controls<=256:raise ValueError("Controls per independent trajectory must be in [4,256]")
    requested=dict(train=int(collection.get("train_trajectories",24)),seen_val=int(collection.get("val_trajectories",6)),
        seen_test=int(collection.get("seen_test_trajectories",0)),unseen_test=int(collection.get("unseen_test_trajectories",0)))
    for split in splits:
        if split not in caps or requested[split]<1:raise ValueError("Split is not explicitly budgeted")
    output.mkdir(parents=True,exist_ok=True)
    shapes=TRAIN_LETTERS+(UNSEEN_LETTERS if "unseen_test" in splits else ())
    coverage_by_split={split:{letter:contour_segments(letter,8) for letter in
        (UNSEEN_LETTERS if split=="unseen_test" else TRAIN_LETTERS)} for split in splits}
    coverage={letter:contour_segments(letter,8) for letter in shapes};reachability={}
    environments={letter:make_env(source_root,letter,CONTROLLER) for letter in shapes}
    for letter,env in environments.items():
        env.reset_scene();reachability[letter]=_reachable_segments(env,coverage[letter])
    entries=[];start=time.monotonic();scene_hashes=set();failure=None
    locations=("center","left","right","top","bottom","left-top","right-top","left-bottom","right-bottom")
    try:
        for split in splits:
            letters=UNSEEN_LETTERS if split=="unseen_test" else TRAIN_LETTERS
            for repetition in range(requested[split]):
                for letter_index,letter in enumerate(letters):
                    remaining=caps[split]-ledger.value["charged_controls"][split]
                    if remaining<4:continue
                    count=min(controls,remaining)
                    seed=scene_seed(config.get("data_seed",20261028),split,repetition,letter_index)
                    rng=np.random.default_rng(seed);env=environments[letter]
                    current_coverage=coverage_by_split[split][letter]
                    location=locations[repetition%len(locations)]
                    chosen=None;reset_records=[]
                    for attempt in range(int(collection.get("max_reset_attempts",4))):
                        _wall_scene(env,location)
                        # Rotate at reset, then fit bounds back inside the requested walls.
                        angle=float(rng.uniform(-math.pi,math.pi));env.block.angle=angle
                        bounds=env.world_geometry().bounds
                        dx=(10-bounds[0]) if "left" in location else (501-bounds[2]) if "right" in location else max(0,10-bounds[0])-max(0,bounds[2]-501)
                        dy=(10-bounds[1]) if "top" in location else (501-bounds[3]) if "bottom" in location else max(0,10-bounds[1])-max(0,bounds[3]-501)
                        env.block.position=tuple(np.asarray(env.block.position)+(dx,dy))
                        candidates=[row for row in _reachable_segments(env,current_coverage) if row["status"].startswith("reachable")]
                        # Each split has its own counters; validation cannot steer training collection.
                        candidates.sort(key=lambda row:(current_coverage[row["segment_id"]]["contact_count"],
                            -len(current_coverage[row["segment_id"]]["features"]),float(rng.random())))
                        if candidates:
                            chosen=candidates[0];env.agent.position=tuple(chosen["staging"])
                            env.space.reindex_shapes_for_body(env.agent);env.space.reindex_shapes_for_body(env.block)
                        issues=[]
                        if chosen is None:issues.append("no_exterior_connected_contact_staging")
                        if np.max(env.penetrations())>.1:issues.append("initial_penetration")
                        state_hash=hashlib.sha256(env.state().tobytes()).hexdigest()
                        if state_hash in scene_hashes:issues.append("duplicate_scene")
                        reset_records.append(dict(attempt=attempt,issues=issues,initial_state=env.state().tolist(),
                            reachability=chosen))
                        if not issues:break
                        ledger.reject(dict(split=split,letter=letter,seed=seed,issues=issues,details=reset_records[-1]));chosen=None
                    if chosen is None:continue
                    scene_hashes.add(state_hash)
                    trajectory_id=f"{split}-{letter}-{repetition:04d}-{seed}"
                    path=output/"trajectories"/f"{trajectory_id}.npz"
                    reservation=ledger.reserve(split,count,f"native RGB {trajectory_id}; least-covered exterior arc staging")
                    local_normal=np.asarray(current_coverage[chosen["segment_id"]]["normal"])
                    action_rows=[]
                    def actions():
                        phase=0;done=0
                        while done<count:
                            c,s=math.cos(env.block.angle),math.sin(env.block.angle)
                            normal=np.asarray(((c,-s),(s,c)))@local_normal
                            tangent=np.asarray((-normal[1],normal[0]))
                            mode=(repetition+phase)%8 if phase else 0
                            direction=(-normal,(-normal+tangent)/math.sqrt(2),(-normal-tangent)/math.sqrt(2),
                                tangent,-tangent,normal,(normal+tangent)/math.sqrt(2),(normal-tangent)/math.sqrt(2))[mode]
                            distance=(24,96,256)[(repetition+phase)%3]
                            duration=(2,4,8,16)[(repetition+phase)%4]
                            target=np.clip(np.asarray(env.agent.position)+direction*distance,0,511)
                            for _ in range(min(duration,count-done)):
                                state=env.state()
                                action_rows.append(dict(mode=mode,distance_category=("near","mid","far")[(repetition+phase)%3],
                                    requested_distance=distance,duration=duration,
                                    moving=bool(np.linalg.norm(state[7:9])>.5 or abs(state[9])>.02)))
                                yield target
                                done+=1
                            phase+=1
                    stats=_run_actions(env,actions(),path,stop_invalid=True)
                    ledger.finish(reservation,stats["controls"],**stats)
                    with np.load(path,allow_pickle=False) as arrays:
                        for t in range(stats["controls"]):
                            lo,hi=arrays["contact_offsets"][t:t+2]
                            _contact_coverage(letter,current_coverage,arrays["contact_points"][lo:hi],trajectory_id,
                                action_rows[t]["moving"],str(action_rows[t]["mode"]),arrays["contact_angles"][lo:hi])
                        near_walls={name:0 for name in ("left","right","top","bottom")};corner_controls=0
                        from shapely import affinity
                        for recorded in arrays["states"]:
                            geom=affinity.rotate(env.local_geometry,float(recorded[4]),origin=(0,0),use_radians=True)
                            geom=affinity.translate(geom,*recorded[2:4]);xmin,ymin,xmax,ymax=geom.bounds
                            close=(xmin<23,xmax>488,ymin<23,ymax>488)
                            for name,flag in zip(near_walls,close):near_walls[name]+=int(flag)
                            corner_controls+=int((close[0] or close[1]) and (close[2] or close[3]))
                    entries.append(dict(trajectory_id=trajectory_id,shape=letter,split=split,seed=seed,
                        file=str(path.relative_to(output)).replace("\\","/"),sha256=file_sha256(path),
                        initial_scene_hash=state_hash,location=location,staging_segment=chosen["segment_id"],
                        reset_records=reset_records,action_metadata=action_rows[:stats["controls"]],
                        actual_wall_frame_counts=near_walls,actual_corner_frames=corner_controls,**stats))
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc))
    summary={split:_write_coverage(output/"coverage"/split,split_coverage,
        {letter:reachability[letter] for letter in split_coverage}) for split,split_coverage in coverage_by_split.items()}
    write_json(output/"coverage/summary.json",summary)
    manifest=dict(kind="native-rgb-compact-data-v1",controller=CONTROLLER,controller_protocol=CONTROL_PROTOCOL,
        protocol_sha256=PROTOCOL_SHA256,source_root=str(Path(source_root).resolve()),source_identity="pinned official physics SHA256",
        collector_source=source_identity,
        train_letters=list(TRAIN_LETTERS),unseen_letters=list(UNSEEN_LETTERS),entries=entries,
        config=config,state_columns=STATE_COLUMNS,render_size=96,control_hz=10,physics_hz=100,
        data_seed=int(config.get("data_seed",20261028)),requested_trajectories=requested,
        requested_workers=int(collection.get("workers",1)),actual_workers=1,
        actual_controls=sum(entry["controls"] for entry in entries),actual_trajectories=len(entries),
        collection_seconds=time.monotonic()-start,coverage_summary=summary,
        material_change="finite-mass force-limited dynamic pusher and solver settings; official geometry/object inertia/walls retained",
        privileged_sampling="exact collision-contour staging, reachability and contact coverage; not model inputs/targets",
        budget=ledger.value,failure=failure,split_geometry_audit=rotational_split_audit(TRAIN_LETTERS,UNSEEN_LETTERS))
    write_json(output/"manifest.json",manifest)
    audit=audit_dataset(output)
    if failure:raise RuntimeError(f"Collection failed after preserving diagnostics: {failure}")
    return dict(manifest=manifest,audit=audit)


def audit_dataset(data_root):
    """Recompute every file's identity and all transition gates, preserving failures."""
    import numpy as np
    root=Path(data_root);manifest=json.loads((root/"manifest.json").read_text())
    verify_collection_source(root,manifest)
    issues=[];invalid=0;rows=[];seen_ids=set();seen_scenes={};maxima=np.zeros(3);environments={}
    if manifest.get("protocol_sha256")!=PROTOCOL_SHA256:issues.append("controller_protocol_changed")
    if manifest.get("failure"):issues.append("collector_error")
    if not manifest.get("split_geometry_audit",{}).get("passed"):issues.append("geometry_split_leakage")
    for entry in manifest["entries"]:
        path=(root/entry["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path)!=entry["sha256"]:
            issues.append(f"file_hash:{entry['trajectory_id']}");continue
        if entry["trajectory_id"] in seen_ids:issues.append("duplicate_trajectory_id")
        seen_ids.add(entry["trajectory_id"])
        scene=entry["initial_scene_hash"]
        if scene in seen_scenes:issues.append("duplicate_initial_scene")
        seen_scenes[scene]=entry["split"]
        allowed=UNSEEN_LETTERS if entry["split"]=="unseen_test" else TRAIN_LETTERS
        if entry["shape"] not in allowed:issues.append("shape_split_leakage")
        with np.load(path,allow_pickle=False) as arrays:
            n=len(arrays["actions"]);state=arrays["states"];penetration=arrays["penetrations"]
            if state.shape!=(n+1,15) or penetration.shape!=(n+1,3):issues.append("state_alignment")
            if not np.isfinite(state).all() or not np.isfinite(penetration).all():issues.append("nonfinite_state")
            if hashlib.sha256(state[0].tobytes()).hexdigest()!=scene:issues.append("initial_scene_identity")
            if arrays["actions"].shape!=(n,2) or not np.isfinite(arrays["actions"]).all() or (arrays["actions"]<0).any() or (arrays["actions"]>511).any():issues.append("invalid_absolute_action")
            computed=np.max(penetration[1:],axis=1)<=2
            if not np.array_equal(computed,arrays["valid_transition"]):issues.append("invalid_flag_mismatch")
            bad=int((~computed).sum());invalid+=bad
            maxima=np.maximum(maxima,np.max(penetration,axis=0))
            if entry["shape"] not in environments:
                environment=make_env(manifest["source_root"],entry["shape"],manifest["controller"])
                environment.reset_scene();environments[entry["shape"]]=environment
            environment=environments[entry["shape"]]
            # Independent endpoint geometry is checked without extra physics controls.
            for frame,recorded in enumerate(state):
                environment.set_render_state(recorded)
                geometric=environment.penetrations()
                if (geometric>penetration[frame]+1e-3).any():issues.append("underreported_geometric_penetration")
                if geometric.max()>2:issues.append("invalid_geometric_endpoint")
            if arrays["contact_offsets"].shape!=(n+1,) or arrays["contact_offsets"][-1]!=len(arrays["contact_points"]):issues.append("contact_alignment")
            rows.append(dict(trajectory_id=entry["trajectory_id"],shape=entry["shape"],split=entry["split"],controls=n,invalid_controls=bad))
    if invalid:issues.append("invalid_physical_transitions_retained")
    if maxima.max()>2:issues.append("penetration_threshold_exceeded")
    if not manifest["entries"]:issues.append("empty_dataset")
    for split in ("train","seen_val"):
        if split in manifest["config"].get("collection",{}).get("required_splits",("train","seen_val")):
            present={entry["shape"] for entry in manifest["entries"] if entry["split"]==split}
            if present!=set(TRAIN_LETTERS):issues.append(f"missing_letters:{split}")
    report=dict(kind="native-rgb-physical-audit-v1",physics_collection_gate_passed=not issues,
        dataset_manifest_sha256=file_sha256(root/"manifest.json"),dataset_npz_sha256_verified=not any(issue.startswith("file_hash") for issue in issues),
        issues=sorted(set(issues)),invalid_controls=invalid,max_penetration=maxima.tolist(),trajectories=rows,
        actual_controls=sum(row["controls"] for row in rows),invalid_transitions_policy="retained on disk; excluded from dataset windows; any invalid control fails the whole collection gate")
    write_json(root/"physics-audit.json",report)
    return report


def verify_dataset(root,require_gate=True):
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest.get("kind")=="native-rgb-supervised-reference-data-v1":
        from .native_rgb_reference import verify_reference_dataset
        return verify_reference_dataset(root,manifest,require_gate=require_gate)
    verify_collection_source(root,manifest)
    audit=json.loads((root/"physics-audit.json").read_text())
    if audit.get("dataset_manifest_sha256")!=file_sha256(root/"manifest.json"):
        raise ValueError("Physical audit does not identify this dataset manifest")
    for entry in manifest["entries"]:
        path=(root/entry["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path)!=entry["sha256"]:
            raise ValueError("Dataset file/hash changed")
    if not audit.get("dataset_npz_sha256_verified"):raise ValueError("Dataset hashes were not verified")
    if require_gate and not audit.get("physics_collection_gate_passed"):
        raise RuntimeError("Physical collection gate failed; training is prohibited")
    return manifest,audit


class NativeRGBDataset:
    """Compact states on disk, uint8 RGB LRU in memory; float windows on demand."""
    def __init__(self,root,split="train",cache_bytes=32*1024**2,require_gate=True,source_root=None,physics_source_root=None):
        import numpy as np
        self.root=Path(root)
        self.split=split
        self.manifest,self.audit=verify_dataset(root,require_gate=require_gate)
        if source_root is not None and physics_source_root is not None and Path(source_root).resolve()!=Path(physics_source_root).resolve():
            raise ValueError("source_root is the legacy physics alias; it must match physics_source_root")
        self.source_root=physics_source_root or source_root or self.manifest.get("physics_source_root") or self.manifest["source_root"]
        self.physics_source_root=self.source_root
        if self.manifest.get("kind")=="native-rgb-supervised-reference-data-v1":
            if split not in ("train","seen_val"):raise ValueError("Reference view exposes only formal TRAIN and seen_val")
            expected=Path(self.manifest["physics_source_root"]).resolve()
            if Path(self.source_root).resolve()!=expected or not (expected/"diffusion_policy/env/pusht/pusht_env.py").is_file():
                raise ValueError("Explicit official physics source required; GPC source is not a renderer")
            if cache_bytes>128*1024**2:raise ValueError("Reference-view uint8 image cache is capped at128MiB")
        self.entries=[entry for entry in self.manifest["entries"] if entry["split"]==split]
        self.windows=[];self.h4_windows=[]
        self.cache_bytes=int(cache_bytes)
        if self.cache_bytes<0:raise ValueError("Negative image cache")
        self._images=OrderedDict();self._image_bytes=0;self._environments={};self._trajectory_cache=None
        for i,entry in enumerate(self.entries):
            with np.load(self._trajectory_path(entry),allow_pickle=False) as arrays:
                valid=arrays["valid_transition"]
                for anchor in range(3,len(valid)):
                    if valid[anchor-3:anchor+1].all():self.windows.append((i,anchor))
                for anchor in range(3,len(valid)-3):
                    if valid[anchor-3:anchor+4].all():self.h4_windows.append((i,anchor))
    def _trajectory_path(self,entry):
        if self.manifest.get("kind")=="native-rgb-supervised-reference-data-v1":
            from .native_rgb_reference import resolve_reference_entry
            return resolve_reference_entry(self.root,self.manifest,entry)
        return self.root/entry["file"]
    def __len__(self):return len(self.windows)
    def trajectory(self,index):
        import numpy as np
        entry=self.entries[index]
        if self._trajectory_cache is None or self._trajectory_cache[0]!=index:
            with np.load(self._trajectory_path(entry),allow_pickle=False) as source:
                arrays={key:source[key] for key in source.files}
            self._trajectory_cache=(index,arrays)
        return entry,self._trajectory_cache[1]
    def _env(self,entry):
        key=entry["shape"]
        if key not in self._environments:
            env=make_env(self.source_root,key,self.manifest["controller"])
            env.reset_scene()
            self._environments[key]=env
        return self._environments[key]
    def render_layers(self,entry,state,size=96):
        env=self._env(entry);env.set_render_state(state)
        return env.render_layers(size)
    def render_state(self,entry,state):return self.render_layers(entry,state)["rgb"]
    def _image(self,i,frame,entry,state):
        key=(i,frame)
        if key in self._images:
            value=self._images.pop(key);self._images[key]=value;return value
        value=self.render_state(entry,state)
        while self._images and self._image_bytes+value.nbytes>self.cache_bytes:
            _,old=self._images.popitem(last=False);self._image_bytes-=old.nbytes
        if value.nbytes<=self.cache_bytes:
            self._images[key]=value;self._image_bytes+=value.nbytes
        return value
    def __getitem__(self,index):
        import numpy as np
        i,anchor=self.windows[index];entry,arrays=self.trajectory(i)
        image=np.stack([self._image(i,t,entry,arrays["states"][t]) for t in range(anchor-3,anchor+2)]).astype(np.float32)/255
        action=arrays["actions"][anchor-3:anchor+1].astype(np.float32)/511*2-1
        return dict(image=image,action=action,trajectory_id=entry["trajectory_id"],shape=entry["shape"],anchor=anchor)
