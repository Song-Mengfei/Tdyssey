"""Stage-independent, offline motion statistics for recorded native RGB controls.

Nothing here steps physics, runs a model, filters static data to pass a gate, or
builds a training index. Geometry/state are privileged sampling/audit metadata.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from .envs.official import official_shape
from .native_rgb_data import TRAIN_LETTERS

SPATIAL_CLASSES=("center","left","right","top","bottom","left-top","right-top","left-bottom","right-bottom")
WALLS=("left","right","top","bottom")
MOTION_PROTOCOL=dict(version="native-rgb-recorded-motion.v1",control_dt=.1,
    moving_translation_world_px_gte=2.,moving_rotation_deg_gte=1.,
    near_static_translation_world_px_lte=.5,near_static_rotation_rad_lte=.005,
    candidate_history_controls=4,candidate_anchor_min=3,penetration_limit_world_px=2.,
    physics_substeps_per_control=10,image_pixels_per_world_pixel=96/512,
    candidate_moving_fraction_gte=.40,candidate_near_static_fraction_lte=.45,
    wall_inner_left_top=7.,wall_inner_right_bottom=504.,actual_spatial_near_wall_limit_px=64.)
STRATA=("moving","intermediate","near_static_contact","near_static_no_contact")
STAGE1_LETTERS=("T","L","O","C","X","B")


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):digest.update(block)
    return digest.hexdigest()


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def _finite_or_none(value):
    value=float(value)
    return value if math.isfinite(value) else None


def area_centroids(letter,states):
    """Area centroid from body origin; T's legacy physical CoG is not used."""
    import numpy as np
    states=np.asarray(states,dtype=np.float64)
    if states.ndim!=2 or states.shape[1]!=15:raise ValueError("Expected state[N,15], angle at column4")
    local=np.asarray(official_shape(letter).centroid,dtype=np.float64)
    c,s=np.cos(states[:,4]),np.sin(states[:,4])
    return states[:,2:4]+np.column_stack((c*local[0]-s*local[1],s*local[0]+c*local[1]))


def geometric_wall_gaps(letter,states):
    """Signed object-bounding gaps to the actual inner walls, in world pixels."""
    import numpy as np
    states=np.asarray(states,dtype=np.float64)
    vertices=np.asarray([v for rect in official_shape(letter).components for v in rect.vertices])
    c,s=np.cos(states[:,4,None]),np.sin(states[:,4,None])
    x=c*vertices[:,0]-s*vertices[:,1]+states[:,2,None]
    y=s*vertices[:,0]+c*vertices[:,1]+states[:,3,None]
    return np.column_stack((x.min(1)-7,504-x.max(1),y.min(1)-7,504-y.max(1)))


def gap_band(value):
    if value is None or not math.isfinite(float(value)):return "unknown"
    if value<0:return "negative"
    if value<=8:return "0-8"
    if value<=32:return "8-32"
    if value<=64:return "32-64"
    return ">64"


def actual_spatial_class(gaps):
    import numpy as np
    gaps=np.asarray(gaps,dtype=float)
    if gaps.shape!=(4,) or not np.isfinite(gaps).all():return "unknown"
    horizontal=("left" if gaps[0]<=gaps[1] else "right") if min(gaps[:2])<=64 else None
    vertical=("top" if gaps[2]<=gaps[3] else "bottom") if min(gaps[2:])<=64 else None
    return "-".join(value for value in (horizontal,vertical) if value) or "center"


def motion_components(letter,states,angle_convention="pymunk_unwrapped"):
    import numpy as np
    states=np.asarray(states,dtype=np.float64)
    centers=area_centroids(letter,states)
    translation=np.linalg.norm(np.diff(centers,axis=0),axis=1)
    raw=np.diff(states[:,4]);wrapped=np.arctan2(np.sin(raw),np.cos(raw))
    if angle_convention not in ("pymunk_unwrapped","wrapped"):
        raise ValueError("Declare pymunk_unwrapped or wrapped angle convention")
    rotation=np.abs(raw if angle_convention=="pymunk_unwrapped" else wrapped)
    shape=official_shape(letter);cx,cy=shape.centroid
    radius=math.sqrt(sum(rect.area*((rect.x1-rect.x0)**2/12+(rect.y1-rect.y0)**2/12
        +(rect.center[0]-cx)**2+(rect.center[1]-cy)**2) for rect in shape.components)/shape.area)
    classes=[]
    for distance,angle in zip(translation,rotation):
        if not math.isfinite(distance) or not math.isfinite(angle):name="unclassified_nonfinite"
        elif distance>=2 or angle>=math.radians(1):name="moving"
        elif distance<=.5 and angle<=.005:name="near_static"
        else:name="intermediate"
        classes.append(name)
    return dict(centers=centers,translation_world_px=translation,raw_dtheta_rad=raw,
        wrapped_dtheta_rad=wrapped,rotation_abs_rad=rotation,rotation_arc_world_px=rotation*radius,
        motion_class=classes,shape_rms_radius_world_px=radius)


def transition_rows(entry,arrays,steps=None,*,require_substeps=True,angle_convention="pymunk_unwrapped"):
    """Return one row for every recorded real transition, including invalid ones."""
    import numpy as np
    letter=entry["shape"]
    if letter not in TRAIN_LETTERS:raise ValueError("Only approved seen letters may be inspected")
    actions=np.asarray(arrays["actions"],dtype=np.float64);n=len(actions)
    states=np.asarray(arrays["states"],dtype=np.float64)
    if states.shape!=(n+1,15) or actions.shape!=(n,2):raise ValueError("State/action alignment differs")
    contact=np.asarray(arrays["contact"],dtype=np.float64)
    declared=np.asarray(arrays["valid_transition"],dtype=bool)
    penetration=np.asarray(arrays["penetrations"],dtype=np.float64)
    if contact.shape!=(n,) or declared.shape!=(n,) or penetration.shape!=(n+1,3):
        raise ValueError("Contact/validity/penetration alignment differs")
    motion=motion_components(letter,states,angle_convention)
    gaps=geometric_wall_gaps(letter,states)
    finite_state=np.isfinite(states).all(1)
    finite_contact=np.isfinite(contact)&(contact>=0)
    finite_action=np.isfinite(actions).all(1)&(actions>=0).all(1)&(actions<=511).all(1)
    finite_pen=np.isfinite(penetration).all(1)
    maximum=np.max(penetration[1:],axis=1)
    substeps=arrays.get("substep_penetrations")
    substep_evidence=substeps is not None
    if substep_evidence:
        substeps=np.asarray(substeps,dtype=np.float64)
        if substeps.shape!=(n,10,3):raise ValueError("Expected all10 substep penetration triples per control")
        substep_finite=np.isfinite(substeps).all((1,2))
        maximum=np.maximum(maximum,np.max(substeps,axis=(1,2)))
    else:substep_finite=np.full(n,not require_substeps)
    endpoint_wall=np.maximum(0,-gaps.min(1))
    logged_maximum=maximum.copy()
    maximum=np.maximum(maximum,np.maximum(endpoint_wall[:-1],endpoint_wall[1:]))
    independent=(finite_state[:-1]&finite_state[1:]&finite_contact&finite_action&finite_pen[:-1]&finite_pen[1:]
        &substep_finite&(maximum<=2)&(endpoint_wall[:-1]<=2)&(endpoint_wall[1:]<=2))
    valid=declared&independent
    steps=list(steps or [])
    identifier=entry.get("trajectory_id",entry.get("file","unknown"))
    rows=[]
    for t in range(n):
        meta=steps[t] if t<len(steps) else {}
        before,after=gaps[t],gaps[t+1]
        before_min,after_min=float(before.min()),float(after.min())
        target_distance=float(np.linalg.norm(actions[t]-states[t,:2]))
        is_contact=bool(contact[t]>0) if finite_contact[t] else None
        name=motion["motion_class"][t]
        stratum=name if name in ("moving","intermediate") else (
            "near_static_contact" if name=="near_static" and is_contact is True else
            "near_static_no_contact" if name=="near_static" and is_contact is False else "unclassified")
        intent=str(meta.get("intent","unknown"))
        declared_wall_intent=meta.get("toward_wall") is True or "toward_wall" in intent or "into_wall" in intent
        nearest=int(np.argmin(before)) if np.isfinite(before).all() else None
        normals=((-1,0),(1,0),(0,-1),(0,1))
        target_delta=actions[t]-states[t,:2]
        points_toward=(nearest is not None and target_distance>1e-8
            and float(target_delta@np.asarray(normals[nearest]))/target_distance>=.5)
        likely_blocked=bool(name=="near_static" and is_contact and before_min<=2 and
                            (declared_wall_intent or points_toward))
        gap_change=float(after[nearest]-before[nearest]) if nearest is not None else math.nan
        if likely_blocked:boundary_behavior="likely_wall_blocked_proxy"
        elif name=="near_static" and is_contact:boundary_behavior="near_static_contact"
        elif min(before_min,after_min)>64:boundary_behavior="interior"
        elif before_min<=64 and gap_change>.5:boundary_behavior="release_from_boundary"
        elif min(before_min,after_min)<=64 and gap_change<-.5:boundary_behavior="approaching_wall"
        elif before_min<=64 and name=="moving" and abs(gap_change)<=.5:boundary_behavior="along_wall_moving"
        else:boundary_behavior="boundary_other"
        h1_valid=bool(t>=3 and valid[t-3:t+1].all())
        requested_location=entry.get("location",entry.get("spatial_class","unknown"))
        corner_walls=[wall for wall in WALLS if wall in requested_location.split("-")]
        corner_before=max(before[WALLS.index(wall)] for wall in corner_walls) if len(corner_walls)==2 else None
        corner_after=max(after[WALLS.index(wall)] for wall in corner_walls) if len(corner_walls)==2 else None
        row=dict(row_id=f"{identifier}:{t}",trajectory_id=identifier,letter=letter,
            split=entry.get("split","stage1"),transition=t,source_file=entry.get("file"),
            requested_spatial_class=requested_location,
            requested_gap_band=entry.get("gap_band_requested","unknown"),
            requested_gap_world_px=entry.get("gap_requested_world_px"),
            actual_spatial_class_before=actual_spatial_class(before),actual_spatial_class_after=actual_spatial_class(after),
            gap_band_before=gap_band(before_min),gap_band_after=gap_band(after_min),
            min_wall_gap_before=_finite_or_none(before_min),min_wall_gap_after=_finite_or_none(after_min),
            requested_corner_max_gap_before=None if corner_before is None else _finite_or_none(corner_before),
            requested_corner_max_gap_after=None if corner_after is None else _finite_or_none(corner_after),
            requested_corner_gap_band_before=gap_band(corner_before),requested_corner_gap_band_after=gap_band(corner_after),
            phase=str(meta.get("phase","unknown")),intent=intent,
            approach_kind=str(meta.get("approach_kind","unknown")),target_bin=str(meta.get("target_bin","unknown")),
            target_distance_class=str(meta.get("target_distance_class","unknown")),
            target_distance_world_px=_finite_or_none(target_distance),
            translation_world_px=_finite_or_none(motion["translation_world_px"][t]),
            translation_image96_px=_finite_or_none(motion["translation_world_px"][t]*96/512),
            rotation_abs_rad=_finite_or_none(motion["rotation_abs_rad"][t]),
            rotation_abs_deg=_finite_or_none(math.degrees(motion["rotation_abs_rad"][t])),
            raw_dtheta_rad=_finite_or_none(motion["raw_dtheta_rad"][t]),
            wrapped_dtheta_rad=_finite_or_none(motion["wrapped_dtheta_rad"][t]),
            rotation_arc_world_px=_finite_or_none(motion["rotation_arc_world_px"][t]),
            rotation_arc_image96_px=_finite_or_none(motion["rotation_arc_world_px"][t]*96/512),
            large_raw_angular_step=bool(abs(motion["raw_dtheta_rad"][t])>math.pi),
            motion_class=name,index_stratum=stratum,contact=is_contact,
            contact_count=_finite_or_none(contact[t]),near_static_contact=bool(name=="near_static" and is_contact),
            likely_wall_blocked_proxy=likely_blocked,target_points_toward_nearest_wall=bool(points_toward),
            boundary_behavior=boundary_behavior,initial_nearest_wall=None if nearest is None else WALLS[nearest],
            initial_nearest_wall_gap_change=_finite_or_none(gap_change),
            declared_valid=bool(declared[t]),independent_valid=bool(independent[t]),valid_transition=bool(valid[t]),
            validity_flag_mismatch=bool(declared[t]!=independent[t]),
            substep_evidence_present=bool(substep_evidence),max_penetration_world_px=_finite_or_none(maximum[t]),
            geometric_wall_penetration_underreported=bool(max(endpoint_wall[t:t+2])>logged_maximum[t]+1e-3),
            h1_anchor_eligible=t>=3,valid_h1_candidate=h1_valid,
            invalid_history_excluded=bool(t>=3 and not h1_valid),metadata_present=t<len(steps))
        for index,wall in enumerate(WALLS):
            row[f"{wall}_gap_before"]=_finite_or_none(before[index])
            row[f"{wall}_gap_after"]=_finite_or_none(after[index])
        rows.append(row)
    return rows


def continuous_distribution(values):
    import numpy as np
    values=list(values);available=[float(value) for value in values if value is not None and math.isfinite(float(value))]
    probabilities=(0,10,25,50,75,90,95,99,100)
    return dict(total=len(values),finite=len(available),missing_or_nonfinite=len(values)-len(available),
        zeros=sum(value==0 for value in available),
        quantiles={str(q):float(np.percentile(available,q)) if available else None for q in probabilities})


def summarize_rows(rows):
    rows=list(rows);n=len(rows);counts=Counter(row["motion_class"] for row in rows)
    return dict(total=n,valid=sum(row["valid_transition"] for row in rows),invalid=sum(not row["valid_transition"] for row in rows),
        class_counts=dict(counts),class_fractions={name:counts[name]/n if n else None
            for name in ("moving","intermediate","near_static","unclassified_nonfinite")},
        contact_by_motion={contact:{name:sum(row["motion_class"]==name and
            ("unknown" if row["contact"] is None else "contact" if row["contact"] else "no_contact")==contact for row in rows)
            for name in ("moving","intermediate","near_static","unclassified_nonfinite")}
            for contact in ("contact","no_contact","unknown")},
        continuous={name:continuous_distribution(row.get(name) for row in rows) for name in
            ("translation_world_px","translation_image96_px","rotation_abs_rad","rotation_abs_deg",
             "raw_dtheta_rad","wrapped_dtheta_rad","rotation_arc_world_px","rotation_arc_image96_px",
             "min_wall_gap_before","min_wall_gap_after","requested_corner_max_gap_before","requested_corner_max_gap_after",
             "initial_nearest_wall_gap_change","target_distance_world_px","max_penetration_world_px")},
        near_static_contact=sum(row.get("near_static_contact",False) for row in rows),
        likely_wall_blocked_proxy=sum(row.get("likely_wall_blocked_proxy",False) for row in rows),
        metadata_missing=sum(not row.get("metadata_present",False) for row in rows),
        substep_evidence_missing=sum(not row.get("substep_evidence_present",False) for row in rows),
        large_raw_angular_steps=sum(row.get("large_raw_angular_step",False) for row in rows))


def proposed_index_feasibility(rows,requested_count=None):
    """Plan only. Exact70/10/10/10 requires every stratum; no samples are made."""
    rows=list(rows);counts=Counter(row["index_stratum"] for row in rows)
    units=min(counts["moving"]//7,*(counts[name] for name in STRATA[1:]))
    natural_count=len(rows)
    proposed=natural_count//10*10 if requested_count is None else requested_count
    if type(proposed) is not int or proposed<0:raise ValueError("A finite nonnegative integer index count is required")
    exact_count=proposed>0 and proposed%10==0
    needed={name:(proposed//10*(7 if name=="moving" else 1)) if exact_count else None for name in STRATA}
    missing=[name for name in STRATA if counts[name]==0]
    return dict(planned_only=True,raw_distribution_unchanged=True,natural_valid_candidates=natural_count,
        target_ratios=dict(zip(STRATA,(.7,.1,.1,.1))),unique_support={name:counts[name] for name in STRATA},
        missing_strata=missing,proposed_index_count=proposed,exact_ratio_integer_count=exact_count,
        required_unique_counts=needed,max_exact_without_replacement_count=10*units,
        feasible_without_replacement=bool(exact_count and all(needed[name]<=counts[name] for name in STRATA)),
        feasible_with_replacement=bool(exact_count and not missing),
        required_draws_per_unique_support={name:needed[name]/counts[name] if exact_count and counts[name] else None for name in STRATA})


def _groups(rows,key):
    groups=defaultdict(list)
    if "spatial_class" in key:
        for name in SPATIAL_CLASSES:groups[name]=[]
    if key in ("gap_band_before","gap_band_after","requested_corner_gap_band_before","requested_corner_gap_band_after"):
        for name in ("negative","0-8","8-32","32-64",">64","unknown"):groups[name]=[]
    for row in rows:groups[str(row.get(key,"unknown"))].append(row)
    return {name:summarize_rows(selected) for name,selected in sorted(groups.items())}


def summarize_records(records,*,output=None,require_substeps=True,angle_convention="pymunk_unwrapped",expected_letters=STAGE1_LETTERS):
    """All real rows and all raw valid H1 anchors; static rows remain untouched."""
    expected_letters=tuple(expected_letters)
    if not expected_letters or any(letter not in TRAIN_LETTERS for letter in expected_letters):
        raise ValueError("Expected letters must be nonempty approved seen letters")
    all_rows=[];trajectories=[]
    for record in records:
        entry,arrays=record["entry"],record["arrays"]
        rows=transition_rows(entry,arrays,record.get("steps"),require_substeps=require_substeps,
                             angle_convention=angle_convention)
        all_rows.extend(rows)
        gaps=geometric_wall_gaps(entry["shape"],arrays["states"])
        trajectories.append(dict(trajectory_id=entry.get("trajectory_id"),letter=entry["shape"],controls=len(rows),
            source_file=entry.get("file"),requested_location=entry.get("location"),
            initial_wall_gaps=dict(zip(WALLS,map(_finite_or_none,gaps[0]))),
            initial_actual_spatial_class=actual_spatial_class(gaps[0]),
            initial_min_wall_gap=_finite_or_none(gaps[0].min()),initial_min_gap_band=gap_band(gaps[0].min()),
            requested_gap_band=entry.get("gap_band_requested"),requested_gap_world_px=entry.get("gap_requested_world_px"),
            all_real=summarize_rows(rows),valid_h1=summarize_rows([row for row in rows if row["valid_h1_candidate"]])))
    universe=[row for row in all_rows if row["h1_anchor_eligible"]]
    candidates=[row for row in universe if row["valid_h1_candidate"]]
    present=sorted({entry["letter"] for entry in trajectories})
    letters=sorted(set(present)|set(expected_letters))
    gates={}
    for letter in letters:
        raw=[row for row in all_rows if row["letter"]==letter]
        selected=[row for row in candidates if row["letter"]==letter]
        summary=summarize_rows(selected);moving=summary["class_fractions"]["moving"];static=summary["class_fractions"]["near_static"]
        gates[letter]=dict(all_real_controls=len(raw),invalid_real_controls=sum(not row["valid_transition"] for row in raw),
            candidate_universe=sum(row["letter"]==letter for row in universe),valid_h1_candidates=len(selected),
            moving_fraction=moving,near_static_fraction=static,
            moving_target_passed=bool(selected and moving>=.4),near_static_target_passed=bool(selected and static<=.45),
            passed=bool(selected and moving>=.4 and static<=.45 and all(row["valid_transition"] for row in raw)),
            index_feasibility=proposed_index_feasibility(selected))
    result=dict(kind="native-rgb-recorded-motion-statistics.v1",protocol=dict(MOTION_PROTOCOL),
        statistics_source_sha256=_sha(__file__),angle_convention=angle_convention,
        all_real_transitions=summarize_rows(all_rows),raw_h1_anchor_universe=summarize_rows(universe),
        valid_h1_candidates=summarize_rows(candidates),invalid_history_windows=len(universe)-len(candidates),
        per_letter={letter:dict(all_real=summarize_rows([r for r in all_rows if r["letter"]==letter]),
            valid_h1=summarize_rows([r for r in candidates if r["letter"]==letter])) for letter in letters},
        per_letter_gate=gates,all_present_letters_raw_gate_passed=bool(present and all(gates[letter]["passed"] for letter in present)),
        expected_letters=list(expected_letters),present_letters=present,missing_letters=sorted(set(expected_letters)-set(present)),
        all_expected_letters_raw_gate_passed=bool(gates and not(set(expected_letters)-set(present)) and all(gates[letter]["passed"] for letter in expected_letters)),
        proposed_index_feasibility=proposed_index_feasibility(candidates),trajectories=trajectories,
        groups={key:dict(all_real=_groups(all_rows,key),valid_h1=_groups(candidates,key)) for key in
            ("requested_spatial_class","actual_spatial_class_before","actual_spatial_class_after","gap_band_before",
             "gap_band_after","requested_corner_gap_band_before","requested_corner_gap_band_after","requested_gap_band",
             "boundary_behavior","phase","intent","approach_kind","target_bin","target_distance_class")},
        initial_scene_distribution={location:dict(trajectories=sum(row["requested_location"]==location for row in trajectories),
            actual_class_counts=dict(Counter(row["initial_actual_spatial_class"] for row in trajectories if row["requested_location"]==location)),
            actual_min_gap_bands=dict(Counter(row["initial_min_gap_band"] for row in trajectories if row["requested_location"]==location)))
            for location in SPATIAL_CLASSES},
        continuous_image_scale_note="2worldpx=.375image96px; .5worldpx=.09375image96px; threshold is privileged motion, not visual identifiability",
        contact_scope="Only agent-object arbiter observations; near_static_contact and likely_wall_blocked_proxy are not wall-contact counts",
        boundary_behavior_scope="Offline signed gap change to initially nearest wall; >.5px release, <-.5px approach, moving within .5px along-wall; contact/static proxy takes precedence; no wall arbiter evidence",
        gap_band_endpoints="negative separate;0<=gap<=8;8<gap<=32;32<gap<=64;gap>64; all four signed gaps retained",
        no_static_samples_deleted_for_gate=True,physics_steps_executed=0,model_calls=0)
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        if (output/"motion-statistics.json").exists():raise FileExistsError("Preserve existing motion statistics")
        fields=sorted({key for row in all_rows for key in row})
        with (output/"real-transition-statistics.csv").open("x",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(all_rows)
        _write(output/"motion-statistics.json",result)
    result["rows"]=all_rows
    return result


def read_records(data_root,manifest=None):
    import numpy as np
    root=Path(data_root).resolve()
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8")) if manifest is None else manifest
    records=[]
    for entry in manifest["entries"]:
        if entry["shape"] not in TRAIN_LETTERS:raise ValueError("Unseen records remain sealed")
        path=(root/entry["file"]).resolve()
        if not path.is_relative_to(root):raise ValueError("Trajectory path escapes data root")
        if entry.get("sha256") and _sha(path)!=entry["sha256"]:raise ValueError("Trajectory hash changed")
        with np.load(path,allow_pickle=False) as source:arrays={key:source[key] for key in source.files}
        metadata={}
        if entry.get("metadata_file"):
            meta=(root/entry["metadata_file"]).resolve()
            if not meta.is_relative_to(root):raise ValueError("Metadata path escapes data root")
            metadata=json.loads(meta.read_text(encoding="utf-8"))
        records.append(dict(entry=entry,arrays=arrays,steps=metadata.get("steps",[]),metadata=metadata))
    return records


def summarize_dataset(data_root,*,manifest=None,output=None,require_substeps=True,
                      angle_convention="pymunk_unwrapped",expected_letters=STAGE1_LETTERS):
    return summarize_records(read_records(data_root,manifest),output=output,require_substeps=require_substeps,
                             angle_convention=angle_convention,expected_letters=expected_letters)


def render_motion_contact_sheet(records,source_root,output,*,max_cases=8):
    """At most8 rows of five truly consecutive saved frames; never call step()."""
    import numpy as np
    from PIL import Image,ImageDraw
    from .native_rgb_env import make_env,CONTROLLER
    if type(max_cases) is not int or not 1<=max_cases<=12:raise ValueError("Choose1..12 bounded contact-sheet cases")
    lookup={record["entry"].get("trajectory_id",record["entry"].get("file","unknown")):record for record in records}
    rows=[row for record in records for row in transition_rows(record["entry"],record["arrays"],record.get("steps"))
          if row["valid_h1_candidate"]]
    pools=dict(moving=[row for row in rows if row["motion_class"]=="moving"],
               near_static_contact=[row for row in rows if row["near_static_contact"]])
    selected=[];seen=set()
    for label,pool in pools.items():
        pool.sort(key=lambda row:(-int(row["likely_wall_blocked_proxy"]) if label=="near_static_contact" else 0,
                                  -float(row["translation_world_px"] or 0),row["row_id"]))
        for row in pool:
            key=(row["letter"],row["requested_spatial_class"])
            if (label,key) in seen:continue
            selected.append((label,row));seen.add((label,key))
            if sum(name==label for name,_ in selected)>=max_cases//2:break
    selected=selected[:max_cases]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/"motion-contact-sheet.json").exists():raise FileExistsError("Preserve existing contact sheet")
    environments={};evidence=[]
    canvas=Image.new("RGB",(5*128,max(1,len(selected))*160),"white");draw=ImageDraw.Draw(canvas)
    for index,(label,row) in enumerate(selected):
        record=lookup[row["trajectory_id"]];letter=row["letter"]
        if letter not in environments:
            env=make_env(source_root,letter,CONTROLLER);env.reset_scene();environments[letter]=env
        env=environments[letter];anchor=row["transition"];frames=list(range(anchor-3,anchor+2))
        draw.text((4,index*160),f"{label} {letter} {row['requested_spatial_class']} t={anchor} dx={row['translation_world_px']:.2f}px a={row['rotation_abs_deg']:.2f}deg",fill="black")
        for column,frame in enumerate(frames):
            env.set_render_state(record["arrays"]["states"][frame]);image=env.render_clean()
            pixels=np.moveaxis(np.asarray(image),0,-1)
            canvas.paste(Image.fromarray(pixels).resize((128,128),Image.Resampling.NEAREST),(column*128,index*160+28))
            draw.text((column*128+4,index*160+14),f"frame {frame}",fill="black")
        evidence.append(dict(category=label,trajectory_id=row["trajectory_id"],letter=letter,anchor=anchor,frames=frames,
            translation_world_px=row["translation_world_px"],rotation_abs_deg=row["rotation_abs_deg"],
            likely_wall_blocked_proxy=row["likely_wall_blocked_proxy"],raw_rgb_unmodified=True))
    if selected:canvas.save(output/"motion-vs-near-static-contact.png")
    result=dict(cases=evidence,available_categories={key:len(value) for key,value in pools.items()},
        missing_categories=[key for key,value in pools.items() if not value],
        physics_steps_executed=0,model_calls=0,render_only=True)
    _write(output/"motion-contact-sheet.json",result)
    return result
