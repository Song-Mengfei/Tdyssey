"""Saved-array and geometry audit only. No model/physics/renderer construction."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'v1'
PILOT=OUT.parent.parent
ROOT=PILOT.parent.parent
started=time.perf_counter()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
original=json.loads((BASE/'summary.json').read_text(encoding='utf-8'))
def geometry(identity):
    shapes=identity['physics']['shapes']
    parts=[]
    for item in shapes:
        if item['role']!='object':continue
        g=item['geometry']
        assert g['kind']=='poly' and g['radius']==0
        parts.append(Polygon(g['vertices']))
    return unary_union(parts)
def measures(geom,states):
    """Exact rigid union geometry, vectorized state transform; no simulation."""
    states=np.asarray(states,float)
    if not len(states):return None
    c,s=np.cos(states[:,4]),np.sin(states[:,4])
    dx=states[:,0]-states[:,2];dy=states[:,1]-states[:,3]
    local=shapely.points(np.column_stack((c*dx+s*dy,-s*dx+c*dy)))
    solid=np.asarray(shapely.contains(geom,local),bool)
    boundary_dist=np.asarray(shapely.distance(geom.boundary,local),float)
    signed_dist=np.where(solid,-boundary_dist,boundary_dist)
    pieces=list(geom.geoms)if hasattr(geom,'geoms')else[geom]
    holes=[Polygon(ring)for part in pieces for ring in part.interiors]
    hole=np.zeros(len(states),bool)
    for ring in holes:hole|=shapely.contains(ring,local)
    verts=np.concatenate([np.asarray(part.exterior.coords)for part in pieces])
    xx=states[:,2,None]+c[:,None]*verts[None,:,0]-s[:,None]*verts[None,:,1]
    yy=states[:,3,None]+s[:,None]*verts[None,:,0]+c[:,None]*verts[None,:,1]
    wall=np.min(np.column_stack((xx.min(axis=1)-7,504-xx.max(axis=1),yy.min(axis=1)-7,504-yy.max(axis=1))),axis=1)
    pusher_wall=np.min(np.column_stack((states[:,0]-22,489-states[:,0],states[:,1]-22,489-states[:,1])),axis=1)
    gap=np.column_stack((wall,signed_dist-15,pusher_wall))
    center=np.asarray((geom.centroid.x,geom.centroid.y))
    centers=states[:,2:4]+np.column_stack((c*center[0]-s*center[1],s*center[0]+c*center[1]))
    return dict(gaps=gap,overlaps=np.maximum(0,-gap),solid=solid,hole=hole,centers=centers)

rows=[];totals=Counter();hash_bad=[];geometry_bad=[]
for index,entry in enumerate([*original['candidates'],*original['original']]):
    path=BASE/entry['file']
    actual_sha=sha(path)
    if actual_sha!=entry['sha256']:hash_bad.append(path.name)
    with np.load(path,allow_pickle=False)as data:
        identity=json.loads(str(data['identity_json']))
        spec=json.loads(str(data['spec_json']))
        controls=json.loads(str(data['control_metadata_json']))
        micro=json.loads(str(data['substep_metadata_json']))
        states=data['states'].copy();pre=data['substep_pre'].copy();post=data['substep_post'].copy()
        rgb=data['rgb'].copy();actions=data['actions'].copy()
    assert len(micro)==len(pre)==len(post)
    assert len(states)==len(controls)+1==len(rgb)
    assert np.array_equal(actions,np.asarray(spec['actions']))
    geom=geometry(identity);mx=measures(geom,post);pm=measures(geom,pre);boundary=measures(geom,states)
    expected=np.asarray([r.get('penetrations',[0,0,0])for r in micro],float)
    max_pen_diff=float(np.max(np.abs(expected-mx['overlaps'])))if len(micro)else 0.
    if max_pen_diff>1e-7:geometry_bad.append(dict(file=path.name,max_abs_difference=max_pen_diff))
    executed=np.asarray([bool(r.get('executed'))for r in micro])
    dt=np.asarray([float(r['dt'])for r in micro])
    constrained=np.asarray([bool(r.get('intervened',False))or (isinstance(r.get('intervention'),dict)and r['intervention'].get('delta_norm',0)>1e-8)for r in micro])
    illegal_saved=sum(r.get('pusher_region','')=='solid'or str(r.get('pusher_region','')).startswith('hole-')for r in micro)
    valid_controls=sum(bool(r.get('valid'))for r in controls)
    completed_duration=sum(float(r.get('physics_seconds',.1))for r in controls if r.get('valid'))
    physical_seconds=float(dt[executed].sum())
    actual_contact_seconds={kind:sum(float(r['dt'])for r in micro if r.get('executed')and any(e.get('phase')=='post_solve'and e.get('kind')==kind for e in r.get('contacts',[])))for kind in('agent_object','object_wall','agent_wall')}
    model_zero=True
    finite=bool(np.isfinite(states).all()and np.isfinite(pre).all()and np.isfinite(post).all())
    limits=mx['overlaps'].max(axis=0).tolist()if len(micro)else[0,0,0]
    center_solid=int(mx['solid'].sum());center_hole=int(mx['hole'].sum())
    geometry_safe=finite and not center_solid and not center_hole and max(limits)<=1+1e-8 and valid_controls==24
    path_world=float(np.linalg.norm(mx['centers']-pm['centers'],axis=1)[executed].sum())
    angular_path=float(np.degrees(np.abs(post[:,4]-pre[:,4]))[executed].sum())
    chord=np.linalg.norm(np.diff(boundary['centers'],axis=0),axis=1)
    angles=np.degrees(np.abs(np.diff(states[:,4])))
    validmask=np.asarray([bool(c.get('valid'))for c in controls])
    row=dict(file=path.name,letter=entry['letter'],scenario=entry['scenario'],kind=entry['kind'],sha256=actual_sha,
        attempted_controls=len(controls),valid_completed_controls=valid_controls,invalid_partial_controls=len(controls)-valid_controls,
        executed_microsteps=int(executed.sum()),physical_seconds=physical_seconds,
        completed_control_seconds=completed_duration,partial_seconds=physical_seconds-completed_duration,
        unique_dt_values=sorted(set(dt.tolist())),max_penetrations_OW_AO_AW_world_px=limits,
        minimum_signed_gaps_OW_AO_AW_world_px=mx['gaps'].min(axis=0).tolist(),
        saved_penetration_max_abs_error=max_pen_diff,center_solid_microsteps=center_solid,external_closed_hole_microsteps=center_hole,
        saved_illegal_region_microsteps=illegal_saved,
        intervened_microsteps=int((constrained&executed).sum()),intervention_seconds=float(dt[constrained&executed].sum()),
        intervention_time_ratio=float(dt[constrained&executed].sum()/physical_seconds)if physical_seconds else None,
        contact_seconds=actual_contact_seconds,finite=finite,geometry_gate_passed=geometry_safe,
        cumulative_microstep_translation_world_px=path_world,cumulative_microstep_translation_image_px=path_world*96/512,
        cumulative_control_chord_translation_world_px=float(chord.sum()),cumulative_microstep_rotation_deg=angular_path,
        max_valid_H1_translation_image_px=float((chord[validmask]*96/512).max())if validmask.any()else None,
        max_valid_H1_rotation_deg=float(angles[validmask].max())if validmask.any()else None,
        protocol_sha256=identity['protocol_sha256'],stored_identity_substep_dt=identity['physics']['substep_dt'],
        stored_identity_substeps_per_control=identity['physics']['substeps_per_control'],
        legacy_case_completed_controls=entry['completed_controls'],legacy_intervened_microsteps=entry['intervened_microsteps'])
    rows.append(row)
    totals['attempted_controls']+=len(controls);totals['completed_controls']+=valid_controls;totals['microsteps']+=int(executed.sum())
    totals['invalid_partial_controls']+=len(controls)-valid_controls
    if (index+1)%24==0:print(f'audited {index+1}/120',flush=True)
candidate=[r for r in rows if r['kind']=='candidate'];baseline=[r for r in rows if r['kind']=='original']
perletter=[]
for letter in 'TLOCXB':
    part=[r for r in candidate if r['letter']==letter]
    sec=sum(r['physical_seconds']for r in part)
    perletter.append(dict(letter=letter,cases=len(part),passing_cases=sum(r['geometry_gate_passed']for r in part),
        attempted_controls=sum(r['attempted_controls']for r in part),completed_controls=sum(r['valid_completed_controls']for r in part),
        invalid_partial_controls=sum(r['invalid_partial_controls']for r in part),physical_seconds=sec,
        intervention_time_ratio=sum(r['intervention_seconds']for r in part)/sec,
        max_penetrations_OW_AO_AW_world_px=np.max([r['max_penetrations_OW_AO_AW_world_px']for r in part],axis=0).tolist(),
        max_valid_H1_translation_image_px=max(r['max_valid_H1_translation_image_px']for r in part),
        max_valid_H1_rotation_deg=max(r['max_valid_H1_rotation_deg']for r in part)))

audit=dict(kind='saved-only-contact-regression-v1-audit.v1',checked_at_utc=datetime.now(timezone.utc).isoformat(),
    new_physics_controls=0,new_model_calls=0,GPU_calls=0,optimizer_updates=0,
    raw_files=len(rows),candidate_cases=len(candidate),original_cases=len(baseline),hash_mismatches=hash_bad,
    recomputed_geometry_mismatches=geometry_bad,cost_reconstructed=dict(totals),cost_original=original['cost'],
    cost_matches=all(totals[k]==original['cost'][k]for k in('attempted_controls','completed_controls','microsteps')),
    corrected_gate_passed=all(r['geometry_gate_passed']for r in candidate),candidate_failures=sum(not r['geometry_gate_passed']for r in candidate),
    per_letter=perletter,rows=rows,
    scope_limitations=[
        'Original per-case completed_controls includes invalid attempted control; use valid_completed_controls here.',
        'Original per-case intervened_microsteps uses truthiness of a nonempty dict; use delta_norm>1e-8 and actual dt here.',
        'Original illegal_microsteps field omits pusher_region. This audit recomputes solid/closed-hole region from saved geometry and post-state.',
        'Original cumulative translation sums control-boundary chords; this audit separately supplies actual microstep path.',
        'This v1 attempt ran v3 physics protocol d106d5c10f40a341121fd54531ace02066ce493dd4ba6a30913bd2982afa2ae0. Its raw identity still says adaptive/variable although every candidate microstep is .001s. Metadata-only revision changes identity and snapshot protocol binding, not motion.',
        'started.json binds the regression runner SHA, not the runtime physics source SHA. The reported e6ad source and its preserved snapshot are supported by parent execution history and metadata-revision; do not claim started.json contains that binding.',
        'Report-only audit does not establish a passing future candidate or authorize collection/training.',
    ],
    physics_source_metadata_revision=json.loads((PILOT/'physics-design/metadata-revision.json').read_text(encoding='utf-8')),
    reviewer_script_sha256=sha(__file__),wall_seconds=time.perf_counter()-started)
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'AUDIT-v1.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
md=['# First physical regression: saved-only audit','',
    f'120 raw files checked:96 candidate,24 original. Candidate gate FAILED: {audit["candidate_failures"]}/96 cases failed. No new model/GPU/physics or optimizer work.',
    '',f'Cost reconstructed: {dict(totals)}. Matches actual aggregate ledger: {audit["cost_matches"]}.',
    f'Raw hashes mismatched: {len(hash_bad)}; exact geometric penetration recomputation mismatched: {len(geometry_bad)}.','',
    '| Letter | Pass/16 | Actual attempted/completed controls | Intervention time | Max OW/AO/AW world px | max valid H1 px / deg |',
    '| --- | --- | --- | --- | --- | --- |']
for r in perletter:md.append(f'|{r["letter"]}|{r["passing_cases"]}|{r["attempted_controls"]}/{r["completed_controls"]}|{r["intervention_time_ratio"]:.2%}|'+ '/'.join(f'{v:.4f}'for v in r['max_penetrations_OW_AO_AW_world_px'])+f'|{r["max_valid_H1_translation_image_px"]:.3f}/{r["max_valid_H1_rotation_deg"]:.3f}|')
md+=['','## Corrections and identity limits','']+['- '+s for s in audit['scope_limitations']]
(OUT/'AUDIT-v1.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(json.dumps({k:audit[k]for k in('raw_files','candidate_failures','cost_reconstructed','cost_matches','hash_mismatches','recomputed_geometry_mismatches','wall_seconds')},indent=2))
assert not hash_bad and not geometry_bad and audit['cost_matches']
