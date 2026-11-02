"""Cold actual-contact/effective-motion extraction from saved numeric arrays.

No simulator, renderer, Torch, model or dynamics is constructed or invoked.
Arbiter total_impulse is copied per contact point by the original recorder:
only per-substep existence is measurable here, never a sum of point impulses.
"""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from .envs.official import official_shape
from .native_rgb_training import atomic_json

LETTERS=tuple('TLOCXB')
PHASES=('approach','push','withdraw','recontact','static')
MODES=('hold','free_route','contact_push_intent')
EPSILONS=(0.,1e-8,1e-6,1e-4)
DEFINITIONS_SHA='f47ca51f69f8ff276a30672fd06f5ba19d74b90ac3c86db6640c0b1e4a70380b'
REQUIRED_ARRAYS=('states','actions','attempted_actions','control_valid','substep_states','substep_status',
    'substep_control_index','substep_local_index','contact_flags','contact_values','contact_tags',
    'contact_arc_offsets','contact_arc_ids','control_phase','control_mode','control_substep_offsets')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_npz(path,**arrays):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)
def materialize(path):
    """Decode each needed zip member once, then release its file handle."""
    with np.load(path,allow_pickle=False) as z:
        return {key:z[key] for key in REQUIRED_ARRAYS if key in z.files}
@lru_cache(maxsize=6)
def centroid(letter):return np.asarray(official_shape(letter).centroid,np.float64)


def motion(states,letter):
    states=np.asarray(states,np.float64);local=centroid(letter);c=np.cos(states[:,4]);s=np.sin(states[:,4])
    area=states[:,2:4]+np.column_stack((c*local[0]-s*local[1],s*local[0]+c*local[1]))
    trans=np.linalg.norm(np.diff(area,axis=0),axis=1)*96/512
    angle=np.diff(states[:,4])*180/math.pi
    pusher=np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1)*96/512
    return trans,angle,pusher


def classify_motion(translation,rotation):
    translation=np.asarray(translation);rotation=np.asarray(rotation)
    obvious=(translation>=2.)|(np.abs(rotation)>=5.)
    near=(translation<.1)&(np.abs(rotation)<.1)
    classes=np.where(obvious,2,np.where(near,0,1)).astype(np.uint8)
    return classes,(translation>=.5)|(np.abs(rotation)>=1.)


def boolean_runs(bits):
    bits=np.asarray(bits,bool);diff=np.diff(np.r_[False,bits,False].astype(np.int8))
    return list(zip(np.flatnonzero(diff==1).tolist(),np.flatnonzero(diff==-1).tolist()))


def actual_signals(arrays):
    flags=np.asarray(arrays['contact_flags']);m=len(flags);tags=np.asarray(arrays['contact_tags']);values=np.asarray(arrays['contact_values'])
    ao=(flags[:,0]&2)!=0;ow=(flags[:,1]&2)!=0;aw=(flags[:,2]&2)!=0
    known=np.zeros(m,bool);maxnorm=np.zeros(m,np.float64);arc_substeps=[set() for _ in range(m)]
    post=(tags[:,1]==0)&(tags[:,2]==1) if len(tags) else np.zeros(0,bool)
    selected=np.flatnonzero(post&((tags[:,4]&4)!=0)) if len(tags) else np.empty(0,np.int64)
    if len(selected):
        ids=tags[selected,0].astype(np.int64);vectors=values[selected,7:9].astype(np.float64)
        finite=np.isfinite(vectors).all(axis=1);ids=ids[finite];vectors=vectors[finite]
        if np.any(ids<0) or np.any(ids>=m) or not ao[ids].all():raise ValueError('Valid postsolve AO point does not match original AO flag')
        known[ids]=True
        # Maximum norm is an existence/sensitivity helper only. Repeated point
        # copies and multiple arbiters never inflate a per-substep count.
        np.maximum.at(maxnorm,ids,np.linalg.norm(vectors,axis=1))
    maxnorm[ao&~known]=np.nan
    offsets=np.asarray(arrays['contact_arc_offsets']);arcs=np.asarray(arrays['contact_arc_ids'])
    for point in np.flatnonzero(post):
        substep=int(tags[point,0])
        if not 0<=substep<m or not ao[substep]:raise ValueError('Contact arc row outside recorded AO postsolve')
        arc_substeps[substep].update(map(int,arcs[offsets[point]:offsets[point+1]]))
    counts=np.stack([known&ao&(maxnorm>eps) for eps in EPSILONS],axis=1)
    return dict(ao=ao,ow=ow,aw=aw,impulse_known=known&ao,impulse_max_norm=maxnorm,nonzero=counts,arcs=arc_substeps)


def control_arc_summary(signals,ncontrols):
    offsets=[0];values=[];dominant=[]
    for control in range(ncontrols):
        # Each arc gets at most one vote per actual100Hz substep, even when a
        # corner maps several points/arbiters to the same arc.
        votes=Counter(arc for arcs in signals['arcs'][control*10:(control+1)*10] for arc in arcs)
        unique=sorted(votes);values.extend(unique);offsets.append(len(values))
        dominant.append(min(unique,key=lambda arc:(-votes[arc],arc)) if unique else -1)
    return np.asarray(dominant,np.int16),np.asarray(offsets,np.int32),np.asarray(values,np.int16)


def time_summary(signals,phases,*,substeps=None):
    ao=signals['ao'];m=len(ao) if substeps is None else substeps;runs=boolean_runs(ao[:m]);lengths=np.asarray([end-start for start,end in runs],np.float64)*.01
    result=dict(contact_substeps=int(np.count_nonzero(ao[:m])),contact_seconds=float(np.count_nonzero(ao[:m])*.01),
        contact_run_count=len(runs),contact_run_max_seconds=float(lengths.max()) if len(lengths) else 0.,
        contact_run_p10_seconds=float(np.percentile(lengths,10)) if len(lengths) else None,
        contact_run_p50_seconds=float(np.percentile(lengths,50)) if len(lengths) else None,
        contact_run_p90_seconds=float(np.percentile(lengths,90)) if len(lengths) else None,
        contact_run_p95_seconds=float(np.percentile(lengths,95)) if len(lengths) else None,
        known_impulse_contact_substeps=int(signals['impulse_known'][:m].sum()),
        unknown_impulse_contact_substeps=int((ao[:m]&~signals['impulse_known'][:m]).sum()))
    phase=np.asarray(phases);recontact=np.flatnonzero(phase==PHASES.index('recontact'))
    start_control=int(recontact[0]) if len(recontact) else None;start=None if start_control is None else start_control*10
    observed=np.flatnonzero(ao[start:m])+start if start is not None and start<m else np.empty(0,int)
    next_sub=int(observed[0]) if len(observed) else None
    new_run=next((begin for begin,end in runs if start is not None and begin>=start),None)
    result.update(recontact_start_control=start_control,recontact_start_substep=start,
        recontact_start_time_seconds=None if start is None else start*.01,
        next_actual_contact_start_substep=next_sub,next_actual_contact_start_time_seconds=None if next_sub is None else next_sub*.01,
        recontact_latency_seconds=None if next_sub is None else (next_sub-start)*.01,
        next_contact_observed_poststate_time_seconds=None if next_sub is None else (next_sub+1)*.01,
        observed_poststate_delay_seconds=None if next_sub is None else (next_sub+1-start)*.01,
        recontact_censored=None if start is None else next_sub is None,
        recontact_censor_seconds=None if start is None or next_sub is not None else (m-start)*.01,
        contact_at_recontact_start=None if start is None or start>=m else bool(ao[start]),
        next_new_contact_run_start_substep=new_run)
    return result,runs


def b014_regression(trajectory_id,signals,phases,scenario,roles):
    summary,runs=time_summary(signals,phases)
    expected=[(30,181),(391,443)]
    if runs!=expected or scenario!='release_recontact' or summary['recontact_start_control']!=26 or summary['next_actual_contact_start_substep']!=391 or abs(summary['recontact_latency_seconds']-1.31)>1e-12:
        raise AssertionError('Frozen B014 raw-contact/recontact regression differs')
    endpoint={str(frame):None if frame==0 else bool(signals['ao'][frame*10-1]) for frame in (0,12,24,36,48)}
    if endpoint!={'0':None,'12':True,'24':False,'36':False,'48':False}:raise AssertionError('B014 terminal frame mapping differs')
    return dict(trajectory_id=trajectory_id,passed=True,source_scenario=scenario,original_exhibition_roles=roles,
        actual_runs=[dict(start_substep=a,end_substep_exclusive=b,length_substeps=b-a,duration_seconds=(b-a)*.01) for a,b in runs],
        endpoint_postsolve_contact=endpoint,**summary,
        timing_note='Primary interval-start proxy3.91/delay1.31; corresponding poststate observation3.92/delay1.32, not an exact within-step collision onset.')


def duplicated_impulse_example(tid,arrays,signals):
    tags=arrays['contact_tags'];values=arrays['contact_values'];select=(tags[:,1]==0)&(tags[:,2]==1)&((tags[:,4]&4)!=0)
    for substep in np.unique(tags[select,0]):
        indices=np.flatnonzero(select&(tags[:,0]==substep))
        vectors=values[indices,7:9];unique,counts=np.unique(vectors,axis=0,return_counts=True)
        if len(counts) and counts.max()>1:
            i=int(np.argmax(counts));return dict(trajectory_id=tid,substep=int(substep),raw_postsolve_AO_point_rows=len(indices),
                duplicated_vector=unique[i].tolist(),identical_vector_point_rows=int(counts[i]),
                counted_once_as_nonzero=bool(signals['nonzero'][int(substep),1]),
                event_identity_not_recorded=True,total_impulse_not_summed=True)
    return None


def extract(stage):
    stage=Path(stage).resolve();began=time.perf_counter();definitions=stage/'definitions.json'
    if sha(definitions)!=DEFINITIONS_SHA:raise ValueError('Previously frozen contact definitions changed')
    definition=read(definitions);source=(stage/definition['source_manifest']).resolve();base=source.parent;manifest=read(source);index=read(base/'index.json')
    if not manifest.get('collection_closed') or len(manifest['entries'])!=912 or not index.get('collection_closed'):raise ValueError('Complete912 library required')
    core=stage/'core';core.mkdir(parents=True,exist_ok=True)
    chunks={};trajectories=[];time_rows=[];all_runs=[];arc_offsets=[0];arc_values=[];examples=[];b014=None;source_checks=[]
    provenance=base/'report/videos/video-provenance.json';media=read(provenance) if provenance.exists() else {'records':[]};media_roles={row['trajectory_id']:row.get('roles',[]) for row in media['records']}
    def append(key,value):chunks.setdefault(key,[]).append(np.asarray(value))
    for ti,entry in enumerate(manifest['entries']):
        path=base/entry['file'];metadata_path=base/entry['metadata_file']
        if sha(path)!=entry['sha256'] or sha(metadata_path)!=entry['metadata_sha256']:raise ValueError('Original R3A identity differs: '+entry['trajectory_id'])
        meta=read(metadata_path);arrays=materialize(path)
        if arrays['states'].shape!=(49,15) or arrays['contact_flags'].shape!=(480,3) or not arrays['control_valid'].all():raise ValueError('Success timeline/validity differs')
        signals=actual_signals(arrays);trans,angle,push=motion(arrays['states'],entry['letter']);classes,intermediate=classify_motion(trans,angle)
        dominant,offsets,arcs=control_arc_summary(signals,48)
        for key,value in dict(trajectory_index=np.full(48,ti,np.int16),control=np.arange(48,dtype=np.uint8),
            split=np.full(48,0 if entry['split']=='train' else 1,np.uint8),letter=np.full(48,LETTERS.index(entry['letter']),np.uint8),
            phase=arrays['control_phase'].astype(np.uint8),mode=arrays['control_mode'].astype(np.uint8),
            translation_image96_px=trans,signed_rotation_deg=angle,pusher_translation_image96_px=push,
            contact_substeps=signals['ao'].reshape(48,10).sum(axis=1).astype(np.uint8),terminal_contact=signals['ao'].reshape(48,10)[:,-1],
            object_wall_substeps=signals['ow'].reshape(48,10).sum(axis=1).astype(np.uint8),agent_wall_substeps=signals['aw'].reshape(48,10).sum(axis=1).astype(np.uint8),
            known_impulse_contact_substeps=signals['impulse_known'].reshape(48,10).sum(axis=1).astype(np.uint8),
            nonzero_impulse_counts=signals['nonzero'].reshape(48,10,4).sum(axis=1).astype(np.uint8),
            motion_class=classes,intermediate_motion=intermediate,dominant_arc=dominant,
            post_ao=signals['ao'].reshape(48,10),post_ow=signals['ow'].reshape(48,10),post_aw=signals['aw'].reshape(48,10),
            post_ao_impulse_known=signals['impulse_known'].reshape(48,10),post_ao_impulse_max_norm=signals['impulse_max_norm'].reshape(48,10)).items():append(key,value)
        start=len(arc_values);arc_values.extend(arcs.tolist());arc_offsets.extend((offsets[1:]+start).tolist())
        timing,runs=time_summary(signals,arrays['control_phase'])
        record=dict(trajectory_index=ti,trajectory_id=entry['trajectory_id'],family_id=entry['family_id'],split=entry['split'],letter=entry['letter'],
            initial_space=entry['initial_space'],scenario=entry['scenario'],seed=entry['seed'],source_npz_path=str(path),source_npz_sha256=entry['sha256'],
            source_metadata_path=str(metadata_path),source_metadata_sha256=entry['metadata_sha256'],original_exhibition_roles=media_roles.get(entry['trajectory_id'],[]),**timing)
        trajectories.append(record);time_rows.append(record)
        for a,b in runs:all_runs.append(dict(trajectory_index=ti,trajectory_id=entry['trajectory_id'],split=entry['split'],letter=entry['letter'],
            start_substep=a,end_substep_exclusive=b,length_substeps=b-a,start_time_seconds=a*.01,end_time_seconds=b*.01,duration_seconds=(b-a)*.01))
        if len(examples)<6:
            example=duplicated_impulse_example(entry['trajectory_id'],arrays,signals)
            if example is not None:examples.append(example)
        if entry['trajectory_id']=='R3A-seen_val-B-014-a0':b014=b014_regression(entry['trajectory_id'],signals,arrays['control_phase'],entry['scenario'],media_roles.get(entry['trajectory_id'],[]))
        if (ti+1)%152==0:print(json.dumps(dict(materialized_successes=ti+1,new_physics=0,model_calls=0)),flush=True)
    controls={key:np.concatenate(value,axis=0) for key,value in chunks.items()}
    controls['control_arc_offsets']=np.asarray(arc_offsets,np.int32);controls['control_arc_ids']=np.asarray(arc_values,np.int16)
    if controls['control'].shape!=(43776,) or not np.array_equal(controls['trajectory_index'],np.repeat(np.arange(912,dtype=np.int16),48)):raise ValueError('Canonical row order/count differs')
    if b014 is None:raise ValueError('Mandatory B014 regression source missing')
    write_npz(core/'controls.npz',**controls)
    atomic_json(core/'trajectories.json',dict(kind='contact-audit-manifest-ordered-trajectories.v1',definitions_sha256=sha(definitions),
        phase_codes=dict(enumerate(PHASES)),mode_codes=dict(enumerate(MODES)),letter_codes=dict(enumerate(LETTERS)),split_codes={0:'train',1:'seen_val'},trajectories=trajectories))
    scalar_fields=[key for key,value in time_rows[0].items() if not isinstance(value,(list,dict))]
    with (core/'trajectory-time.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=scalar_fields,extrasaction='ignore');writer.writeheader();writer.writerows(time_rows)
    with gzip.open(core/'contact-runs.csv.gz','wt',encoding='utf-8',newline='') as f:
        fields=['trajectory_index','trajectory_id','split','letter','start_substep','end_substep_exclusive','length_substeps','start_time_seconds','end_time_seconds','duration_seconds']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(all_runs)
    # Failure attempts use their actual complete/partial timelines and remain
    # outside the43776 successful-control denominator.
    failed=[]
    for scene in index['slots']:
        for attempt in scene['attempts']:
            if attempt['status']=='completed':continue
            path=base/attempt['trajectory_file'];arrays=materialize(path);meta=read(base/attempt['metadata_file'])
            if sha(path)!=attempt['trajectory_sha256']:raise ValueError('Failed raw evidence SHA changed')
            signals=actual_signals(arrays);n=len(arrays['actions']);c=len(arrays['attempted_actions']);m=len(arrays['substep_status'])
            trans,angle,pusher=motion(arrays['states'],scene['letter']);classes,intermediate=classify_motion(trans,angle)
            partial=m-n*10;partial_t=partial_r=partial_p=None
            if partial:
                partial_t,partial_r,partial_p=(float(v[0]) for v in motion(np.stack((arrays['states'][-1],arrays['substep_states'][-1])),scene['letter']))
            timing,_=time_summary(signals,arrays['control_phase'])
            failed.append(dict(trajectory_id=attempt['trajectory_id'],slot_id=scene['slot_id'],letter=scene['letter'],split=scene['split'],status=attempt['status'],
                charged_controls=attempt['charged_controls'],completed_controls=n,saved_attempted_controls=c,partial_controls=int(partial>0),partial_substeps=partial,
                known_substeps=int((arrays['substep_status']==1).sum()),unknown_advance_substeps=int((arrays['substep_status']==2).sum()),
                reserved_not_started_controls=attempt['charged_controls']-c,**timing,
                complete_control_nearstatic=int((classes==0).sum()),complete_control_small=int((classes==1).sum()),complete_control_obvious=int((classes==2).sum()),
                complete_control_intermediate=int(intermediate.sum()),complete_translation_image96_px=trans.tolist(),complete_signed_rotation_deg=angle.tolist(),
                complete_pusher_translation_image96_px=pusher.tolist(),partial_translation_image96_px=partial_t,partial_signed_rotation_deg=partial_r,partial_pusher_translation_image96_px=partial_p,
                source_file=str(path),source_sha256=attempt['trajectory_sha256'],failure=meta.get('failure')))
    failed_summary=dict(attempts=len(failed),charged_controls=sum(r['charged_controls'] for r in failed),completed_controls=sum(r['completed_controls'] for r in failed),
        known_substeps=sum(r['known_substeps'] for r in failed),unknown_advance_substeps=sum(r['unknown_advance_substeps'] for r in failed),reserved_not_started_controls=sum(r['reserved_not_started_controls'] for r in failed))
    atomic_json(core/'failed-attempts.json',dict(kind='contact-audit-failures-separate-from-main-controls.v1',summary=failed_summary,attempts=failed))
    failure_fields=[key for key,value in failed[0].items() if not isinstance(value,(dict,list))]
    with (core/'failed-attempts.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=failure_fields,extrasaction='ignore');writer.writeheader();writer.writerows(failed)
    atomic_json(core/'B014-regression.json',b014)
    source_file=Path(__file__).with_name('native_rgb_r3a_collection.py');source_lines=source_file.read_text(encoding='utf-8').splitlines()
    atomic_json(core/'impulse-provenance.json',dict(kind='duplicated-arbiter-total-impulse-observation-limit.v1',source_file=str(source_file),source_sha256=sha(source_file),
        source_evidence=[dict(line=i+1,text=line.strip()) for i,line in enumerate(source_lines) if "for point in event.get('points'" in line or "values[7:9]=event['total_impulse']" in line],
        examples=examples,per_point_summation_forbidden=True,event_identity_available=False,
        main_metric='any finite valid-mask4 postsolveAO impulse norm>1e-8 in each substep; point duplicates and multiple arbiters counted at most once',
        unknown='AO postsolve flag without any valid impulse row remains unknown; never a known zero'))
    schema=dict(kind='contact-audit-canonical-control-arrays.v1',rows=43776,row_order='manifest trajectory_index*48+control',
        fields={key:dict(dtype=str(value.dtype),shape=list(value.shape)) for key,value in controls.items()},
        impulse_epsilon_order=list(EPSILONS),unknown_impulse_derivation='contact_substeps-known_impulse_contact_substeps',
        post_ao_impulse_max_norm='existence helper only; noncontact0; flag-without-valid-row NaN; never a sum of event/point impulses',
        motion_class_codes={0:'nearstatic(<.1px AND abs<.1deg)',1:'small',2:'obvious(>=2px OR abs>=5deg)'},
        intermediate='>=.5image96px OR abs>=1deg',dominant_arc='distinct-substep coverage votes; tie smallestID; -1 when unmapped',
        timing=definition['time_interval_convention'],definitions_sha256=sha(definitions),no_original_RGB_or_substate_copies=True)
    atomic_json(core/'schema.json',schema)
    files=[dict(file=p.relative_to(core).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(core.iterdir()) if p.is_file()]
    total=sum(row['bytes'] for row in files)
    result=dict(kind='contact-audit-core-closed-manifest.v1',closed=True,successful_trajectories=912,successful_controls=43776,TRAIN_controls=36864,validation_controls=6912,
        legal_TRAIN_H1_target_controls=34560,legal_validation_H1_target_controls=6480,failed_summary=failed_summary,
        input_manifest_sha256=sha(source),input_index_sha256=sha(base/'index.json'),definitions_sha256=sha(definitions),files=files,
        core_bytes_before_manifest=total,compressed_core_under5MiB=total<5*1024**2,materialized_successful_NPZ=912,materialized_failed_NPZ=len(failed),
        each_NPZ_materialized_once=True,impulse_never_summed=True,B014_regression_passed=True,new_physics_controls=0,environment_constructions=0,RGB_render_calls=0,Torch_imported='torch' in __import__('sys').modules,model_calls=0,
        elapsed_seconds=time.perf_counter()-began)
    atomic_json(core/'manifest.json',result);return result
