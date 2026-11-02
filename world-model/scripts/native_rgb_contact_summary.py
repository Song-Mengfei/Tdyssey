"""Independent CPU contact denominators and frozen sampling-exposure audit.

Reads the compact contact core, not the 912 original trajectory archives. No
models, rendering, physics steps, new draw order, or training are invoked.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LETTERS=tuple('TLOCXB')
PHASES=('approach','push','withdraw','recontact','static')
MODES=('hold','free_route','contact_push_intent')
MOTIONS=('nearstatic','small','obvious')
STAGE=ROOT/'artifacts/native-rgb-supervised-20261029/contact-audit-20261030'
R3A=ROOT/'artifacts/native-rgb-supervised-20261029/r3a'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path),bytes=Path(path).stat().st_size)
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])if rows else ['no_rows'])
        writer.writeheader();writer.writerows(rows)
def fraction(n,d):return float(n/d)if d else None
def quantiles(values,weights=None):
    values=np.asarray(values)
    if weights is not None:values=np.repeat(values,np.asarray(weights,dtype=np.int64))
    if not len(values):return dict(count=0,minimum=None,p10=None,median=None,p90=None,p95=None,maximum=None)
    if not np.isfinite(values).all():raise ValueError('Motion summaries may not silently discard nonfinite values')
    return dict(count=len(values),**dict(zip(('minimum','p10','median','p90','p95','maximum'),
        map(float,np.quantile(values,[0,.1,.5,.9,.95,1])))))


def motion_classes(translation,rotation,definitions):
    t=np.asarray(translation);a=np.abs(rotation);near=definitions['nearstatic'];obvious=definitions['obvious']
    low=(t<near['translation_image96_px_lt'])&(a<near['abs_rotation_deg_lt'])
    high=(t>=obvious['translation_image96_px_ge'])|(a>=obvious['OR_abs_rotation_deg_ge'])
    if np.any(low&high):raise ValueError('Mutually exclusive motion thresholds overlap')
    return np.where(high,2,np.where(low,0,1)).astype(np.uint8)


def validate_core(core,definitions):
    n=len(core['control']);ti=core['trajectory_index'];ci=core['control']
    if n!=912*48 or len(set(zip(map(int,ti),map(int,ci))))!=n:raise ValueError('912 complete distinct48-control trajectories required')
    if set(map(int,ti))!=set(range(912))or not np.array_equal(np.bincount(ti),np.full(912,48)):
        raise ValueError('Core trajectory denominator differs')
    if ci.min()!=0 or ci.max()!=47:raise ValueError('Control range must be0..47')
    for field in('post_ao','post_ow','post_aw','post_ao_impulse_known','post_ao_impulse_max_norm'):
        if core[field].shape!=(n,10):raise ValueError('Expected ten actual100Hz substeps per control: '+field)
    for flag,total in(('post_ao','contact_substeps'),('post_ow','object_wall_substeps'),('post_aw','agent_wall_substeps')):
        if core[flag].dtype!=bool or not np.array_equal(core[flag].sum(axis=1),core[total]):raise ValueError('Time count differs from actual postsolve substep flags')
    if not np.array_equal(core['post_ao'][:,-1],core['terminal_contact']):raise ValueError('Terminal contact is not the final substep')
    classes=motion_classes(core['translation_image96_px'],core['signed_rotation_deg'],definitions)
    if not np.array_equal(classes,core['motion_class']):raise ValueError('Frozen motion thresholds differ from the core')
    intermediate=(core['translation_image96_px']>=definitions['intermediate']['translation_image96_px_ge'])|(np.abs(core['signed_rotation_deg'])>=definitions['intermediate']['OR_abs_rotation_deg_ge'])
    if not np.array_equal(intermediate,core['intermediate_motion']):raise ValueError('Intermediate threshold differs')
    known=core['post_ao']&core['post_ao_impulse_known'];norm=core['post_ao_impulse_max_norm']
    if not np.isfinite(norm[known]).all()or np.any(norm[known]<0):raise ValueError('Known impulse norms must be finite and nonnegative')
    if np.any(~np.isnan(norm[core['post_ao']&~known])):raise ValueError('Unknown-contact impulse must be NaN, not falsely zero')
    if not np.array_equal(known.sum(axis=1),core['known_impulse_contact_substeps']):raise ValueError('Known impulse denominator differs')
    for j,epsilon in enumerate(definitions['impulse_epsilon_sensitivity']):
        count=(known&(norm>epsilon)).sum(axis=1)
        if not np.array_equal(count,core['nonzero_impulse_counts'][:,j]):raise ValueError('Impulse existence sensitivity differs; never sum contact points')
    return dict(controls=n,substeps=n*10,successful_trajectories=912,all_primary_array_contracts_passed=True)


def summarize(core,indices,definitions,*,weights=None,unit='distinct_control'):
    idx=np.asarray(indices,dtype=np.int64)
    w=np.ones(len(idx),np.int64)if weights is None else np.asarray(weights,dtype=np.int64)
    if w.shape!=(len(idx),)or np.any(w<0):raise ValueError('Nonnegative explicit integer exposure weights required')
    total=int(w.sum());contact=core['contact_substeps'][idx]>0;motion=core['motion_class'][idx]
    wall=core['object_wall_substeps'][idx]>0;agentwall=core['agent_wall_substeps'][idx]>0
    count=lambda flag:int(w[np.asarray(flag)].sum())
    cross={f'{c}_{m}':dict(count=count((contact==yes)&(motion==i)),denominator=total)
        for c,yes in(('no_contact',False),('contact',True))for i,m in enumerate(MOTIONS)}
    for v in cross.values():v['fraction']=fraction(v['count'],total)
    if sum(v['count']for v in cross.values())!=total:raise AssertionError('Six classes must partition the full denominator')
    substeps=total*10;duty=int(core['contact_substeps'][idx].astype(np.int64)@w)
    ow=int(core['object_wall_substeps'][idx].astype(np.int64)@w);aw=int(core['agent_wall_substeps'][idx].astype(np.int64)@w)
    sensitivity={str(e):dict(substep_presence_count=int(core['nonzero_impulse_counts'][idx,j].astype(np.int64)@w),
        substep_denominator=substeps)for j,e in enumerate(definitions['impulse_epsilon_sensitivity'])}
    for v in sensitivity.values():v['duty']=fraction(v['substep_presence_count'],substeps)
    bins=core['contact_substeps'][idx]
    bin_counts={name:dict(count=count(mask),denominator=total)for name,mask in(
        ('0',bins==0),('1-4',(bins>=1)&(bins<=4)),('5-9',(bins>=5)&(bins<=9)),('10',bins==10))}
    for v in bin_counts.values():v['fraction']=fraction(v['count'],total)
    known=int(core['known_impulse_contact_substeps'][idx].astype(np.int64)@w)
    nearwall={name:dict(count=count(contact&(motion==0)&(wall==yes)),denominator=total)
        for name,yes in(('object_wall',True),('no_object_wall',False))}
    for v in nearwall.values():v['fraction_all_controls']=fraction(v['count'],total)
    push=core['phase'][idx]==PHASES.index('push')
    intent=core['mode'][idx]==MODES.index('contact_push_intent')
    pusher_moving=core['pusher_translation_image96_px'][idx]>=.1
    routing_phase=np.isin(core['phase'][idx],[PHASES.index(v)for v in('push','recontact','approach')])
    routing=routing_phase&(core['mode'][idx]==MODES.index('free_route'))&~contact
    operations={}
    for label,flag in(('no_AO_and_pusher_moving',~contact&pusher_moving),
            ('no_AO_object_nearstatic_and_pusher_moving',~contact&(motion==0)&pusher_moving),
            ('routing_or_recovery_candidate',routing),('routing_or_recovery_candidate_and_pusher_moving',routing&pusher_moving),
            ('planned_withdraw_without_AO',(core['phase'][idx]==PHASES.index('withdraw'))&~contact),
            ('planned_static_without_AO',(core['phase'][idx]==PHASES.index('static'))&~contact)):
        num=count(flag);operations[label]=dict(control_or_exposure_count=num,denominator=total,fraction=fraction(num,total),
            unique_classified_interval_seconds=num*.1 if unit!='action_prefix_control_exposure'else None,
            repeated_interval_exposure_equivalent_seconds=num*.1 if unit=='action_prefix_control_exposure'else None)
    obvious_contact=contact&(motion==2);primary=definitions['impulse_epsilon_sensitivity'].index(definitions['primary_impulse_norm_epsilon'])
    impulse=core['nonzero_impulse_counts'][idx,primary]>0
    evidence={}
    for label,flag in(('valid_impulse_above_primary_epsilon',impulse),('contact_substeps_at_least5',bins>=5),
            ('contact_substeps_all10',bins==10),('valid_impulse_and_contact_substeps_at_least5',impulse&(bins>=5))):
        mask=obvious_contact&flag;num=count(mask)
        evidence[label]=dict(count=num,all_control_or_exposure_denominator=total,fraction_all=fraction(num,total),
            any_contact_and_obvious_denominator=count(obvious_contact),fraction_among_any_contact_and_obvious=fraction(num,count(obvious_contact)),
            distinct_trajectory_support=len(np.unique(core['trajectory_index'][idx[mask&(w>0)]])))
    tzero=core['translation_image96_px'][idx]==0;rzero=core['signed_rotation_deg'][idx]==0
    zero_motion={name:dict(count=count(mask),denominator=total,fraction=fraction(count(mask),total))
        for name,mask in(('translation_exact_zero',tzero),('signed_rotation_exact_zero',rzero),('both_exact_zero',tzero&rzero))}
    return dict(unit=unit,denominator=total,distinct_raw_controls_with_positive_weight=int(np.count_nonzero(w)),
        measured_substep_or_exposure_denominator=substeps,
        acquired_seconds=total*.1 if unit=='distinct_acquired_control'else None,
        six_mutually_exclusive_classes=cross,any_actual_contact_controls=count(contact),any_actual_contact_fraction=fraction(count(contact),total),
        terminal_contact_controls=count(core['terminal_contact'][idx]),terminal_contact_fraction=fraction(count(core['terminal_contact'][idx]),total),
        contact_substeps=duty,contact_substep_duty=fraction(duty,substeps),
        object_wall_controls=count(wall),object_wall_substeps=ow,object_wall_substep_duty=fraction(ow,substeps),
        pusher_wall_controls=count(agentwall),pusher_wall_substeps=aw,pusher_wall_substep_duty=fraction(aw,substeps),
        contact_substep_bins=bin_counts,nonzero_impulse_presence_sensitivity=sensitivity,
        known_impulse_contact_substeps=known,unknown_impulse_actual_contact_substeps=duty-known,
        contact_and_obvious_persistence_and_impulse_evidence=evidence,
        exact_zero_motion=zero_motion,exact_zero_note='Exact numeric equality to zero, not the nearstatic thresholds or rounding to a display precision.',
        persistence_note='At least5 of10 postsolve intervals need not be consecutive. These are co-occurrence/persistence sensitivity checks, not a causal pushing definition.',
        impulse_note='Existence of a valid postsolveAO norm above epsilon, not summed point/arc impulse, force or work.',
        contact_nearstatic_objectwall_vs_nonwall=nearwall,
        planned_push_controls=count(push),planned_push_without_actual_contact_controls=count(push&~contact),
        planned_push_without_actual_contact_fraction=fraction(count(push&~contact),count(push)),
        planned_contact_push_intent_controls=count(intent),planned_contact_push_intent_no_contact_controls=count(intent&~contact),
        planned_contact_push_intent_no_contact_fraction=fraction(count(intent&~contact),count(intent)),
        no_contact_non_nearstatic_motion_controls=count(~contact&(motion!=0)),
        intermediate_motion_controls=count(core['intermediate_motion'][idx]),intermediate_motion_fraction=fraction(count(core['intermediate_motion'][idx]),total),
        continuous_motion=dict(translation_image96_px=quantiles(core['translation_image96_px'][idx],w),
            signed_rotation_deg=quantiles(core['signed_rotation_deg'][idx],w),
            abs_rotation_deg=quantiles(np.abs(core['signed_rotation_deg'][idx]),w)),
        operational_interval_classifications=operations,
        operational_definition='At10Hz, noAO means no actual postsolveAO in any of10substeps; pusher moving means endpoint displacement>=.1image96px, not path length. Routing/recovery is only a planned approach/push/recontact free_route noAO candidate. Withdraw/static remain separate intended phases.',
        interval_duration_note='Classified0.1s control intervals are not the100Hz measured contact duration; overlapping denominator populations and repeated action-prefix exposures must not be added as acquired time.',
        causality='Contact plus displacement is observed co-occurrence, not a proof that all displacement was caused by the pusher.')


def compact_row(name,summary,**stratum):
    row=dict(denominator_name=name,**stratum,unit=summary['unit'],count=summary['denominator'],denominator=summary['denominator'],
        any_contact_count=summary['any_actual_contact_controls'],any_contact_fraction=summary['any_actual_contact_fraction'],
        contact_substep_count=summary['contact_substeps'],substep_denominator=summary['measured_substep_or_exposure_denominator'],
        contact_substep_duty=summary['contact_substep_duty'],terminal_contact_count=summary['terminal_contact_controls'],terminal_contact_fraction=summary['terminal_contact_fraction'],
        object_wall_controls=summary['object_wall_controls'],pusher_wall_controls=summary['pusher_wall_controls'],
        planned_push_controls=summary['planned_push_controls'],planned_push_no_contact=summary['planned_push_without_actual_contact_controls'],
        intent_controls=summary['planned_contact_push_intent_controls'],intent_no_contact=summary['planned_contact_push_intent_no_contact_controls'],
        intermediate_motion_count=summary['intermediate_motion_controls'],intermediate_motion_fraction=summary['intermediate_motion_fraction'])
    for key,v in summary['six_mutually_exclusive_classes'].items():row[key+'_count']=v['count'];row[key+'_fraction']=v['fraction']
    for label,v in summary['contact_substep_bins'].items():row['contact_substeps_'+label]=v['count']
    for label,v in summary['nonzero_impulse_presence_sensitivity'].items():
        row['impulse_presence_substeps_eps'+label]=v['substep_presence_count'];row['impulse_presence_duty_eps'+label]=v['duty']
    for label,v in summary['operational_interval_classifications'].items():
        row[label+'_count']=v['control_or_exposure_count'];row[label+'_fraction']=v['fraction']
    for label,v in summary['exact_zero_motion'].items():row[label+'_count']=v['count'];row[label+'_fraction']=v['fraction']
    for label,v in summary['continuous_motion'].items():
        for q in('minimum','p10','median','p90','p95','maximum'):row[label+'_'+q]=v[q]
    return row


def core_metadata_rows(payload):
    if isinstance(payload,list):return payload
    for key in('trajectories','rows','entries'):
        if key in payload:return payload[key]
    raise ValueError('Core trajectory metadata must identify its ordered manifest trajectories')


def frozen_exposures(core,metadata):
    with np.load(R3A/'data-interface/window-catalogue.npz',allow_pickle=False)as z:
        selected=z['selected_h1_training_indices'].copy();ti=z['h1_trajectory_index'][selected];ci=z['h1_anchor'][selected]
    lookup={(int(t),int(c)):i for i,(t,c)in enumerate(zip(core['trajectory_index'],core['control']))}
    target=np.asarray([lookup[int(t),int(c)]for t,c in zip(ti,ci)],np.int32)
    if len(target)!=23984 or len(np.unique(target))!=len(target)or np.any(core['split'][target]!=0):raise ValueError('Selected targets must be23984 distinct TRAIN controls')
    proposal=read(ROOT/'configs/native_rgb_r3b_proposal.json');reference=proposal['training_plan']['index_plan']
    if sha(reference['path'])!=reference['sha256']:raise ValueError('Frozen eight-epoch proposal index plan changed')
    with np.load(reference['path'],allow_pickle=False)as z:
        if not np.array_equal(z['selected_h1_indices'],selected):raise ValueError('Proposal selected-local mapping changed')
        epochs=z['batch_epochs'];offsets=z['batch_offsets'];flat=z['flat_local_indices']
        first=np.flatnonzero(epochs==1);batches=[flat[offsets[j]:offsets[j+1]].copy()for j in first]
    order=np.concatenate(batches)
    if len(batches)!=1499 or any(len(b)!=16 for b in batches)or not np.array_equal(np.sort(order),np.arange(len(target))):
        raise ValueError('Exact original first epoch must deliver each selected target once in1499 full16 batches')
    ids=[f"{metadata[int(core['trajectory_index'][i])]['trajectory_id']}:{int(core['control'][i])}:H1"for i in target[order]]
    order_sha=hashlib.sha256(json.dumps(ids,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if order_sha!='800655f0dc54234f92c9784dc4b0ff94e44910d488d39f32439ff625a2ee2335':
        raise ValueError('Actual first epoch window-ID order differs from the frozen800655... baseline')
    weights=np.zeros(len(core['control']),np.uint8)
    for t,c in zip(ti,ci):
        for prefix in range(int(c)-3,int(c)+1):weights[lookup[int(t),prefix]]+=1
    if int(weights.sum())!=95936 or np.any(weights>4):raise ValueError('Four aligned action-prefix exposures per selected target required')
    return target,batches,weights,dict(bind(reference['path']),first_epoch_window_order_sha256=order_sha)


def sampler_audit(core,metadata,target,batches,definitions):
    rows=[];families=[r['family_id']for r in metadata]
    for index,local in enumerate(batches):
        ids=target[local];ti=core['trajectory_index'][ids]
        if len({families[int(t)]for t in ti})!=len(ti):raise ValueError('Frozen batch repeats a trajectory family')
        s=summarize(core,ids,definitions,unit='distinct_selected_target_control')
        row=compact_row('C_TRAIN_selected',s,batch_index=index)
        trans=core['translation_image96_px'][ids]>=2;rot=np.abs(core['signed_rotation_deg'][ids])>=5
        for name,flag in(('both',trans&rot),('translation_only',trans&~rot),('rotation_only',~trans&rot),('other',~trans&~rot)):
            row[name+'_windows']=int(np.count_nonzero(flag))
        rows.append(row)
    def spread(field):
        values=np.asarray([r[field]for r in rows]);j=int(np.argmin(values))
        return dict(quantiles=quantiles(values),minimum_batch_index=j,minimum=float(values[j]),
            zero_batches=int(np.count_nonzero(values==0)),denominator_batches=len(rows))
    summary=dict(first_epoch_batches=len(rows),batch_size=16,targets_delivered=len(target),distinct_targets=len(target),
        repeated_targets=0,unexposed_selected_targets=0,family_separation_passed=True,
        any_contact_fraction=spread('any_contact_fraction'),contact_and_obvious_fraction=spread('contact_obvious_fraction'),
        order_policy='Exact frozen proposal first epoch; no redrawn order or category quota.',
        optimizer_attempts_executed=0)
    return summary,rows


def trajectory_support(core,metadata,indices):
    idx=np.asarray(indices);ti=core['trajectory_index'][idx];contact=core['contact_substeps'][idx]>0;motion=core['motion_class'][idx]
    groups={f'{c}_{m}':(contact==yes)&(motion==i)for c,yes in(('no_contact',False),('contact',True))for i,m in enumerate(MOTIONS)}
    groups.update(any_actual_contact=contact,contact_and_obvious=contact&(motion==2),all_selected=np.ones(len(idx),bool))
    trans=core['translation_image96_px'][idx]>=2;rot=np.abs(core['signed_rotation_deg'][idx])>=5
    for name,flag in(('both',trans&rot),('translation_only',trans&~rot),('rotation_only',~trans&rot),('other',~trans&~rot)):
        groups['original_category_'+name]=flag;groups['contact_original_category_'+name]=contact&flag
    result={}
    for name,mask in groups.items():
        counts=np.bincount(ti[mask],minlength=len(metadata));total=int(counts.sum());rank=np.argsort(-counts,kind='stable')
        top=[dict(trajectory_id=metadata[int(t)]['trajectory_id'],windows=int(counts[t]))for t in rank[:10]if counts[t]>0]
        result[name]=dict(windows=total,independent_trajectory_support=int(np.count_nonzero(counts)),
            all_TRAIN_trajectory_denominator=768,top1_fraction=fraction(int(counts[rank[:1]].sum()),total),
            top5_fraction=fraction(int(counts[rank[:5]].sum()),total),top10_fraction=fraction(int(counts[rank[:10]].sum()),total),
            top10=top,window_counts_per_TRAIN_trajectory=quantiles(counts[[i for i,r in enumerate(metadata)if r['split']=='train']]))
    return result


def arc_concentration(core,indices,metadata):
    idx=np.asarray(indices);contact=core['contact_substeps'][idx]>0;ids=idx[contact];counts=Counter()
    for i in ids:
        arc=int(core['dominant_arc'][i]);letter=LETTERS[int(core['letter'][i])]
        counts[letter+':'+(str(arc)if arc>=0 else'unknown')]+=1
    total=len(ids)
    if sum(counts.values())!=total:raise AssertionError('Dominant arc assigns exactly one class to each contacted control')
    ordered=sorted(counts.items(),key=lambda item:(-item[1],item[0]))
    byletter={}
    for letter in LETTERS:
        choices=[(k,v)for k,v in ordered if k.startswith(letter+':')];n=sum(v for k,v in choices)
        byletter[letter]=dict(contacted_control_denominator=n,counts=dict(choices),
            fractions={k:fraction(v,n)for k,v in choices},top1_fraction=fraction(sum(v for k,v in choices[:1]),n),
            top5_fraction=fraction(sum(v for k,v in choices[:5]),n))
    return dict(contacted_control_denominator=total,dominant_arc_counts=dict(ordered),by_letter=byletter,
        dominant_arc_fraction={k:fraction(v,total)for k,v in ordered},
        top1_fraction=fraction(sum(v for k,v in ordered[:1]),total),top5_fraction=fraction(sum(v for k,v in ordered[:5]),total),
        unit='one dominant actual postsolve arc per contacted control; unknown included',
        multi_arc_marginals_summed=False)


def recontact_audit(core,metadata):
    rows=[]
    for t,record in enumerate(metadata):
        idx=np.flatnonzero(core['trajectory_index']==t);idx=idx[np.argsort(core['control'][idx])]
        planned=np.flatnonzero(core['phase'][idx]==PHASES.index('recontact'))
        if not len(planned):continue
        start=int(planned[0]);post=core['post_ao'][idx].reshape(-1);found=np.flatnonzero(post[start*10:])
        observed=bool(len(found));delay=float(found[0]*.01)if observed else None
        rows.append(dict(trajectory_id=record['trajectory_id'],split=record['split'],letter=record['letter'],
            scenario=record['scenario'],regiontype=region_type(record['initial_space']),initial_space=record['initial_space'],
            planned_recontact_start_control=start,planned_recontact_start_seconds=start*.1,observed=observed,delay_seconds=delay,
            actual_contact_interval_start_seconds=None if delay is None else start*.1+delay,
            actual_contact_postsolve_endpoint_seconds=None if delay is None else start*.1+delay+.01,
            secondary_endpoint_delay_seconds=None if delay is None else delay+.01,
            censor_seconds=(48-start)*.1 if not observed else None))
    def aggregate(selected):
        known=[r['delay_seconds']for r in selected if r['observed']]
        return dict(planned_recontact_trajectories=len(selected),observed_recovery_trajectories=len(known),
            unrecovered_right_censored_trajectories=len(selected)-len(known),unrecovered_fraction=fraction(len(selected)-len(known),len(selected)),
            observed_delay_seconds=quantiles(known),censor_intervals_seconds=quantiles([r['censor_seconds']for r in selected if not r['observed']]))
    strata=[]
    for split in('all','train','seen_val'):
        scoped=rows if split=='all'else[r for r in rows if r['split']==split]
        for field in('letter','scenario','regiontype','initial_space'):
            for value in sorted({r[field]for r in scoped}):
                strata.append(dict(split=split,stratum=field,value=value,**aggregate([r for r in scoped if r[field]==value])))
    return dict(all=aggregate(rows),by_split={split:aggregate([r for r in rows if r['split']==split])for split in('train','seen_val')},strata=strata,
        timing='First actual postsolve interval start at/after first planned recontact control; observed delays and right-censored failures kept separate.'),rows


def region_type(space):return'center'if space=='center'else'corners'if'-'in space else'edges'


def run(core_path,trajectory_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    definitions=read(STAGE/'definitions.json');payload=read(trajectory_path);metadata=core_metadata_rows(payload)
    if len(metadata)!=912:raise ValueError('Complete ordered metadata for912 successful trajectories required')
    if isinstance(payload,dict)and payload.get('definitions_sha256')!=sha(STAGE/'definitions.json'):
        raise ValueError('Core metadata does not bind the frozen definitions')
    original=read(R3A/'manifest.json')['entries']
    for record,entry in zip(metadata,original):
        if any(record[key]!=entry[key]for key in('trajectory_id','family_id','split','letter','initial_space','scenario')):
            raise ValueError('Core metadata order or stratum differs from the frozen successful manifest')
    with np.load(core_path,allow_pickle=False)as z:core={k:z[k].copy()for k in z.files}
    for field,labels in(('split',('train','seen_val')),('letter',LETTERS)):
        if any(labels[int(value)]!=metadata[int(t)][field]for value,t in zip(core[field],core['trajectory_index'])):
            raise ValueError('Core split/letter labels differ from trajectory metadata')
    contract=validate_core(core,definitions);target,batches,weights,plan=frozen_exposures(core,metadata)
    train=np.flatnonzero(core['split']==0);val=np.flatnonzero(core['split']==1)
    trainlegal=train[core['control'][train]>=3];vallegal=val[core['control'][val]>=3]
    if list(map(len,(train,val,trainlegal,vallegal)))!=[36864,6912,34560,6480]:raise ValueError('All/eligible TRAIN and validation denominators changed')
    groups={'A_TRAIN_all':(train,None,'distinct_acquired_control'),'B_TRAIN_legal_H1':(trainlegal,None,'distinct_H1_eligible_target_control'),
        'C_TRAIN_selected':(target,None,'distinct_selected_target_control'),'VAL_all':(val,None,'distinct_acquired_control'),
        'VAL_legal_H1':(vallegal,None,'distinct_H1_eligible_target_control'),
        'TRAIN_action_prefix_exposure':(train,weights[train],'action_prefix_control_exposure')}
    summaries={k:summarize(core,ids,definitions,weights=w,unit=unit)for k,(ids,w,unit)in groups.items()}
    crossrows=[]
    for name,s in summaries.items():
        for key,v in s['six_mutually_exclusive_classes'].items():crossrows.append(dict(denominator_name=name,unit=s['unit'],class_name=key,**v))
    csv_write(out/'mutually-exclusive.csv',crossrows)
    dimensions=dict(split=np.asarray([('train','seen_val')[int(v)]for v in core['split']]),
        letter=np.asarray([LETTERS[int(v)]for v in core['letter']]),
        regiontype=np.asarray([region_type(metadata[int(t)]['initial_space'])for t in core['trajectory_index']]),
        initial_space=np.asarray([metadata[int(t)]['initial_space']for t in core['trajectory_index']]),
        scenario=np.asarray([metadata[int(t)]['scenario']for t in core['trajectory_index']]),
        half=np.where(core['control']<24,'first24','last24'),control_index=core['control'].astype(str),
        phase=np.asarray([PHASES[int(v)]for v in core['phase']]),mode=np.asarray([MODES[int(v)]for v in core['mode']]))
    dimensions['phase_x_mode']=np.char.add(np.char.add(dimensions['phase'],':'),dimensions['mode'])
    dimensions['object_wall']=np.where(core['object_wall_substeps']>0,'post_contact','no_post_contact')
    dimensions['pusher_wall']=np.where(core['agent_wall_substeps']>0,'post_contact','no_post_contact')
    stratarows=[]
    for name,(ids,w,unit)in groups.items():
        for field,values in dimensions.items():
            for value in sorted(np.unique(values[ids])):
                mask=values[ids]==value;s=summarize(core,ids[mask],definitions,weights=None if w is None else w[mask],unit=unit)
                stratarows.append(compact_row(name,s,stratum=field,value=str(value),population_denominator=int(len(ids)if w is None else w.sum())))
    csv_write(out/'strata.csv',stratarows)
    operationrows=[]
    for name,s in summaries.items():
        for label,v in s['operational_interval_classifications'].items():
            operationrows.append(dict(denominator_name=name,stratum='all',value='all',unit=s['unit'],operation=label,**v))
    for row in stratarows:
        if row['stratum']not in('phase','mode','phase_x_mode'):continue
        for label in summaries[row['denominator_name']]['operational_interval_classifications']:
            count=row[label+'_count'];weighted=row['unit']=='action_prefix_control_exposure'
            operationrows.append(dict(denominator_name=row['denominator_name'],stratum=row['stratum'],value=row['value'],unit=row['unit'],
                operation=label,control_or_exposure_count=count,denominator=row['denominator'],fraction=row[label+'_fraction'],
                unique_classified_interval_seconds=None if weighted else count*.1,
                repeated_interval_exposure_equivalent_seconds=count*.1 if weighted else None))
    csv_write(out/'operation-time.csv',operationrows)
    sampler,batchrows=sampler_audit(core,metadata,target,batches,definitions);csv_write(out/'batches.csv',batchrows)
    recovery,recoveryrows=recontact_audit(core,metadata)
    csv_write(out/'recontact.csv',recoveryrows)
    np.savez_compressed(out/'input-exposure.npz',raw_control_row=np.arange(len(weights),dtype=np.int32),
        trajectory_index=core['trajectory_index'],control=core['control'],split=core['split'],action_prefix_exposure_weight=weights,
        selected_target_control_rows=target)
    result=dict(kind='R3A-independent-contact-denominators-and-sampling-summary.v1',passed=True,
        definitions=bind(STAGE/'definitions.json'),core=bind(core_path),trajectory_metadata=bind(trajectory_path),
        exact_first_epoch_index_plan=plan,core_contract=contract,denominators=summaries,sampler=sampler,
        selected_target_support_and_concentration=trajectory_support(core,metadata,target),
        dominant_actual_arc=dict(TRAIN_all=arc_concentration(core,train,metadata),selected_target=arc_concentration(core,target,metadata),
            VAL_all=arc_concentration(core,val,metadata)),
        action_prefix_weights=dict(total_exposures=int(weights.sum()),distinct_raw_TRAIN_controls=int(np.count_nonzero(weights[train])),
            zero_exposure_raw_TRAIN_controls=int(np.count_nonzero(weights[train]==0)),raw_TRAIN_control_denominator=len(train),
            weights_including_zeros=quantiles(weights[train]),positive_weights_only=quantiles(weights[train][weights[train]>0]),
            unit='Repeated input-action control exposures, not additional acquired physics time or independent samples'),
        recontact=recovery,excluded_early_TRAIN_controls=len(train)-len(trainlegal),
        failure_partial_scope='Six failed attempts and their costs remain separate from this912-successful-scene contact denominator; no failed controls are silently promoted to valid training targets.',
        no_contact_interpretation='No-contact nearstatic and no-contact residual/small/obvious movement remain separate; planned intent is not actual contact.',
        selected_subset_inference='Selected-target rates describe the fixed selected TRAIN pool. Deterministic negative selection is not an inverse-probability sample of the full library.',
        new_physics_controls=0,model_calls=0,checkpoint_loads=0,new_training_draw_order_generated=False,Torch_imported='torch'in sys.modules)
    if result['Torch_imported']:raise RuntimeError('CPU statistics unexpectedly imported Torch')
    write(out/'main_summary.json',result)
    files=[p for p in out.rglob('*')if p.is_file()and p.name!='inventory.json'];size=sum(p.stat().st_size for p in files)
    if size>3*1024**2:raise ValueError('Compact statistics exceeded3MiB; do not create a verbose control CSV')
    write(out/'inventory.json',dict(files=[bind(p)for p in files],statistics_bytes_before_inventory=size,cap_bytes=3*1024**2))
    print(json.dumps(dict(passed=True,statistics_bytes=size,denominators={k:v['denominator']for k,v in summaries.items()},
        contact_fraction={k:v['any_actual_contact_fraction']for k,v in summaries.items()},sampler=sampler)))
    return 0


def self_test():
    definitions=read(STAGE/'definitions.json');n=6
    core=dict(control=np.arange(n,dtype=np.uint8),trajectory_index=np.zeros(n,np.int16),
        contact_substeps=np.asarray([0,0,0,1,5,10],np.uint8),object_wall_substeps=np.asarray([0,0,0,1,0,0],np.uint8),
        agent_wall_substeps=np.zeros(n,np.uint8),terminal_contact=np.asarray([0,0,0,0,0,1],bool),
        translation_image96_px=np.asarray([0.,.2,2.,0.,.2,2.]),signed_rotation_deg=np.asarray([0.,0.,-6.,0.,0.,6.]),
        pusher_translation_image96_px=np.ones(n),
        known_impulse_contact_substeps=np.asarray([0,0,0,1,5,10],np.uint8),nonzero_impulse_counts=np.tile([0,0,0,1,5,10],(4,1)).T,
        phase=np.full(n,PHASES.index('push')),mode=np.full(n,MODES.index('contact_push_intent')),
        intermediate_motion=np.asarray([0,0,1,0,0,1],bool))
    core['motion_class']=motion_classes(core['translation_image96_px'],core['signed_rotation_deg'],definitions)
    s=summarize(core,np.arange(n),definitions)
    assert all(v['count']==1 for v in s['six_mutually_exclusive_classes'].values())
    assert s['contact_substeps']==16 and s['contact_substep_duty']==16/60
    assert s['any_actual_contact_fraction']==.5 and s['terminal_contact_controls']==1
    assert s['continuous_motion']['signed_rotation_deg']['minimum']==-6
    assert s['exact_zero_motion']['translation_exact_zero']['count']==2
    w=np.asarray([0,2,0,3,0,1]);x=summarize(core,np.arange(n),definitions,weights=w,unit='action_prefix_control_exposure')
    assert x['denominator']==6 and x['contact_substeps']==13 and x['acquired_seconds']is None
    assert sum(v['count']for v in x['six_mutually_exclusive_classes'].values())==6
    delayed=dict(trajectory_index=np.repeat(np.arange(2),48),control=np.tile(np.arange(48),2),
        phase=np.full(96,PHASES.index('push')),post_ao=np.zeros((96,10),bool))
    delayed['phase'][26:31]=PHASES.index('recontact');delayed['phase'][48+26:48+31]=PHASES.index('recontact')
    delayed['post_ao'][39,1]=True
    records=[dict(trajectory_id='fixture'+str(t),split='train',letter='B',scenario='release_recontact',initial_space='center')for t in range(2)]
    recovery,details=recontact_audit(delayed,records)
    assert recovery['all']['unrecovered_fraction']==.5 and abs(details[0]['delay_seconds']-1.31)<1e-12
    assert abs(details[0]['actual_contact_postsolve_endpoint_seconds']-3.92)<1e-12
    return dict(passed=True,fixture_only=True,checks=['six exclusive classes sum100%','one-substep contact is .01s not .1s',
        'terminal versus any contact differ','signed rotation and exact-zero counts preserved','integer prefix exposure weights preserve unit and denominator',
        'recontact interval-start1.31 versus endpoint1.32 delay and right censor kept separate'],
        model_calls=0,physics_controls=0,Torch_imported='torch'in sys.modules)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core',type=Path,default=STAGE/'core/controls.npz')
    p.add_argument('--trajectories',type=Path,default=STAGE/'core/trajectories.json')
    p.add_argument('--output',type=Path,default=STAGE/'statistics');p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()));return 0
    return run(a.core,a.trajectories,a.output)


if __name__=='__main__':raise SystemExit(main())
