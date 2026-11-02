"""Independent saved-score review. Run only after owner's completed callback.

No Torch/model/simulator, raw prediction loading, or change to original scoring.
"""
import collections, hashlib, itertools, json, math, pathlib, statistics

HERE=pathlib.Path(__file__).resolve().parent
RUN=HERE.parent.parent/'loss-definition-screen-20261102'
NEW=['G_ONLY','NATIVE_G','NATIVE_T']

def read(path): return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))
def save(name,value): (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def avg(values):
    good=[x for x in values if isinstance(x,(int,float)) and math.isfinite(x)]
    return statistics.mean(good) if good else None

def main():
    c=read(RUN/'frozen-config.json'); done=read(RUN/'completion.json')
    assert done['complete'] and not done['failed'], 'Owner result is not a completed experiment'
    data={'E2':read(c['E2_metrics']),'BASE':read(c['BASE_metrics'])}
    missing=[]
    for a in NEW:
        path=RUN/f'evaluation/{a}/metrics.json'
        if path.exists(): data[a]=read(path)
        else: missing.append(a)
    groups=collections.defaultdict(list)
    for name,d in data.items():
        assert len(d['rows'])==1248 and len(d['raw_files'])==78
        assert d['evidence']==data['E2']['evidence'],name
        unique=set()
        for row in d['rows']:
            key=(row['sample_id'],row['input_mode'],row['horizon'])
            assert key not in unique
            unique.add(key)
            group=(name,row['selection_role'],row['letter'],row['input_mode'],row['horizon'])
            groups[group].append(row)
            if row['input_mode']=='real_history' and row['strong_H1']:
                groups[(name,row['selection_role'],row['letter'],'real_history','strong_H1')].append(row)
    curves=[]
    for (name,role,letter,mode,h),rr in groups.items():
        curves.append(dict(label=name,role=role,letter=letter,mode=mode,horizon=h,n=len(rr),
            IoU=avg([r['metrics']['raw_mask_iou'] for r in rr]),
            position=avg([r['metrics']['object_position_error_image_px'] for r in rr]),
            contour=avg([r['metrics']['raw_contour_error_image_px'] for r in rr]),
            pusher_position=avg([r['metrics']['pusher_position_error_image_px'] for r in rr]),
            topology=dict(collections.Counter(r['visible_topology_status'] for r in rr)),
            shape=dict(collections.Counter(r['visible_shape_status'] for r in rr)),
            rotation=dict(collections.Counter(r['rotation']['status'] for r in rr))))
    own={(r['label'],r['role'],r['letter'],r['mode'],r['horizon']):r for r in curves}
    means_compared=0
    grouped_path=RUN/'report/grouped-all16.json'
    if grouped_path.exists():
        for row in read(grouped_path):
            key=(row['label'],row['role'],row['letter'],row['mode'],row['horizon'])
            for field in ['IoU','position','pusher_position']:
                x,y=own[key][field],row[field]
                assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-10,(key,field,x,y)
                means_compared+=1
            for field in ['topology','shape','rotation']: assert own[key][field]==row[field]
    paired=[]; support=[]
    for arm in NEW:
        if arm not in data: continue
        for role,letter,mode,h in itertools.product(['train','seen_val','original_failure'],'OCB',['real_history','autoregressive'],[1,4,8,16]):
            before=groups[('BASE',role,letter,mode,h)]
            after={r['sample_id']:r for r in groups[(arm,role,letter,mode,h)]}
            cases=[]
            for b in before:
                a=after[b['sample_id']]
                x,y=a['metrics']['raw_mask_iou'],b['metrics']['raw_mask_iou']
                px,py=a['metrics']['object_position_error_image_px'],b['metrics']['object_position_error_image_px']
                cases.append(dict(sample_id=b['sample_id'],family_id=b['family_id'],
                    delta_IoU=x-y if x is not None and y is not None else None,
                    delta_position=px-py if px is not None and py is not None else None,
                    BASE_topology=b['visible_topology_status'],candidate_topology=a['visible_topology_status'],
                    BASE_shape=b['visible_shape_status'],candidate_shape=a['visible_shape_status']))
            paired.append(dict(arm=arm,role=role,letter=letter,mode=mode,horizon=h,cases=cases,
                IoU_better=sum(r['delta_IoU'] is not None and r['delta_IoU']>1e-12 for r in cases),
                IoU_worse=sum(r['delta_IoU'] is not None and r['delta_IoU']< -1e-12 for r in cases),
                position_better=sum(r['delta_position'] is not None and r['delta_position']< -1e-12 for r in cases),
                position_worse=sum(r['delta_position'] is not None and r['delta_position']>1e-12 for r in cases)))
        for letter in 'OCB':
            ba=own[('BASE','seen_val',letter,'autoregressive',4)]; ca=own[(arm,'seen_val',letter,'autoregressive',4)]
            br=own[('BASE','seen_val',letter,'real_history','strong_H1')]; cr=own[(arm,'seen_val',letter,'real_history','strong_H1')]
            conditions=dict(AR4_IoU_improved=ca['IoU'] is not None and ba['IoU'] is not None and ca['IoU']>ba['IoU'],
                AR4_position_not_worse=ca['position'] is not None and ba['position'] is not None and ca['position']<=ba['position'],
                RH_strong_IoU_not_worse=cr['IoU'] is not None and br['IoU'] is not None and cr['IoU']>=br['IoU'],
                RH_strong_position_not_worse=cr['position'] is not None and br['position'] is not None and cr['position']<=br['position'])
            topo=[dict(horizon=h,BASE=own[('BASE','seen_val',letter,'autoregressive',h)]['topology'],
                       candidate=own[(arm,'seen_val',letter,'autoregressive',h)]['topology']) for h in [1,4,8,16]]
            top_ok=all(r['candidate'].get('fail',0)<=r['BASE'].get('fail',0) for r in topo)
            support.append(dict(arm=arm,letter=letter,conditions=conditions,topology=topo,
                                topology_no_worse=top_ok,predeclared_directional_support=all(conditions.values()) and top_ok))
    coeff=read(RUN/'coefficients.json'); pf=read(RUN/'preflight.json')
    independently_calculated={}
    for key in ['G','T']:
        ratios=[r['objectives']['native']['gradient_norm']/r['objectives'][key]['gradient_norm'] for r in pf['rows']]
        value=statistics.median(ratios)
        assert math.isclose(value,coeff['c'+key],rel_tol=1e-12,abs_tol=0)
        independently_calculated['c'+key]=value
    logs={'BASE':read(c['BASE_log'])}; restore={}; training={}; rng_pairs=0
    for arm in NEW:
        path=RUN/f'training/{arm}-log.json'
        if not path.exists(): continue
        log=read(path); assert len(log)==128
        logs[arm]=log
        for a,b in zip(logs['BASE'],log):
            assert a['branches']==b['branches'] and a['conditioning_rng_hashes']==b['conditioning_rng_hashes']
            rng_pairs+=1
        norms=[r['gradient_norm_before_clip'] for r in log]
        training[arm]=dict(attempts=len(log),last_updates=log[-1]['updates'],clip_count=sum(v>1 for v in norms),
            max_norm=max(norms),mean_norm=avg(norms),result=read(RUN/f'training/{arm}-result.json'))
        path=RUN/f'restore-{arm}.json'
        restore[arm]=read(path) if path.exists() else None
    source_checks=[]
    for b in c['bindings']:
        path=pathlib.Path(b['path'])
        if path.parent==RUN and path.suffix=='.py':
            actual=digest(path); source_checks.append(dict(path=str(path),frozen_sha256=b['sha256'],current_sha256=actual,matches=actual==b['sha256']))
    counts=done['ledger']['counts']; charged=sum(r['images_attempted'] for r in counts.values()); completed=sum(r['images_completed'] for r in counts.values())
    assert charged<=c['caps']['model_image_forwards']
    diagnostics=read(RUN/'diagnosis/summary.json'); measurements=read(RUN/'diagnosis/measurements.json')
    assert diagnostics['model_calls']==0 and diagnostics['optimizer_updates']==0 and len(measurements)<=288
    assert diagnostics['tensor_only_gradient_calls']<=64
    diag_groups=collections.defaultdict(list)
    for row in measurements: diag_groups[row['variant']].append(row)
    diag_aggregate=[]
    for variant,rr in diag_groups.items():
        diag_aggregate.append(dict(variant=variant,n=len(rr),qualifications=dict(collections.Counter(r['qualification'] for r in rr)),
            all_losses_finite=all(r['finite'] for r in rr),
            mean_losses={name:avg([r['losses'][name] for r in rr if r['qualification'].startswith('eligible')]) for name in rr[0]['losses']}))
    auxiliary=[]
    for label,d in data.items():
        aux=d.get('auxiliary_RGB_mask_diagnostics',[])
        for role,letter,mode,h in itertools.product(['train','seen_val','original_failure'],'OCB',['real_history','autoregressive'],[1,4,8,16]):
            rr=[r for r in aux if (r['role'],r['letter'],r['mode'],r['horizon'])==(role,letter,mode,h)]
            if rr:
                auxiliary.append(dict(label=label,role=role,letter=letter,mode=mode,horizon=h,n=len(rr),
                    background_MSE=avg([r['background_MSE'] for r in rr]),
                    RGB_outside_01_fraction=avg([r['RGB_outside_01_fraction'] for r in rr]),
                    gray_empty=sum(r['gray_empty'] for r in rr),blue_empty=sum(r['blue_empty'] for r in rr)))
    audit=dict(run_id=c['run_id'],config_sha256=digest(RUN/'frozen-config.json'),test_spec_sha256=digest(HERE/'TEST_SPEC.txt'),
        owner_completion=done,missing_evaluations=missing,source_score_rows=sum(len(d['rows']) for d in data.values()),groups=len(curves),
        source_means_compared=means_compared,canonical_observed_noise_evidence_equal=True,native_rng_step_pairs_checked=rng_pairs,
        unique_training_target_indices_each=len({i for b in read(RUN/'schedule.json')['native_h1_batches'] for i in b}),
        calibration_coefficients_recomputed_from_saved_parameter_norms=independently_calculated,preflight=pf,
        training=training,full_restores=restore,source_checks=source_checks,charged_image_forwards=charged,completed_image_forwards=completed,
        absolute_acceptance={label:{role:v['all_passed'] for role,v in d['support'].items()} for label,d in data.items()},
        predeclared_support=support,definition_diagnostics=diagnostics,
        scope='Independent reaggregation of saved original scores/pairing/calibration records, not a full raw rescore or new gradient measurement.',
        models_called_by_this_analysis=0,training_or_physics_by_this_analysis=0)
    save('INDEPENDENT_ALL16.json',curves);save('PAIRED_CASES.json',paired);save('INDEPENDENT_CHECKS.json',support)
    save('DIAGNOSTIC_AGGREGATE.json',diag_aggregate);save('AUXILIARY_RGB_RISKS.json',auxiliary);save('AUDIT.json',audit)
    print(json.dumps(dict(run_id=c['run_id'],labels=list(data),source_score_rows=audit['source_score_rows'],groups=len(curves),
        calibrated_units=independently_calculated,native_pairs=rng_pairs,charged_NFE=charged,completed_NFE=completed,
        validation_AR16=[{k:r[k] for k in ['label','letter','IoU','position','pusher_position','topology','shape']} for r in curves if r['role']=='seen_val' and r['mode']=='autoregressive' and r['horizon']==16],
        directional_support=[{k:r[k] for k in ['arm','letter','predeclared_directional_support']} for r in support]),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
