"""Run after owner callback: audit saved scores/logs only; no Torch/model."""
import collections
import hashlib
import itertools
import json
import math
import pathlib
import statistics

P=pathlib.Path(__file__).resolve().parent
RUN=P.parent.parent/'loss-screen-20261102'
ARMS=['BASE','SNR','GDL','FFL']
def read(f):return json.loads(f.read_text(encoding='utf-8-sig'))
def save(n,v):(P/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def mean(v):
    v=[x for x in v if isinstance(x,(float,int)) and math.isfinite(x)]
    return statistics.mean(v) if v else None

c=read(RUN/'frozen-config.json')
original_done=read(RUN/'completion.json')
done=read(RUN/'continuation-completion.json') if (RUN/'continuation-completion.json').exists() else original_done
continuation=read(RUN/'continuation-config.json') if (RUN/'continuation-config.json').exists() else None
data={'E2':read(pathlib.Path(c['E2_metrics']))}
missing=[]
for arm in ARMS:
    path=RUN/f'evaluation/{arm}/metrics.json'
    if path.exists():data[arm]=read(path)
    else:missing.append(arm)
groups=collections.defaultdict(list)
for label,d in data.items():
    assert len(d['rows'])==1248 and len(d['raw_files'])==78
    assert d['evidence']==data['E2']['evidence']
    seen=set()
    for r in d['rows']:
        key=(r['sample_id'],r['input_mode'],r['horizon'])
        assert key not in seen;seen.add(key)
        groups[(label,r['selection_role'],r['letter'],r['input_mode'],r['horizon'])].append(r)
        if r['input_mode']=='real_history' and r['strong_H1']:
            groups[(label,r['selection_role'],r['letter'],'real_history','strong_H1')].append(r)
rows=[]
for (label,role,letter,mode,h),rr in groups.items():
    rows.append(dict(label=label,role=role,letter=letter,mode=mode,horizon=h,n=len(rr),
        IoU=mean([r['metrics']['raw_mask_iou'] for r in rr]),
        position=mean([r['metrics']['object_position_error_image_px'] for r in rr]),
        contour=mean([r['metrics']['raw_contour_error_image_px'] for r in rr]),
        pusher_position=mean([r['metrics']['pusher_position_error_image_px'] for r in rr]),
        topology=dict(collections.Counter(r['visible_topology_status'] for r in rr)),
        shape=dict(collections.Counter(r['visible_shape_status'] for r in rr)),
        rotation=dict(collections.Counter(r['rotation']['status'] for r in rr))))
own={(r['label'],r['role'],r['letter'],r['mode'],r['horizon']):r for r in rows}
save('INDEPENDENT_SCREEN_CURVES.json',rows)
means_compared=0
for r in read(RUN/'report/grouped-all16.json'):
    key=(r['label'],r['role'],r['letter'],r['mode'],r['horizon'])
    for m in ['IoU','position','pusher_position']:
        a,b=own[key][m],r[m]
        assert a is None and b is None or a is not None and b is not None and abs(a-b)<1e-10,(key,m,a,b)
        means_compared+=1
    for m in ['topology','shape','rotation']:assert own[key][m]==r[m],(key,m)
paired=[];checks=[]
for arm in ['SNR','GDL','FFL']:
    if arm not in data or 'BASE' not in data:continue
    for role,letter,mode,h in itertools.product(['train','seen_val','original_failure'],'OCB',['real_history','autoregressive'],[1,4,8,16]):
        before=groups[('BASE',role,letter,mode,h)]
        after={r['sample_id']:r for r in groups[(arm,role,letter,mode,h)]}
        cases=[]
        for r in before:
            a=after[r['sample_id']]
            cases.append(dict(sample_id=r['sample_id'],family_id=r['family_id'],
                              delta_IoU=a['metrics']['raw_mask_iou']-r['metrics']['raw_mask_iou'],
                              delta_position=a['metrics']['object_position_error_image_px']-r['metrics']['object_position_error_image_px']))
        paired.append(dict(arm=arm,role=role,letter=letter,mode=mode,horizon=h,cases=cases,
            IoU_better=sum(x['delta_IoU']>1e-12 for x in cases),IoU_worse=sum(x['delta_IoU']< -1e-12 for x in cases),
            position_better=sum(x['delta_position']< -1e-12 for x in cases),position_worse=sum(x['delta_position']>1e-12 for x in cases)))
    for letter in 'OCB':
        A=own[('BASE','seen_val',letter,'autoregressive',4)]
        B=own[(arm,'seen_val',letter,'autoregressive',4)]
        RA=own[('BASE','seen_val',letter,'real_history','strong_H1')]
        RB=own[(arm,'seen_val',letter,'real_history','strong_H1')]
        conditions=dict(AR4_IoU_improved=B['IoU']>A['IoU'],AR4_position_not_worse=B['position']<=A['position'],
            RH_strong_IoU_not_worse=RB['IoU']>=RA['IoU'],RH_strong_position_not_worse=RB['position']<=RA['position'])
        topology=[dict(h=h,BASE=own[('BASE','seen_val',letter,'autoregressive',h)]['topology'],
                        candidate=own[(arm,'seen_val',letter,'autoregressive',h)]['topology']) for h in [1,4,8,16]]
        top_ok=all(x['candidate'].get('fail',0)<=x['BASE'].get('fail',0) for x in topology)
        checks.append(dict(arm=arm,letter=letter,conditions=conditions,topology=topology,
            topology_no_worse=top_ok,predeclared_directional_support=all(conditions.values()) and top_ok))
for check in checks:
    original=next(r for r in read(RUN/'report/comparison-checks.json') if (r['arm'],r['letter'])==(check['arm'],check['letter']))
    assert original['numeric_directional_support']==check['predeclared_directional_support']
save('PAIRED_SCREEN_CASES.json',paired);save('INDEPENDENT_SCREEN_CHECKS.json',checks)

logs={};restores={};train_cost={}
for arm in ARMS:
    path=RUN/f'training/{arm}-log.json'
    if path.exists():logs[arm]=read(path)
    path=RUN/f'restore-{arm}.json'
    if path.exists():restores[arm]=read(path)
    path=RUN/f'training/{arm}-result.json'
    if path.exists():train_cost[arm]=read(path)
native_pairs=0
if 'BASE' in logs:
    for arm,log in logs.items():
        if arm=='BASE':continue
        if not missing:assert len(log)==len(logs['BASE'])==128
        for a,b in zip(logs['BASE'],log):
            assert a['branches']==b['branches'] and a['conditioning_rng_hashes']==b['conditioning_rng_hashes']
            native_pairs+=1
absolute={label:{role:d['support'][role]['all_passed'] for role in d['support']} for label,d in data.items()}
source_checks=[]
for binding in c['bindings']:
    f=pathlib.Path(binding['path'])
    if f.parent==RUN and f.suffix=='.py':
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        matches=digest==binding['sha256']
        source_checks.append(dict(path=str(f),sha256=digest,frozen_sha256=binding['sha256'],matches=matches))
schedule=read(RUN/'schedule.json')
unique_targets=len({i for batch in schedule['native_h1_batches'] for i in batch})
charged=sum(v['images_attempted'] for v in done['ledger']['counts'].values())
finished=sum(v['images_completed'] for v in done['ledger']['counts'].values())
assert charged<=c['caps']['model_image_forwards'] and charged==23220 and finished==23216
assert done['ledger']['updates']==dict.fromkeys(ARMS,128)
assert done['ledger']['exposures']==dict.fromkeys(ARMS,2048)
assert not done['ledger']['inflight'] and not done['ledger']['stack'] and done['ledger']['active'] is None
audit=dict(run_id=done.get('run_id',c['run_id']),config_sha256=hashlib.sha256((RUN/'frozen-config.json').read_bytes()).hexdigest(),
    continuation_config=continuation,continuation_config_sha256=hashlib.sha256((RUN/'continuation-config.json').read_bytes()).hexdigest() if continuation else None,
    missing_evaluations=missing,source_score_rows=sum(len(d['rows']) for d in data.values()),groups=len(rows),
    canonical_evaluation_evidence_values_equal=True,
    noise_provenance={a:d.get('noise_provenance','Original persisted observed-noise evidence') for a,d in data.items()},
    native_rng_step_pairs_checked=native_pairs,full_restores=restores,
    source_checks=source_checks,absolute_acceptance=absolute,predeclared_support=checks,
    whole_execution_completion=done,original_interrupted_completion=original_done,training_results=train_cost,
    source_metric_means_compared=means_compared,unique_training_target_indices_each=unique_targets,
    charged_image_forwards=charged,completed_image_forwards=finished,
    owner_saved_only_audit=read(RUN/'continuation-saved-audit.json'),
    BASE_cross_check=read(RUN/'BASE-vs-R1-A128.json') if (RUN/'BASE-vs-R1-A128.json').exists() else None,
    scope='Reaggregated saved source scores and pairing evidence; not full raw re-score/hash.',
    models_called_by_independent_review=0,training_by_independent_review=0,physics_by_independent_review=0)
save('SCREEN_AUDIT.json',audit)
print(json.dumps(dict(available=list(data),missing=missing,source_scores=audit['source_score_rows'],groups=len(rows),native_pairs=native_pairs,
    AR16_val=[{k:r[k] for k in ['label','letter','IoU','position']} for r in rows if r['role']=='seen_val' and r['mode']=='autoregressive' and r['horizon']==16],
    support=[{k:r[k] for k in ['arm','letter','predeclared_directional_support']} for r in checks]),ensure_ascii=False,indent=2))
