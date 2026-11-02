"""Independent source-score audit of LOSS-AR4-R1; no model/Torch import."""
import collections
import hashlib
import itertools
import json
import math
import pathlib
import statistics

P=pathlib.Path(__file__).resolve().parent
ART=P.parent.parent
RUN=ART/'loss-rollout-ab-20261102-r1'
LABELS=['E2','A128','A256','B128','B256']
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def save(name,value):(P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def avg(v):
    v=[x for x in v if isinstance(x,(float,int)) and math.isfinite(x)]
    return statistics.mean(v) if v else None
def close(x,y):return x is None and y is None or x is not None and y is not None and abs(x-y)<1e-10

c=read(RUN/'config.json');done=read(RUN/'completion.json')
config_sha=hashlib.sha256((RUN/'config.json').read_bytes()).hexdigest()
assert config_sha=='f652a7cc5963f4dc43bb46d8bcc38f2a5fdbaa99801f0ddf9be851311cf58293'
data={label:read(RUN/f'evaluation/{label}/metrics.json') for label in LABELS}
assert all(len(d['rows'])==1248 and len(d['raw_files'])==78 for d in data.values())
assert all(d['evidence']==data['E2']['evidence'] for d in data.values())
groups=collections.defaultdict(list)
for label,d in data.items():
    keys=set()
    for r in d['rows']:
        key=(r['sample_id'],r['input_mode'],r['horizon'])
        assert key not in keys;keys.add(key)
        groups[(label,r['selection_role'],r['letter'],r['input_mode'],r['horizon'])].append(r)
        if r['input_mode']=='real_history' and r['strong_H1']:
            groups[(label,r['selection_role'],r['letter'],'real_history','strong_H1')].append(r)
rows=[]
for (label,role,letter,mode,h),rr in groups.items():
    rows.append(dict(label=label,role=role,letter=letter,mode=mode,horizon=h,n=len(rr),
        IoU=avg([r['metrics']['raw_mask_iou'] for r in rr]),
        position=avg([r['metrics']['object_position_error_image_px'] for r in rr]),
        pusher_position=avg([r['metrics']['pusher_position_error_image_px'] for r in rr]),
        topology=dict(collections.Counter(r['visible_topology_status'] for r in rr)),
        shape=dict(collections.Counter(r['visible_shape_status'] for r in rr)),
        rotation=dict(collections.Counter(r['rotation']['status'] for r in rr))))
own={(r['label'],r['role'],r['letter'],r['mode'],r['horizon']):r for r in rows}
compared=0
for r in read(RUN/'report/grouped-all16.json'):
    key=(r['label'],r['role'],r['letter'],r['mode'],r['horizon'])
    for metric in ['IoU','position','pusher_position']:
        assert close(own[key][metric],r[metric]),(key,metric)
        compared+=1
    for metric in ['topology','shape','rotation']:assert own[key][metric]==r[metric]
save('INDEPENDENT_ALL16.json',rows)

paired=[]
for role,letter,mode,h,before in itertools.product(['train','seen_val','original_failure'],'OCB',['real_history','autoregressive'],[1,4,8,16],['E2','A256']):
    initial=groups[(before,role,letter,mode,h)]
    final={r['sample_id']:r for r in groups[('B256',role,letter,mode,h)]}
    case=[]
    for r in initial:
        f=final[r['sample_id']]
        case.append(dict(sample_id=r['sample_id'],family_id=r['family_id'],
                         delta_IoU=f['metrics']['raw_mask_iou']-r['metrics']['raw_mask_iou'],
                         delta_position=f['metrics']['object_position_error_image_px']-r['metrics']['object_position_error_image_px']))
    paired.append(dict(role=role,letter=letter,mode=mode,horizon=h,before=before,after='B256',cases=case,
                       iou_better=sum(r['delta_IoU']>1e-12 for r in case),iou_worse=sum(r['delta_IoU']< -1e-12 for r in case),
                       position_better=sum(r['delta_position']< -1e-12 for r in case),position_worse=sum(r['delta_position']>1e-12 for r in case)))
save('PAIRED_CASES.json',paired)

checks=[]
for letter in 'OCB':
    A=own[('A256','seen_val',letter,'autoregressive',4)]
    B=own[('B256','seen_val',letter,'autoregressive',4)]
    RA=own[('A256','seen_val',letter,'real_history','strong_H1')]
    RB=own[('B256','seen_val',letter,'real_history','strong_H1')]
    conditions=dict(AR4_IoU_improved=B['IoU']>A['IoU'],AR4_position_not_worse=B['position']<=A['position'],
                    RH_strong_IoU_not_worse=RB['IoU']>=RA['IoU'],RH_strong_position_not_worse=RB['position']<=RA['position'])
    topology=[]
    for h in [1,4,8,16]:
        af=own[('A256','seen_val',letter,'autoregressive',h)]['topology'].get('fail',0)
        bf=own[('B256','seen_val',letter,'autoregressive',h)]['topology'].get('fail',0)
        topology.append(dict(h=h,A_failures=af,B_failures=bf,no_more_failures=bf<=af))
    supported=all(conditions.values()) and all(r['no_more_failures'] for r in topology)
    supplied=next(r for r in read(RUN/'report/comparison-checks.json') if r['letter']==letter)
    assert all(supplied[k]==v for k,v in conditions.items())
    assert supplied['numeric_directional_support']==supported
    checks.append(dict(letter=letter,conditions=conditions,topology=topology,numeric_directional_support=supported))
save('PRESET_CHECKS.json',checks)

# Compare the fresh common starting evaluation with original saved HMF E2 scores.
original=read(ART/'high-motion-fit-20261101/evaluation/0/primary/metrics.json')
old={(r['sample_id'],r['input_mode'],r['horizon']):r for r in original['rows']}
baseline_compared=0;maxdiff=0
for r in data['E2']['rows']:
    o=old[(r['sample_id'],r['input_mode'],r['horizon'])]
    for m in ['raw_mask_iou','object_position_error_image_px','raw_contour_error_image_px']:
        diff=abs(r['metrics'][m]-o['metrics'][m]);maxdiff=max(diff,maxdiff)
        assert diff<1e-10;baseline_compared+=1

logs={arm:read(RUN/f'training/{arm}-log.json') for arm in ['A','B']}
assert len(logs['A'])==len(logs['B'])==256
assert all(a['branches']==b['branches'] and a['conditioning_rng_hashes']==b['conditioning_rng_hashes'] for a,b in zip(logs['A'],logs['B']))
assert all(r['scaler']['applied']==1 for v in logs.values() for r in v)
restore={arm:read(RUN/f'restore-{arm}.json') for arm in ['A','B']}
assert all(all(x.values()) for x in restore.values())
assert read(RUN/'preflight.json')['passed']
counts=done['ledger']['counts']
assert all(v['images_attempted']==v['images_completed'] and v['batches_attempted']==v['batches_completed'] for v in counts.values())
total=sum(v['images_attempted'] for v in counts.values());assert total==33101
assert done['ledger']['successful_updates']=={'A':256,'B':256}
assert done['ledger']['active'] is None
absolute={label:{role:d['support'][role]['all_passed'] for role in d['support']} for label,d in data.items()}
assert all(not x for v in absolute.values() for x in v.values())
source_checks=[]
for binding in c['bindings']:
    f=pathlib.Path(binding['path'])
    if f.parent==RUN and f.suffix=='.py':
        actual=hashlib.sha256(f.read_bytes()).hexdigest();assert actual==binding['sha256']
        source_checks.append(dict(path=str(f),sha256=actual))
audit=dict(run_id=c['run_id'],config_sha256=config_sha,source_score_rows=6240,groups=len(rows),
           mean_values_compared=compared,matching_group_status_counts=True,
           evaluation_evidence_groups_per_checkpoint=len(data['E2']['evidence']),all_five_evaluations_paired=True,
           original_HMF_E2_metric_values_compared=baseline_compared,baseline_max_abs_difference=maxdiff,
           native_training_pairs_compared=256,full_restores=restore,source_checks=source_checks,
           all_absolute_acceptance=absolute,preset_support=checks,
           image_forwards_this_revision=total,prior_failed_preflight_forwards=33,cumulative_image_forwards=total+33,
           ledger=done['ledger'],worker_seconds=done['wall_seconds'],CPU=done['CPU'],own_CPU_seconds=done['own_CPU_seconds'],
           peak_cuda=done['results']['peak_cuda'],storage_at_completion=done['storage'],
           training_seconds={arm:done['results'][arm]['training_seconds'] for arm in ['A','B']},
           whole_run_failed=done['failed'],failure=done['failure'],
           owner_report_txt_exists=(RUN/'REPORT.txt').exists(),owner_html_exists=(RUN/'report/index.html').exists(),
           audit_scope='Saved source metrics and paired logs; no full 390-archive hash or image re-scoring. Separate plotting script checks nine old-failure predictions only.',
           audit_model_calls=0,audit_training=0,audit_physics=0)
save('AUDIT.json',audit)
print(json.dumps(dict(groups=len(rows),means_compared=compared,baseline_compared=baseline_compared,baseline_maxdiff=maxdiff,
    final_AR16=[{k:r[k] for k in ['label','role','letter','IoU','position']} for r in rows if r['label'] in ['A256','B256'] and r['mode']=='autoregressive' and r['horizon']==16],
    checks=checks),ensure_ascii=False,indent=2))
