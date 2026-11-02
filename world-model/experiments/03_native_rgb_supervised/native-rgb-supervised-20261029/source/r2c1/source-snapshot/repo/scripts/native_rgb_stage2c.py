"""Only independent seen_val collection; no model or optimizer invocation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_training import atomic_json,sha256
from scripts.native_rgb_supervised_stage1 import storage_check,verify_source
from scripts.native_rgb_stage2a import LETTERS,LOCATIONS,TrainingLedger,run_slot,audit as audit_batch

FROZEN_SAMPLER='8f84baa2a7a6f4a89ffb2191d09e4bf86acef211b25e6b80918169ea86fd8cd8'


def authorize(output):
    parent=Path(output).resolve().parent;(parent/'SUPERVISION.md').read_text(encoding='utf-8')
    state=json.loads((parent/'stage-state.json').read_text())
    if (state['current_stage']!='2C' or state['authorized_stages']!=['2C'] or state['status']!='in_progress'
        or state['callback']['sent'] or not state['authorization']['verified']):
        raise RuntimeError('Only the released2C sampling/CPU data preparation is allowed')
    return parent,state


def load_config(path):
    cfg=json.loads(Path(path).read_text())
    from tdyssey.native_rgb_env import PROTOCOL_SHA256
    if (tuple(cfg['letters'])!=LETTERS or tuple(cfg['locations'])!=LOCATIONS or cfg['stage']!='2C'
        or cfg['split']!='seen_val' or cfg['max_execution_slots']!=162 or cfg['max_attempted_controls']!=10368
        or cfg['controls_per_trajectory']!=64 or cfg['previous_supervised_controls']!=29312
        or cfg['planned_supervised_controls_after']!=39680 or cfg['model_calls_allowed']!=0
        or cfg['optimizer_updates_allowed']!=0 or cfg['validation_collection_allowed']!=162
        or cfg['TRAIN_collection_allowed']!=0 or cfg['environment']['protocol_sha256']!=PROTOCOL_SHA256
        or cfg['dynamic_counter_initialization']!='zero_seen_val_only'
        or sha256(ROOT/'tdyssey/native_rgb_supervised_sampling.py')!=FROZEN_SAMPLER):
        raise ValueError('Exact2C limits/sampler/physics cannot be enlarged or tuned')
    if sha256(ROOT/cfg['index_spec'])!=cfg['index_spec_sha256']:raise ValueError('Predeclared index rules changed')
    return cfg


def prepare(output,config):
    output=Path(output);parent,_=authorize(output);cfg=load_config(config);b=parent/'stage2b'
    if (output/'frozen-requests.json').exists():raise FileExistsError('Preserve first frozen seen_val requests')
    old=json.loads((b/'dedup-reference.json').read_text());hashes=set(old['initial_state_hashes']);seeds=set(old['historical_seeds'])
    sources=list(old['sources']);manifest=json.loads((b/'data/manifest.json').read_text())
    for e in manifest['entries']:
        for key,hkey in [('file','sha256'),('metadata_file','metadata_sha256')]:
            if sha256(b/'data'/e[key])!=e[hkey]:raise ValueError('Frozen2B original file changed')
        if e['seed'] in seeds or e['initial_scene_hash'] in hashes:raise ValueError('Reviewed2B duplicate identity')
        seeds.add(e['seed']);hashes.add(e['initial_scene_hash'])
    sources.extend([dict(path=str((b/'data/manifest.json').resolve()),sha256=sha256(b/'data/manifest.json'),use='identity-onlyexclusion'),
        dict(path=str((b/'report/merged-train-manifest.json').resolve()),sha256=sha256(b/'report/merged-train-manifest.json'),use='frozenTRAINreference;no validationfeedback')])
    requests=[]
    for location_index,location in enumerate(LOCATIONS):
        for letter_index,letter in enumerate(LETTERS):
            seed=cfg['seed_base']+location_index*1000+letter_index
            if seed in seeds:raise ValueError('Predeclared seed collision; no replacement')
            seeds.add(seed);tier=(location_index-1+letter_index+2)%3
            requests.append(dict(request_id=f'seen-val-2c-{len(requests):03d}',stage='2C',split='seen_val',
                letter=letter,spatial_class=location,location=location,seed=seed,controls=64,
                gap_band=None if location=='center' else cfg['sampling']['wall_gap_bands_world_px'][tier],
                gap_world_px=None if location=='center' else cfg['sampling']['wall_gap_representatives_world_px'][tier],
                sampling=cfg['sampling']))
    assert len(requests)==162 and Counter(q['letter'] for q in requests)==Counter({l:9 for l in LETTERS})
    for location in LOCATIONS[1:]:
        assert Counter(q['gap_world_px'] for q in requests if q['location']==location)==Counter({4:6,18:6,48:6})
    atomic_json(output/'dedup-reference.json',dict(kind='stage2c-identity-only-exclusion.v1',sources=sources,
        initial_state_hashes=sorted(hashes),historical_seeds=sorted(seeds-set(q['seed'] for q in requests)),
        validation_use='excludeidentitiesonly;dynamicvalidationcountersstartzero;TRAINnevermodified'))
    atomic_json(output/'frozen-requests.json',dict(kind='stage2c-frozen-seen-val-requests.v1',requests=requests,
        split='seen_val',maximum_slots=162,maximum_controls=10368,previous_supervised_controls=29312,
        planned_supervised_controls=39680,historical_plus_planned_native_controls=74240,
        config_sha256=sha256(config),dedup_reference_sha256=sha256(output/'dedup-reference.json'),
        index_spec_sha256=cfg['index_spec_sha256'],index_spec=cfg['index_spec'],examples=cfg['examples']))
    return dict(requests=162,controls=10368)


def freeze(output,config,physics_root):
    output=Path(output);cfg=load_config(config);parent=output.parent
    files=['scripts/native_rgb_stage2c.py','scripts/native_rgb_stage2c_indices.py','scripts/native_rgb_stage2c_report.py',
        'scripts/native_rgb_stage2c_coverage.py','scripts/native_rgb_stage2b_coverage.py',
        'scripts/native_rgb_stage2a.py','scripts/native_rgb_stage2a_report.py','scripts/native_rgb_stage2a_coverage.py',
        'scripts/native_rgb_supervised_stage1.py','tdyssey/native_rgb_reference.py','tdyssey/native_rgb_supervised_sampling.py',
        'tdyssey/native_rgb_motion_statistics.py','tdyssey/native_rgb_env.py','tdyssey/native_rgb_data.py','tdyssey/native_rgb_training.py',
        'tdyssey/envs/official.py','tdyssey/geometry.py','tdyssey/letters.py','tdyssey/collection.py','tdyssey/state.py',
        'tdyssey/official.py','tdyssey/source_export.py','manifests/official_dp.json','tests/test_native_rgb_stage2c.py',
        'tests/test_native_rgb_stage2c_indices.py','tests/test_native_rgb_reference.py',
        'tests/test_native_rgb_data.py','tests/test_native_rgb_stage2b.py','tests/test_native_rgb_supervised_planned_attempts.py']
    pairs=[(ROOT/p,p) for p in files]
    pairs.extend([(Path(config),'frozen-config.json'),(ROOT/cfg['index_spec'],'frozen-index-spec.json'),
        (output/'frozen-requests.json','frozen-requests.json'),(output/'dedup-reference.json','dedup-reference.json'),
        (ROOT/cfg['sampling']['prior_coverage'],'frozen-oldTRAIN-prior.json'),
        (parent/'stage2b/report/merged-train-manifest.json','frozen-TRAIN-view.json')])
    pairs.extend((Path(physics_root)/p,'official-source/'+p) for p in ['diffusion_policy/env/pusht/pusht_env.py','diffusion_policy/env/pusht/pymunk_override.py'])
    rows=[]
    for source,name in pairs:
        target=output/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(source.read_bytes())
        rows.append(dict(source=str(source.resolve()),snapshot=target.relative_to(output).as_posix(),sha256=sha256(target),bytes=target.stat().st_size))
    atomic_json(output/'source-identity.json',dict(kind='stage2c-source-and-index-rules-before-controls.v1',files=rows,
        selection_version='motion-room-planned-attempts-v2',sampler_sha256=FROZEN_SAMPLER,
        counter_scope='zeroed seen_val only; frozen historicalTRAINprior separate',no_model_work=True))
    return dict(frozen_files=len(rows))


def collect(output,config,physics_root):
    from tdyssey.native_rgb_env import CONTROLLER,PROTOCOL_SHA256,STATE_COLUMNS
    from tdyssey.native_rgb_data import contour_segments
    output=Path(output).resolve();parent,_=authorize(output);cfg=load_config(config)
    if (output/'data/manifest.json').exists():raise FileExistsError('No seen_val replay/replacement')
    plan=json.loads((output/'frozen-requests.json').read_text());dedup=json.loads((output/'dedup-reference.json').read_text())
    if (len(plan['requests'])!=162 or plan['config_sha256']!=sha256(config)
        or plan['dedup_reference_sha256']!=sha256(output/'dedup-reference.json') or plan['index_spec_sha256']!=cfg['index_spec_sha256']):
        raise RuntimeError('Frozen requests/config/index rules changed')
    verify_source(output);storage_check(cfg,parent)
    coverage={l:contour_segments(l,8) for l in LETTERS}
    for rows in coverage.values():
        for row in rows:row.update(contact_control_count=0,moving_contact_control_count=0,trajectory_contact_controls={})
    counts={l:{int(r['id']):0 for r in coverage[l]} for l in LETTERS};attempts={l:{} for l in LETTERS}
    prior=json.loads((ROOT/cfg['sampling']['prior_coverage']).read_text())['segments']
    historical={l:{int(r['id']):len(r['trajectory_ids']) for r in prior[l]} for l in LETTERS}
    atomic_json(output/'counter-baseline.json',dict(split='seen_val',actual_hit_counts=counts,planned_attempts=attempts,
        old_TRAIN_prior_sha256=sha256(ROOT/cfg['sampling']['prior_coverage']),TRAIN_dynamic_counts_inherited=False,
        validation_counters_may_flow_back_to_TRAIN=False))
    seen=set(dedup['initial_state_hashes']);entries=[];failed=[];failure=None;ledger=None;started=time.monotonic()
    lock=output/'COLLECTION_RUNNING.lock'
    with lock.open('x') as stream:stream.write('2C')
    try:
        ledger=TrainingLedger(output,cap=10368)
        if ledger.value['trajectories'] or ledger.value['charged_controls']:raise RuntimeError('No interrupted replay/replacement')
        ledger.value.update(stage='2C',split='seen_val',maximum_execution_slots=162,previous_supervised_controls=29312,
            historical_native_controls=34560,historical_native_optimizer_attempts=1899,historical_native_training_seconds=5352.516)
        ledger.save()
        for request in plan['requests']:
            verify_source(output);storage_check(cfg,parent);letter=request['letter']
            entry,error=run_slot(output,request,physics_root,ledger,coverage[letter],counts[letter],historical[letter],seen,planned_attempts=attempts[letter])
            if entry is not None:entries.append(entry)
            if error:failed.append(dict(request=request,error=error));failure=error
            atomic_json(output/'planned-attempts-current.json',dict(counts=attempts,scope='actuallystartedseen_valonly'))
            atomic_json(output/'collection-progress.json',dict(executed_slots=len(ledger.value['trajectories']),planned_slots=162,
                charged_controls=ledger.value['charged_controls'],last_request=request['request_id'],failure=failure))
            print(json.dumps(dict(slot=len(ledger.value['trajectories']),letter=letter,location=request['spatial_class'],
                charged_controls=ledger.value['charged_controls'],error=error)),flush=True)
            if failure:break
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        try:final_storage=storage_check(cfg,parent)
        except BaseException as error:
            final_storage=dict(error=type(error).__name__+': '+str(error));failure=failure or 'Final resourcecheck:'+str(error)
        data=output/'data';data.mkdir(exist_ok=True);used=ledger.value['charged_controls'] if ledger else 0;actual=ledger.value['completed_controls'] if ledger else 0
        atomic_json(data/'manifest.json',dict(kind='native-rgb-supervised-formal-validation-batch.v1',stage='2C',split='seen_val',
            entries=entries,failure=failure,failed_requests=failed,planned_slots=162,
            unexecuted_request_ids=[q['request_id'] for q in plan['requests'][len(ledger.value['trajectories']) if ledger else 0:]],
            charged_controls=used,actual_controls=actual,saved_controls=sum(e['controls'] for e in entries),controller=CONTROLLER,
            protocol_sha256=PROTOCOL_SHA256,state_columns=list(STATE_COLUMNS),source_root=str(Path(physics_root).resolve()),
            frozen_requests_sha256=sha256(output/'frozen-requests.json'),source_identity_sha256=sha256(output/'source-identity.json'),
            index_spec_sha256=cfg['index_spec_sha256'],no_probe_or_TRAIN_entries=True,parent_release_required_before_training=True,
            model_calls=0,optimizer_updates=0))
        atomic_json(output/'contact-coverage.json',dict(segments=coverage,scope='seen_valactualcollisiononly;zero baseline;no TRAINdynamic counts'))
        atomic_json(output/'planned-attempts-current.json',dict(counts=attempts,scope='actuallystartedseen_valonly'))
        if ledger:
            ledger.value.update(collection_wall_seconds=time.monotonic()-started,supervised_round_controls=29312+used,
                historical_plus_new_native_controls=63872+used,
                physics_substeps_lower_bound=ledger.value['physics_substeps']+ledger.value['recorded_partial_substeps_lower_bound'],
                charged_physics_substep_allowance=used*10,failure=failure,storage_after_collection=final_storage);ledger.save()
        lock.unlink(missing_ok=True)
    return dict(failure=failure,charged_controls=used,actual_controls=actual,executed_slots=len(ledger.value['trajectories']) if ledger else 0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','freeze','collect','audit'])
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/native-rgb-supervised-20261029/stage2c')
    p.add_argument('--config',type=Path,default=ROOT/'configs/native_rgb_supervised_stage2c.json')
    p.add_argument('--physics-root',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args(argv);authorize(a.output)
    if a.stage=='prepare':result=prepare(a.output,a.config)
    elif a.stage=='freeze':result=freeze(a.output,a.config,a.physics_root)
    elif a.stage=='audit':result=audit_batch(a.output,expected_slots=162,max_controls=10368,stage='2C',split='seen_val')
    else:
        if not a.execute:raise RuntimeError('Real validation collection requires --execute')
        result=collect(a.output,a.config,a.physics_root)
    print(json.dumps({k:v for k,v in result.items() if k in ['failure','passed','charged_controls','actual_controls','executed_slots','issues','requests','controls','frozen_files']},indent=2))
    return 1 if result.get('failure') or result.get('passed') is False else 0


if __name__=='__main__':raise SystemExit(main())
