"""Finite parent-released remaining TRAIN batch; stop after evidence and callback."""
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
from scripts.native_rgb_supervised_stage1 import StageLedger,storage_check,verify_source
from scripts.native_rgb_stage2a import LETTERS,LOCATIONS,run_slot,audit as audit_batch

EXTRAS={
 'A':['center','left','right','top'], 'B':['bottom','left-top','right-top','left-bottom'],
 'C':['right-bottom','center','left','right'], 'E':['top','bottom','left-top','right-top'],
 'F':['left-bottom','right-bottom','center','left'], 'H':['right','top','bottom','center'],
 'I':['right-top','left-bottom','right-bottom','center'], 'J':['left','right','top','bottom'],
 'L':['left-top','right-top','left-bottom','right-bottom'], 'M':['left-top','left','left-bottom','top'],
 'O':['bottom','left-top','right-top','left-bottom'], 'P':['right-bottom','center','left','right'],
 'S':['top','bottom','left-top','right-top'], 'T':['left-bottom','right-bottom','center','left'],
 'U':['right','top','bottom','left-top'], 'V':['right-top','right','right-bottom','center'],
 'X':['left','right','top','bottom'], 'Y':['left-top','right-top','left-bottom','right-bottom']}


def authorize(output):
    parent=Path(output).resolve().parent
    (parent/'SUPERVISION.md').read_text(encoding='utf-8')
    state=json.loads((parent/'stage-state.json').read_text())
    if (state['current_stage']!='2B' or state['authorized_stages']!=['2B'] or
        state['status']!='in_progress' or state['callback']['sent'] or not state['authorization']['verified']):
        raise RuntimeError('Only currently released2B may run; no2C or models')
    return parent,state


def load_config(path):
    cfg=json.loads(Path(path).read_text())
    from tdyssey.native_rgb_env import PROTOCOL_SHA256
    if (tuple(cfg['letters'])!=LETTERS or tuple(cfg['locations'])!=LOCATIONS or cfg['extra_locations']!=EXTRAS
        or cfg['stage']!='2B' or cfg['split']!='train' or cfg['max_execution_slots']!=234
        or cfg['max_attempted_controls']!=14976 or cfg['controls_per_trajectory']!=64
        or cfg['model_calls_allowed']!=0 or cfg['optimizer_updates_allowed']!=0
        or cfg['validation_collection_allowed']!=0 or cfg['environment']['protocol_sha256']!=PROTOCOL_SHA256
        or cfg['previous_supervised_controls']!=14336 or cfg['planned_supervised_controls_after']!=29312):
        raise ValueError('Exact released2B scope/protocol cannot be enlarged')
    return cfg


def initial_attempts(stage2a):
    manifest=json.loads((Path(stage2a)/'data/manifest.json').read_text())
    result={letter:{} for letter in LETTERS};seen=set()
    for row in manifest['entries']:
        if row['controls']<=0:continue
        if row['trajectory_id'] in seen:raise ValueError('Duplicated2A attempted trajectory')
        seen.add(row['trajectory_id']);arc=int(row['selected_initial_arc'])
        counts=result[row['shape']];counts[arc]=counts.get(arc,0)+1
    if len(seen)!=162:raise ValueError('The reviewed2A baseline must have162 started trajectories')
    return result


def prepare(output,config):
    output=Path(output);parent,_=authorize(output);cfg=load_config(config);a=parent/'stage2a'
    if (output/'frozen-requests.json').exists():raise FileExistsError('Never replace a frozen plan')
    old=json.loads((a/'dedup-reference.json').read_text());hashes=set(old['initial_state_hashes']);seeds=set(old['historical_seeds'])
    sources=list(old['sources']);manifest=json.loads((a/'data/manifest.json').read_text())
    for row in manifest['entries']:
        for key,hkey in [('file','sha256'),('metadata_file','metadata_sha256')]:
            path=a/'data'/row[key]
            if sha256(path)!=row[hkey]:raise ValueError('Reviewed2A file identity changed')
        if row['seed'] in seeds or row['initial_scene_hash'] in hashes:raise ValueError('2A identity collision')
        seeds.add(row['seed']);hashes.add(row['initial_scene_hash'])
    sources.append(dict(path=str((a/'data/manifest.json').resolve()),sha256=sha256(a/'data/manifest.json'),use='all2A identities and original file hashes'))
    old_requests=json.loads((a/'frozen-requests.json').read_text())['requests']
    lookup={(q['letter'],q['spatial_class']):q for q in old_requests}
    requests=[]
    for group in ['base9','extra4']:
        for index in range(9 if group=='base9' else 4):
            for letter in LETTERS:
                location=LOCATIONS[index] if group=='base9' else EXTRAS[letter][index]
                oldq=lookup[letter,location];shift=1 if group=='base9' else 2
                tier=None if location=='center' else (cfg['sampling']['wall_gap_representatives_world_px'].index(oldq['gap_world_px'])+shift)%3
                seed=cfg['seed_base']+len(requests)
                if seed in seeds:raise ValueError('Frozen seed collides; no silent replacement')
                seeds.add(seed)
                requests.append(dict(request_id=f'train-2b-{len(requests):03d}',stage='2B',letter=letter,
                    spatial_class=location,location=location,seed=seed,split='train',controls=64,quota_kind=group,
                    gap_tier_shift_from_2a=shift,gap_band=None if tier is None else cfg['sampling']['wall_gap_bands_world_px'][tier],
                    gap_world_px=None if tier is None else cfg['sampling']['wall_gap_representatives_world_px'][tier],sampling=cfg['sampling']))
    assert len(requests)==234 and Counter(q['letter'] for q in requests)==Counter({l:13 for l in LETTERS})
    assert Counter(q['spatial_class'] for q in requests)==Counter({l:26 for l in LOCATIONS})
    atomic_json(output/'dedup-reference.json',dict(kind='stage2b-identity-only-dedup.v1',sources=sources,
        initial_state_hashes=sorted(hashes),historical_seeds=sorted(seeds-set(q['seed'] for q in requests)),
        validation_use='identity-only exclusion; never selection/performance feedback'))
    atomic_json(output/'initial-attempts-2a.json',dict(scope='actualstartedformal2A initialarc once per trajectory',counts=initial_attempts(a)))
    result=dict(kind='stage2b-frozen-formal-train-requests.v1',requests=requests,split='train',maximum_slots=234,
        maximum_controls=14976,previous_supervised_controls=14336,planned_supervised_controls=29312,
        historical_plus_planned_native_controls=63872,no_validation_or_probe_records_in_formal_train=True,
        config_sha256=sha256(config),dedup_reference_sha256=sha256(output/'dedup-reference.json'),examples=cfg['examples'])
    atomic_json(output/'frozen-requests.json',result)
    return dict(requests=len(requests),controls=14976)


def freeze(output,config,physics_root):
    output=Path(output);cfg=load_config(config);a=output.parent/'stage2a'
    files=['scripts/native_rgb_stage2b.py','scripts/native_rgb_stage2b_report.py','scripts/native_rgb_stage2b_coverage.py',
        'scripts/native_rgb_stage2a.py','scripts/native_rgb_stage2a_report.py','scripts/native_rgb_stage2a_coverage.py',
        'scripts/native_rgb_supervised_stage1.py','tdyssey/native_rgb_supervised_sampling.py','tdyssey/native_rgb_motion_statistics.py',
        'tdyssey/native_rgb_env.py','tdyssey/native_rgb_data.py','tdyssey/native_rgb_training.py','tdyssey/envs/official.py',
        'tdyssey/geometry.py','tdyssey/letters.py','tdyssey/collection.py','tdyssey/state.py','tdyssey/official.py',
        'tdyssey/source_export.py','manifests/official_dp.json','tests/test_native_rgb_stage2b.py',
        'tests/test_native_rgb_supervised_planned_attempts.py','tests/test_native_rgb_stage2b_report.py']
    pairs=[(ROOT/p,p) for p in files]
    pairs.extend([(Path(config),'frozen-config.json'),(output/'frozen-requests.json','frozen-requests.json'),
        (output/'dedup-reference.json','dedup-reference.json'),(output/'initial-attempts-2a.json','initial-attempts-2a.json'),
        (a/'data/manifest.json','reviewed2A-manifest.json'),(a/'contact-coverage.json','reviewed2A-contact-coverage.json'),
        (ROOT/cfg['sampling']['prior_coverage'],'historical-TRAIN-contact-coverage.json')])
    pairs.extend((Path(physics_root)/p,'official-source/'+p) for p in ['diffusion_policy/env/pusht/pusht_env.py','diffusion_policy/env/pusht/pymunk_override.py'])
    rows=[]
    for source,name in pairs:
        target=output/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(source.read_bytes())
        rows.append(dict(source=str(source.resolve()),snapshot=target.relative_to(output).as_posix(),sha256=sha256(target),bytes=target.stat().st_size))
    atomic_json(output/'source-identity.json',dict(kind='stage2b-source-before-actual-controls.v1',files=rows,
        selection_version='motion-room-planned-attempts-v2',continuous_logic='frozen1B class unchanged',validation_feedback=False))
    return dict(frozen_files=len(rows))


class RemainingTrainingLedger(StageLedger):
    def begin_trajectory(self,trajectory_id,maximum_controls=64):
        if maximum_controls!=64 or len(self.value['trajectories'])>=234:raise RuntimeError('Released234 slots only')
        if any(r['status']=='running' for r in self.value['trajectories']):raise RuntimeError('No unresolved replay/replacement')
        self.value['trajectories'].append(dict(trajectory_id=trajectory_id,maximum_controls=64,charged_controls=0,
            completed_controls=0,status='running',pending_control=None));self.save()
        return len(self.value['trajectories'])-1


def collect(output,config,physics_root):
    output=Path(output).resolve();parent,state=authorize(output);cfg=load_config(config)
    from tdyssey.native_rgb_env import CONTROLLER,PROTOCOL_SHA256,STATE_COLUMNS
    if (output/'data/manifest.json').exists():raise FileExistsError('No replay/replacement of inspected batch')
    plan=json.loads((output/'frozen-requests.json').read_text());dedup=json.loads((output/'dedup-reference.json').read_text())
    if (len(plan['requests'])!=234 or plan['config_sha256']!=sha256(config)
        or plan['dedup_reference_sha256']!=sha256(output/'dedup-reference.json')):raise RuntimeError('Frozen plan changed')
    verify_source(output);storage_check(cfg,parent)
    coverage=json.loads((parent/'stage2a/contact-coverage.json').read_text())['segments']
    counts={l:{int(r['id']):int(r['contact_control_count']) for r in coverage[l]} for l in LETTERS}
    attempts={l:{int(k):int(v) for k,v in row.items()} for l,row in json.loads((output/'initial-attempts-2a.json').read_text())['counts'].items()}
    prior=json.loads((ROOT/cfg['sampling']['prior_coverage']).read_text())['segments']
    historical={l:{int(r['id']):len(r['trajectory_ids']) for r in prior[l]} for l in LETTERS}
    seen=set(dedup['initial_state_hashes']);entries=[];failed=[];failure=None;ledger=None;started=time.monotonic()
    lock=output/'COLLECTION_RUNNING.lock'
    with lock.open('x') as stream:stream.write('2B')
    try:
        ledger=RemainingTrainingLedger(output,cap=14976)
        if ledger.value['trajectories'] or ledger.value['charged_controls']:raise RuntimeError('No replay/failure replacement')
        ledger.value.update(stage='2B',split='train',maximum_execution_slots=234,previous_supervised_controls=14336,
            historical_native_controls=34560,historical_native_optimizer_attempts=1899,historical_native_training_seconds=5352.516)
        ledger.save()
        for request in plan['requests']:
            verify_source(output);storage_check(cfg,parent);l=request['letter']
            entry,error=run_slot(output,request,physics_root,ledger,coverage[l],counts[l],historical[l],seen,planned_attempts=attempts[l])
            if entry is not None:entries.append(entry)
            if error:failed.append(dict(request=request,error=error));failure=error
            atomic_json(output/'planned-attempts-current.json',dict(counts=attempts,scope='2A plus actuallystarted2B only'))
            atomic_json(output/'collection-progress.json',dict(executed_slots=len(ledger.value['trajectories']),planned_slots=234,
                charged_controls=ledger.value['charged_controls'],last_request=request['request_id'],failure=failure))
            print(json.dumps(dict(slot=len(ledger.value['trajectories']),letter=l,location=request['spatial_class'],
                charged_controls=ledger.value['charged_controls'],error=error)),flush=True)
            if failure:break
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        try:final_storage=storage_check(cfg,parent)
        except BaseException as error:
            final_storage=dict(error=type(error).__name__+': '+str(error));failure=failure or 'Final resource check: '+str(error)
        data=output/'data';data.mkdir(exist_ok=True)
        used=ledger.value['charged_controls'] if ledger else 0;actual=ledger.value['completed_controls'] if ledger else 0
        atomic_json(data/'manifest.json',dict(kind='native-rgb-supervised-formal-train-batch.v1',stage='2B',split='train',
            entries=entries,failure=failure,failed_requests=failed,planned_slots=234,
            unexecuted_request_ids=[q['request_id'] for q in plan['requests'][len(ledger.value['trajectories']) if ledger else 0:]],
            charged_controls=used,actual_controls=actual,saved_controls=sum(e['controls'] for e in entries),controller=CONTROLLER,
            protocol_sha256=PROTOCOL_SHA256,state_columns=list(STATE_COLUMNS),source_root=str(Path(physics_root).resolve()),
            frozen_requests_sha256=sha256(output/'frozen-requests.json'),source_identity_sha256=sha256(output/'source-identity.json'),
            no_probe_or_validation_entries=True,parent_release_required_before_training=True,model_calls=0,optimizer_updates=0))
        atomic_json(output/'contact-coverage.json',dict(segments=coverage,scope='2A+2B actualformalTRAIN collisions only; no probes/prior contacts'))
        atomic_json(output/'planned-attempts-current.json',dict(counts=attempts,scope='2A plus actuallystarted2B only'))
        if ledger:
            ledger.value.update(collection_wall_seconds=time.monotonic()-started,supervised_round_controls=14336+used,
                historical_plus_new_native_controls=48896+used,
                physics_substeps_lower_bound=ledger.value['physics_substeps']+ledger.value['recorded_partial_substeps_lower_bound'],
                charged_physics_substep_allowance=used*10,failure=failure,storage_after_collection=final_storage);ledger.save()
        lock.unlink(missing_ok=True)
    return dict(failure=failure,charged_controls=used,actual_controls=actual,executed_slots=len(ledger.value['trajectories']) if ledger else 0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','freeze','collect','audit'])
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/native-rgb-supervised-20261029/stage2b')
    p.add_argument('--config',type=Path,default=ROOT/'configs/native_rgb_supervised_stage2b.json')
    p.add_argument('--physics-root',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args(argv);authorize(a.output)
    if a.stage=='prepare':result=prepare(a.output,a.config)
    elif a.stage=='freeze':result=freeze(a.output,a.config,a.physics_root)
    elif a.stage=='audit':result=audit_batch(a.output,expected_slots=234,max_controls=14976,stage='2B')
    else:
        if not a.execute:raise RuntimeError('Physicalcollection requires --execute')
        result=collect(a.output,a.config,a.physics_root)
    print(json.dumps({k:v for k,v in result.items() if k in ['failure','passed','charged_controls','actual_controls','executed_slots','issues','requests','controls','frozen_files']},indent=2))
    return 1 if result.get('failure') or result.get('passed') is False else 0


if __name__=='__main__':raise SystemExit(main())
