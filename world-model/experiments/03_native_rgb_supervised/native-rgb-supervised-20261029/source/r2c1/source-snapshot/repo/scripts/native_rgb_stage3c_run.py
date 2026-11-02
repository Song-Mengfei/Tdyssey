"""One explicitly authorized exact continuation from3B129 to cumulative250."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tdyssey.native_rgb_training import atomic_json, sha256
from scripts.native_rgb_stage3b_run import prediction_hook, verify_frozen

PARENT_SHA = 'e94f9349e1affb8372aacc4613bd57ade496704a50b63d257f77cb4c5ad3626a'
PHASE = 'stage3c-visible-microfit-continuation'


def authorize(output):
    parent = Path(output).resolve().parent
    (parent/'SUPERVISION.md').read_text(encoding='utf-8')
    state = json.loads((parent/'stage-state.json').read_text(encoding='utf-8'))
    if state['current_stage'] != '3C' or state['authorized_stages'] != ['3C'] or state['status'] != 'in_progress' or state['callback']['sent']:
        raise RuntimeError('Only the newly released single3C continuation may execute')
    if sha256(Path(output)/'PARENT_REVIEW.md') != state['stage3c_parent_release_sha256']:
        raise ValueError('3C release identity changed')


def validate_prior_ledger(value, prior):
    if value != prior or value['active_stage'] is not None or value['charged_updates'] != 129 or value['completed_updates'] != 129:
        raise ValueError('Exact closed129/129 prior ledger required; never reset or refund')
    if value['charged_seconds'] != 444.6250000000073 or len(value['stages']) != 2:
        raise ValueError('The conserved3+3B444.625 fee and both historical stages are required')
    if value['stages'][0]['observed_seconds'] != 28.3440000000046 or value['stages'][1]['observed_seconds'] != 416.2810000000027:
        raise ValueError('Prior stage fees changed')
    if value['max_updates'] != 3000 or value['max_seconds'] != 5400:
        raise ValueError('Aggregate3000/5400 must remain fixed')
    return dict(max_attempts=121, max_seconds=600., update_cutoff_seconds=420., tail_reserve_seconds=180.)


def compose_comparisons(output):
    """CPU only; reference the existing original baseline, never infer it again."""
    import numpy as np
    output = Path(output); result = {}; (output/'predictions').mkdir(exist_ok=True)
    for cohort in ['train18', 'dev36']:
        with np.load(output/'truth'/f'{cohort}.npz', allow_pickle=False) as f: truth = {k:f[k] for k in f.files}
        arrays = {k:truth[k] for k in ['last', 'true', 'hold', 'last_full_mask', 'true_full_mask', 'window_ids']}
        for phase in ['frozen', 'fitted']:
            path = (output.parent/'stage3b' if phase == 'frozen' else output)/'predictions'/f'{phase}-{cohort}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as f:
                    if not np.array_equal(f['window_ids'], truth['window_ids']): raise ValueError('Raw prediction IDs differ')
                    arrays[phase] = f['prediction']; arrays[phase+'_status'] = f['status']
            else:
                arrays[phase] = np.full_like(truth['last'], np.nan)
                arrays[phase+'_status'] = np.full(len(truth['last']), 'not_measured_budget_or_failure', dtype='<U512')
        path = output/'predictions'/f'{cohort}.npz'
        with path.open('xb') as stream: np.savez_compressed(stream, **arrays)
        result[cohort] = dict(file=str(path), sha256=sha256(path), windows=len(truth['last']), original_baseline_referenced=True)
    atomic_json(output/'comparison-arrays.json', result)


def final_prediction_hook(output, truth, plans):
    native = prediction_hook(output, 'fitted', truth, plans)
    def callback(context):
        if context['progress']['updates'] <= 129:
            summary = dict(skipped=True,reason='No new complete update; update129 must not be predicted again',
                           native_sampler_calls=0,native_prediction_frames=0)
            atomic_json(Path(output)/'final-prediction-skipped.json',summary)
            return summary
        return native(context)
    return callback


def run(output, config):
    import numpy as np
    output = Path(output).resolve(); authorize(output); freeze = verify_frozen(output)
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    plan = json.loads((output/'continuation-plan.json').read_text(encoding='utf-8'))
    parent = json.loads((output/'parent-expected.json').read_text(encoding='utf-8'))
    windows = json.loads((output/'frozen-windows.json').read_text(encoding='utf-8'))
    plans = json.loads((output/'paired-batches.json').read_text(encoding='utf-8'))
    if (cfg['continuation_target_updates'] != 250 or cfg['continuation_start_updates'] != 129 or cfg['stage_max_updates'] != 121
        or cfg['stage_max_model_work_seconds'] != 600 or cfg['no_new_update_after_seconds'] != 420 or plan['phase'] != PHASE
        or cfg['parent_checkpoint_sha256'] != PARENT_SHA or parent['checkpoint_sha256'] != PARENT_SHA
        or cfg['train_indices'] != [r['dataset_index'] for r in windows['train']]
        or cfg['validation_indices'] != [r['dataset_index'] for r in windows['dev']]):
        raise ValueError('Exact3C continuation target/caps/18+36 and published parent boundary required')
    if (output/'MODEL_WORK_STARTED.json').exists(): raise FileExistsError('3C one execution only; no automatic replay')
    truth = {}
    for name in ['train18', 'dev36']:
        with np.load(output/'truth'/f'{name}.npz', allow_pickle=False) as f: truth[name] = {k:f[k] for k in f.files}
        if not np.array_equal(truth[name]['last'], truth[name]['histories'][:,3]): raise ValueError('Saved image timing differs')
    from tdyssey.native_rgb_stage_budget import ModelWorkBudget, StorageBudget, native_storage_paths
    from tdyssey.native_rgb_training import CostLedger, load_checkpoint_payload
    from tdyssey.native_rgb_stage3c_continuation import train_continuation
    ledger = CostLedger(cfg['ledger_path'], 3000, 5400., legacy_cost=dict(historical_native_attempts=1899,historical_native_model_seconds=5352.516,included_in_new_round_budget=False))
    prior = json.loads((output/'prior-ledger.json').read_text(encoding='utf-8'))
    if sha256(output/'prior-ledger.json') != cfg['prior_ledger_sha256']: raise ValueError('Frozen prior ledger changed')
    limits = validate_prior_ledger(ledger.snapshot(), prior)
    if json.loads(Path(cfg['prior_model_work_file']).read_text(encoding='utf-8'))['budget']['ledger'] != prior:
        raise ValueError('Previous model work and live ledger differ')
    source_additions = {str(Path(row['source']).relative_to(ROOT)).replace('\\','/'): row['sha256']
                        for row in freeze['files'] if Path(row['source']).is_relative_to(ROOT)
                        and str(Path(row['source']).relative_to(ROOT)).replace('\\','/') in plan['source_files']}
    if set(source_additions) != set(plan['source_files']): raise ValueError('All new runtime/config identities must be genuine frozen sources')
    storage = StorageBudget(native_storage_paths(ROOT,output.parent,extra_paths=[cfg['gpc_source_root'],cfg['physics_source_root'],cfg['init_checkpoint']]),
                            max_bytes=3*1024**3,atomic_reserve_bytes=256*1024**2,minimum_free_bytes=512*1024**2)
    before_storage = storage.check(force=True)
    atomic_json(output/'MODEL_WORK_STARTED.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha256(config),
        source_identity_sha256=sha256(output/'source-identity.json'),initial_storage=before_storage,no_automatic_retry=True,parent_checkpoint_sha256=PARENT_SHA))
    raw_stage = ledger.start(PHASE); budget = ModelWorkBudget(raw_stage,storage=storage,**limits)
    result = failure = checkpoint_review = None
    peak_allocated = peak_reserved = None
    try:
        budget.require_operation(60., 'full model construction and exact129 continuation restore')
        import torch
        if not torch.cuda.is_available() or '3060 Ti' not in torch.cuda.get_device_name(0): raise RuntimeError('Existing3060Ti only')
        atomic_json(output/'timed-device.json',dict(name=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                    model_work_elapsed=budget.elapsed,device_query_inside_model_clock=True))
        result = train_continuation(cfg,cfg['output'],cfg['joint_data'],cfg['gpc_source_root'],cfg['init_checkpoint'],
            parent_spec=parent,allowed_config_changes=plan['config_changes'],source_additions=source_additions,
            review_path=output/'restore-review.json',resume=cfg['parent_checkpoint'],work_budget=budget,execute=True,phase=PHASE,updates=250,
            physics_source_root=cfg['physics_source_root'],indices=cfg['train_indices'],validation_indices=cfg['validation_indices'],
            checkpoint_policy='last_only',before_updates_callback=None,after_updates_callback=final_prediction_hook(output,truth,plans))
        restore_review = json.loads((output/'restore-review.json').read_text(encoding='utf-8'))
        peak_allocated = max([v for v in [result.get('peak_cuda_allocated'),restore_review.get('restore_boundary_peak_cuda_allocated')] if v is not None],default=None)
        peak_reserved = max([v for v in [result.get('peak_cuda_reserved'),restore_review.get('restore_boundary_peak_cuda_reserved')] if v is not None],default=None)
        path = Path(cfg['output'])/'last.pt'
        if not path.exists(): raise RuntimeError('No complete continuation checkpoint')
        budget.require_operation(15., 'complete final CPU checkpoint semantic verification')
        began = time.monotonic(); payload = load_checkpoint_payload(path)
        required = {'model','optimizer','scaler','scheduler','rng','order','progress','config','identity','ledger'}
        if not required.issubset(payload) or payload['config'] != cfg or payload['identity'] != result['identity']:
            raise ValueError('Final checkpoint fields/current truthful identity differ')
        if not 129 <= payload['progress']['updates'] <= 250 or payload['scheduler']['T_max'] != 3000 or payload['order']['cursor'] != payload['progress']['samples']:
            raise ValueError('Cumulative final checkpoint state differs')
        checkpoint_review = dict(passed=True,required_fields=sorted(required),checkpoint_sha256=sha256(path),updates=payload['progress']['updates'],
            optimizer_parameter_state_entries=len(payload['optimizer']['state']),scheduler_T_max=payload['scheduler']['T_max'],
            order_cursor=payload['order']['cursor'],verification_wall_seconds=time.monotonic()-began,model_work_elapsed=budget.elapsed,CPU_torch_load_counted=True)
        del payload; budget.heartbeat()
    except BaseException as error:
        failure = type(error).__name__+': '+str(error)
        atomic_json(output/'model-work-error.json',dict(error=failure,model_work_elapsed=budget.elapsed,no_retry=True))
    finally:
        budget.finish('failed:'+failure if failure else 'complete' if result and result.get('status')=='complete' else 'bounded_stop')
        atomic_json(output/'model-work.json',dict(failure=failure,training_result=result,checkpoint_review=checkpoint_review,budget=budget.report(),
            parent_checkpoint_sha256=PARENT_SHA,continuation_start_updates=129,cumulative_target=250,physical_controls=0,no_unseen=True,no_H4=True,no_full319=True,one_run_only=True))
        final_work=json.loads((output/'model-work.json').read_text(encoding='utf-8'))
        final_work.update(peak_cuda_allocated=peak_allocated,peak_cuda_reserved=peak_reserved,
                          peak_cuda_scope='max of restoration/boundary counters and core training/native prediction counters')
        atomic_json(output/'model-work.json',final_work)
    compose_comparisons(output)
    return dict(failure=failure,status=None if result is None else result['status'],budget=budget.report())


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/native-rgb-supervised-20261029/stage3c')
    p.add_argument('--config',type=Path,default=ROOT/'configs/native_rgb_supervised_stage3c.json');a=p.parse_args()
    if not a.execute: raise SystemExit('Explicit --execute required; no GPU smoke/retry')
    r=run(a.output,a.config);print(json.dumps(dict(failure=r['failure'],status=r['status'],model_work_seconds=r['budget']['elapsed_model_work_seconds'],
        local_attempts=r['budget']['ledger']['stages'][-1]['reserved_updates'])));raise SystemExit(1 if r['failure'] else 0)
