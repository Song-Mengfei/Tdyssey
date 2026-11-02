"""Fixed H1 conditioner comparison; no training, loss or physical operations."""
from copy import deepcopy
import gc
import hashlib
import json
import struct
import sys
import time

from .gpc_checkpoint_evaluation import EvaluationBudget


class ConditioningBudget(EvaluationBudget):
    CANDIDATE_ID = 'gpc-conditioning-contract-diagnostic-02'
    AUTHORIZATION_ID = 'auth20261026-user-autonomous-gpc-conditioning-contract-diagnostic-02'
    ACCOUNT_KIND = 'gpc_conditioning_diagnostic_account.v1'
    CASE_COUNT = 8
    LOSS_PER_ROLE = 0
    FORECASTS_PER_ROLE = 16
    FORECAST_COUNT = 32
    EULER_COUNT = 96
    TOTAL_SECONDS = 300
    FORECAST_SEEDS = (5478,)
    FORECAST_MODES = (('A', (1,)), ('B', (1,)))


def _digest(value):
    return hashlib.sha256(value).hexdigest()


class NativePair:
    """Two lightweight native sampler delegates, one shared frozen model."""
    def __init__(self, model, check):
        import torch
        from .gpc_conditioning_contract import make_conditioning_sampler
        self.torch, self.check = torch, check
        self.device = next(model.parameters()).device
        self.samplers = {arm: make_conditioning_sampler(model, model.supervised_adaptation_source,
            arm, execute=True) for arm in ('A', 'B')}

    def _rng(self):
        cpu = self.torch.get_rng_state().cpu().numpy().tobytes()
        cuda = self.torch.cuda.get_rng_state(self.device).cpu().numpy().tobytes()
        return struct.pack('>II', len(cpu), len(cuda))+cpu+cuda

    def sample(self, images, actions, arm):
        torch, sampler = self.torch, self.samplers[arm]
        start = len(sampler.denoiser.records)
        with torch.random.fork_rng(devices=[self.device]), torch.inference_mode():
            torch.manual_seed(5478)
            entry = self._rng()
            output, _ = sampler.sample(torch.tensor(images, device=self.device)[None],
                                       torch.tensor(actions, device=self.device)[None])
            torch.cuda.synchronize(self.device)
            exit_state = self._rng()
            self.check()
            raw = output[0].cpu().numpy()
        conditioners = deepcopy(sampler.denoiser.records[start:])
        if len(conditioners) != 3:
            raise ValueError('Exactly three native Euler/denoiser records required')
        return dict(raw=raw, rng_entry=entry, rng_exit=exit_state, conditioners=conditioners)


def run_diagnostic(prepared, binding, *, factory, close_model, writer, make_pair=NativePair,
                   progress=lambda *args: None, memory_check=lambda: None,
                   loader=None, state_loader=None, measure=None, panel=None,
                   clock=time.time, monotonic=time.monotonic):
    """The only32-call order; ordinary fixtures can exercise the same account."""
    import numpy as np
    from .gpc_conditioning_data import load_training_window, measure_h1, training_panel_png
    from .gpc_evaluation_assets import load_model_only_state
    if prepared.get('ready') is not True:
        raise PermissionError('Fixed prepared diagnosis required')
    specs = prepared['inputs']['windows']
    keys = [(spec['window_id'], spec['shape']) for spec in specs]
    loader, state_loader = loader or load_training_window, state_loader or load_model_only_state
    measure, panel = measure or measure_h1, panel or training_panel_png
    seconds = min(300, binding['deadline_unix']-clock(), binding['deadline_monotonic']-monotonic())
    budget = ConditioningBudget.create(writer.directory/'ledger', binding, case_keys=keys,
        deadline_seconds=seconds, clock=clock, monotonic=monotonic)
    model = pair = state_dict = None
    close_attempted = False
    rows, matched, samples, predictions = [], {}, {}, {}

    def check(phase):
        budget.check(); memory_check(); progress(phase, budget.snapshot())

    try:
        for spec in specs:
            samples[spec['window_id']] = loader(spec)
            check('evaluation')
        writer.json('input-read-receipts.json', [samples[key[0]]['read_receipt'] for key in keys])
        for role in ('frozen', 'adapted'):
            close_attempted = False
            operation = budget.reserve_model_load(role)
            check('model_loading')
            state_dict = state_loader(role, prepared['assets'])
            model = factory(role, state_dict=state_dict)
            state_dict = None
            model.requires_grad_(False).eval()
            pair = make_pair(model, memory_check)
            budget.complete_model_load(operation)
            check('evaluation')
            for spec in specs:
                window_id = spec['window_id']
                loaded = samples[window_id]
                images, actions = loaded['images'][:4], loaded['actions']
                image_sha, action_sha = _digest(images.tobytes()), _digest(actions.tobytes())
                predictions.setdefault(window_id, {}).setdefault(role, {})
                for arm in ('A', 'B'):
                    operation = budget.reserve_forecast(metadata=dict(role=role,
                        case=[window_id, spec['shape'], 5478], mode=arm, horizon=1))
                    check('evaluation')
                    result = pair.sample(images, actions, arm)
                    raw = result['raw']
                    if raw.shape != (3,96,96) or raw.dtype != np.float32 or not np.isfinite(raw).all():
                        raise ValueError('Malformed/nonfinite native output')
                    if image_sha != _digest(images.tobytes()) or action_sha != _digest(actions.tobytes()):
                        raise ValueError('Native sampling modified clean history/actions')
                    identity = dict(input_sha256=image_sha, action_sha256=action_sha,
                        rng_entry_sha256=_digest(result['rng_entry']), rng_exit_sha256=_digest(result['rng_exit']))
                    if window_id in matched and identity != matched[window_id]:
                        raise ValueError('A/B or role input/action/entry/exit RNG pairing differs')
                    matched.setdefault(window_id, identity)
                    prefix = f'predictions/{role}-{window_id}-{arm}'
                    raw_path = writer.raw(prefix+'.npy', raw)
                    writer.write(prefix+'-rng-entry.bin', result['rng_entry'], metadata=False)
                    writer.write(prefix+'-rng-exit.bin', result['rng_exit'], metadata=False)
                    budget.complete_forecast(operation)
                    row = dict(role=role, window_id=window_id, shape=spec['shape'], arm=arm,
                        sigma_cond=None if arm=='A' else .002, seed=5478, horizon=1,
                        raw_path=raw_path, raw_sha256=_digest(raw.tobytes()), **identity,
                        conditioners=result['conditioners'], status='completed')
                    writer.json(prefix+'-sampling.json',row)
                    row['diagnostics'] = measure(raw, spec, loaded['images'][4])
                    writer.json(prefix+'-measurement.json',row['diagnostics'])
                    rows.append(row)
                    predictions[window_id][role][arm] = raw.copy()
                    check('evaluation')
            pair = model = None
            close_attempted = True
            close_model()
            budget.release_model(role)
            check('evaluation')
        check('export')
        for spec in specs:
            key = spec['window_id']
            writer.write('panels/'+key+'.png', panel(samples[key], predictions[key]), metadata=False)
            check('export')
        if len(rows) != 32:
            raise ValueError('Full32 forecasts, including invalid diagnostics, required')
        state = budget.finish()
        report = dict(kind='gpc_conditioning_diagnostic_result.v1',status='complete',forecast_records=len(rows),
            source_logical_steps=500,new_optimizer_updates=0,new_training_examples=0,new_loss_examples=0,
            physical_operations=0,policy_operations=0,new_collection=0,model_only_loads=2,
            optimizer_state_restored=False,training_rng_restored=False,denominators=prepared['inputs']['denominators'],
            budget=state,rows=rows,interpretation='Fixed training-window mechanism screen, not dynamics/policy acceptance')
        writer.json('completion.json', report)
        check('complete')
        return report
    except BaseException as error:
        budget.fail('conditioning_diagnostic_failed: '+type(error).__name__+': '+str(error))
        raise
    finally:
        primary = sys.exc_info()[1]
        state_dict = pair = model = None
        active = budget.snapshot()['active_model_role']
        if active is not None and not close_attempted:
            try:
                close_attempted = True
                close_model()
                budget.release_model(active, cleanup=True)
            except BaseException as error:
                if primary is None: raise
                primary.add_note('Conditioning model cleanup failed: '+str(error))


def run_worker(path):
    """Owned Job only; existing SDK trace, heartbeat and append status are reused."""
    from pathlib import Path
    from . import gpc_adaptation_guard as guard
    from .gpc_conditioning_entry import verify_worker
    from .gpc_evaluation_status import Writer
    from .gpc_adaptation import OutputWriter
    handoff,prepared = verify_worker(path)
    cfg = prepared['config']
    runtime = dict(torch=None,model_instances=0)
    trace = guard.StartupTrace(path)
    def refresh():
        values = dict(model_instances=runtime['model_instances'])
        if runtime['torch'] is not None: values.update(guard.cuda_allocator_sample(runtime['torch']))
        return values
    with trace, Writer(Path(handoff['status_path']),pid=handoff['actual_worker_pid'],nonce=handoff['nonce']) as journal:
        with guard.Heartbeat(path,guard.status_template(conditioning_only=True),refresh=refresh,
                publisher=lambda _path,value:journal.publish(value)) as heartbeat:
            try:
                torch,_ = guard.initialize_sdk(heartbeat,trace,cpu_threads=2)
                runtime['torch'] = torch
                from .gpc_supervised_adaptation import make_real_denoiser
                def memory_check():
                    heartbeat.check(); torch.cuda.synchronize(0)
                    if (torch.cuda.max_memory_allocated(0)>cfg['limits']['cuda_allocated_bytes']
                            or torch.cuda.max_memory_reserved(0)>cfg['limits']['cuda_reserved_bytes']):
                        raise MemoryError('Fixed conditioning CUDA ceiling exceeded')
                def progress(phase,state):
                    heartbeat.update(phase=phase,optimizer_updates=0,
                        model_loads=state['spent_reserved_model_loads'],forecasts=state['spent_reserved_forecasts'],
                        euler_steps=state['spent_reserved_euler_steps'],eval_batches=0,eval_examples=0)
                def factory(role,*,state_dict):
                    runtime['model_instances']=1;heartbeat.update(phase='model_loading')
                    return make_real_denoiser(handoff['gpc_root'],handoff['checkpoint'],'cuda:0',
                        dict(execute=True,evaluation_only=True,validated_checkpoint_identity=cfg['sources'][role]),state_dict=state_dict)
                def close_model():
                    gc.collect();torch.cuda.empty_cache();runtime['model_instances']=0;heartbeat.update()
                result=run_diagnostic(prepared,handoff['binding'],factory=factory,close_model=close_model,
                    writer=OutputWriter(Path(handoff['output_dir']),cfg['limits']),progress=progress,memory_check=memory_check)
            except BaseException as error:
                heartbeat.record_error(error,step='conditioning.worker')
                raise
    print(json.dumps(dict(status=result['status'],forecast_records=result['forecast_records'])))
    return 0
