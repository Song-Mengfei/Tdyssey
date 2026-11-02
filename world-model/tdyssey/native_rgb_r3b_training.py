"""One frozen R3A epoch of native GPC training; Torch is loaded only by worker.

No collection, image repair, model head, new sampling order or epoch2 exists here.
The caller owns the background process, evaluation and global accounting lifetime.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import threading
import time
from types import SimpleNamespace
import uuid

import numpy as np
from .native_rgb_training import atomic_json
from .native_rgb_r3a_conditioning import POLICY_SHA

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/native-rgb-supervised-20261029'
MANIFEST_SHA = '0e91e1412f937c7d8f39473243a27a3720317a07474e730a510e628d040c0a19'
PARENT_SHA = 'c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148'
ORDER_SHA = '800655f0dc54234f92c9784dc4b0ff94e44910d488d39f32439ff625a2ee2335'
CATALOGUE_SHA = 'c80581a0d7c4aee7a377a28f9833a4355307a08ef53d2a8759ca1a745d2bae6a'
INDEX_PLAN_SHA = '98b067eeef489af494814cd91ce24a3324c03eed38da55ccd0bbd6a28537683b'
CHECKPOINT_KIND = 'native_gpc_rgb_R3B_training.v1'
SEEDS = dict(branch_seed=1409496638345316890, target_seed=5508129404114374297,
             weak_seed=7192873019884664397)


def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def identity(value): return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize_config(config):
    value = dict(config)
    defaults = dict(stage_root=str(BASE/'r3b'), source_root=str(ROOT.parent/'Tdyssey/third_party/gpc_code'),
        physics_source_root=str(ROOT.parent/'Tdyssey/third_party/diffusion_policy'),
        data_manifest=str(BASE/'r3a/manifest.json'), window_catalogue=str(BASE/'r3a/data-interface/window-catalogue.npz'),
        index_plan=str(BASE/'r3a/r3b-proposal/eight-epoch-index-plan.npz'),
        resume_checkpoint=str(BASE/'r2c3/run/last.pt'), ledger_path=str(BASE/'native-rgb-cost-ledger.json'),
        resume_authorized=False, device='cuda', frame_cache_bytes=16*1024**2, state_cache_bytes=8*1024**2,
        stage_cap_bytes=320*1024**2, atomic_reserve_bytes=256*1024**2, max_cuda_bytes=6*1024**3,
        engineering_max_attempts=2, max_consecutive_AMP_overflows=16)
    for key, val in defaults.items(): value.setdefault(key, val)
    for alias, target in (('output', 'stage_root'), ('physics_source', 'physics_source_root'),
                          ('manifest', 'data_manifest'), ('starting_checkpoint', 'resume_checkpoint')):
        if alias in config and target not in config: value[target] = config[alias]
    if value['engineering_max_attempts'] != 2 or value['max_consecutive_AMP_overflows'] != 16:
        raise ValueError('Frozen focused-probe/ordinary AMP overflow bounds changed')
    return value


class FrozenFirstEpochOrder:
    """A committed consumed cursor; CPU prefetch never advances this object."""
    def __init__(self, rows, batches, *, index_plan_sha256):
        self.rows = rows; self.batches = [tuple(map(int, batch)) for batch in batches]
        flat = [i for batch in self.batches for i in batch]
        if len(rows) != 23984 or len(self.batches) != 1499 or any(len(b) != 16 for b in self.batches):
            raise ValueError('Only23984distinct targets in1499full16batches are released')
        if sorted(flat) != list(range(23984)) or any(len({rows[i]['family_id'] for i in b}) != 16 for b in self.batches):
            raise ValueError('Frozen epoch repeats/drops windows or repeats a family within a batch')
        ids = [rows[i]['window_id'] for i in flat]
        if identity(ids) != ORDER_SHA: raise ValueError('Original first-epoch window-ID order differs')
        self.identity = dict(kind='R3B-frozen-first-epoch.v1', window_order_sha256=ORDER_SHA,
            index_plan_sha256=index_plan_sha256, selected_windows=23984, batch_size=16, epoch_batches=1499)
        self.batch_cursor = 0; self.total_exposures = 0

    def remaining_batches(self): return self.batches[self.batch_cursor:]
    def consume(self, expected):
        if self.batch_cursor >= 1499: raise StopIteration('Firstepoch ended; epoch2 is not authorized')
        indices = self.batches[self.batch_cursor]
        if tuple(expected) != indices: raise ValueError('Actual rendered batch differs from frozen consumed cursor')
        self.batch_cursor += 1; self.total_exposures += len(indices)
        return indices
    def state_dict(self):
        state = dict(kind='R3B-frozen-first-epoch-state.v1', identity=self.identity,
            batches=[list(b) for b in self.batches], batch_cursor=self.batch_cursor,
            total_exposures=self.total_exposures, epoch=1, repeated_exposures=0)
        state['state_sha256'] = identity(state); return state
    def load_state_dict(self, state):
        value = deepcopy(state); digest = value.pop('state_sha256', None)
        if digest != identity(value) or value['identity'] != self.identity:
            raise ValueError('Sampler state/data/order identity differs')
        if value['batches'] != [list(b) for b in self.batches] or value['epoch'] != 1 or value['repeated_exposures'] != 0:
            raise ValueError('Only the original unmodified firstepoch may resume')
        cursor = value['batch_cursor']
        if type(cursor) is not int or not 0 <= cursor <= 1499 or value['total_exposures'] != cursor*16:
            raise ValueError('Consumed cursor and exposure count disagree')
        self.batch_cursor = cursor; self.total_exposures = cursor*16


def training_inputs(config):
    """Rebuild exact metadata from the compact archive, without any rendering."""
    manifest = read(config['data_manifest'])
    if sha(config['data_manifest']) != MANIFEST_SHA or len(manifest['entries']) != 912:
        raise ValueError('Only the accepted R3A library is released; CP1/CP2 excluded')
    metadata = read(Path(config['window_catalogue']).with_suffix('.json'))
    if sha(config['window_catalogue']) != CATALOGUE_SHA or metadata['sha256'] != CATALOGUE_SHA:
        raise ValueError('Original compact window catalogue changed')
    if sha(config['index_plan']) != INDEX_PLAN_SHA:
        raise ValueError('Original frozen eight-epoch proposal index archive changed')
    with np.load(config['window_catalogue'], allow_pickle=False) as archive:
        a = {key: archive[key] for key in ('h1_trajectory_index', 'h1_anchor', 'h1_translation_image96_px',
            'h1_signed_rotation_deg', 'h1_category', 'selected_h1_training_indices')}
    entries = manifest['entries']; rows = []; codes = ['other','translation_only','rotation_only','both']
    for i, (ti, anchor) in enumerate(zip(a['h1_trajectory_index'], a['h1_anchor'])):
        ti, anchor = int(ti), int(anchor); entry = entries[ti]
        trans = float(a['h1_translation_image96_px'][i]); rotation = float(a['h1_signed_rotation_deg'][i])
        translation, rot = trans >= 2, abs(rotation) >= 5
        rows.append(dict(dataset_index=i, window_id=f"{entry['trajectory_id']}:{anchor}:H1",
            trajectory_index=ti, trajectory_id=entry['trajectory_id'], family_id=entry['family_id'],
            letter=entry['letter'], split=entry['split'], anchor=anchor, horizon=1,
            translation_image96_px=trans, signed_rotation_deg=rotation, abs_rotation_deg=abs(rotation),
            translation=translation, rotation=rot, moving=translation or rot,
            category=codes[int(a['h1_category'][i])], valid_window=True, control_dt=.1))
    if identity(rows) != metadata['catalogue_identity']['h1_sha256']:
        raise ValueError('Compact metadata does not reconstruct the frozen H1 catalogue')
    selected = a['selected_h1_training_indices'].astype(np.int32)
    pool = dict(metadata['selected_pool'], selected_h1_training_indices=selected.tolist())
    selected_rows = [rows[int(i)] for i in selected]
    if any(r['split'] != 'train' for r in selected_rows): raise ValueError('Validation leaked into selected TRAIN')
    with np.load(config['index_plan'], allow_pickle=False) as archive:
        if not np.array_equal(archive['selected_h1_indices'], selected): raise ValueError('Frozen plan selected-local mapping differs')
        flat, offsets, epochs = archive['flat_local_indices'], archive['batch_offsets'], archive['batch_epochs']
        batches = [flat[offsets[i]:offsets[i+1]].tolist() for i in np.flatnonzero(epochs == 1)]
    order = FrozenFirstEpochOrder(selected_rows, batches, index_plan_sha256=sha(config['index_plan']))
    catalogue = dict(h1=rows, manifest_sha256=MANIFEST_SHA, identity=metadata['catalogue_identity'])
    kwargs = dict(manifest_path=config['data_manifest'], source_root=config['physics_source_root'], split='train',
        catalogue=catalogue, selected_pool=pool, frame_cache_bytes=config['frame_cache_bytes'],
        state_cache_bytes=config['state_cache_bytes'], require_complete_library=True)
    return kwargs, order


class R3BModelLedger:
    """Append to the existing record-only ledger; never resets historical cost."""
    def __init__(self, config):
        self.config = normalize_config(config); self.path = Path(self.config['ledger_path']).resolve()
        self.lock = self.path.with_name(self.path.name+'.lock'); self.guard = threading.RLock()
        self.token = None; self.started = None; self.closed = False
        value = read(self.path)
        if value['kind'] != 'native_gpc_rgb_cost.v2-record-only-time' or value.get('max_seconds') is not None:
            raise ValueError('Existing user record_only_v2 ledger required; no model-time cap')
        prior = self.config.get('prior_ledger_copy')
        if prior:
            if sha(prior) != self.config['prior_ledger_sha256']: raise ValueError('Frozen prior model ledger copy differs')
            old = read(prior)
            if value['stages'][:len(old['stages'])] != old['stages']:
                raise ValueError('Historical cost stages changed')
            if not self.config['resume_authorized'] and sha(self.path) != self.config['prior_ledger_sha256']:
                raise ValueError('First actor ledger changed after preflight')

    @contextmanager
    def _locked(self):
        with self.guard:
            fd = os.open(self.lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            try: os.write(fd, str(os.getpid()).encode()); yield
            finally: os.close(fd); self.lock.unlink(missing_ok=True)

    def begin(self):
        with self._locked():
            value = read(self.path)
            if value['active_stage'] is not None: raise RuntimeError('Unfinished global model stage must be explicitly reconciled, never stolen')
            self.token = uuid.uuid4().hex; self.started = time.monotonic()
            value['active_stage'] = dict(token=self.token, phase='R3B-background-first-epoch', pid=os.getpid(),
                started_unix=time.time(), seconds_before=value['charged_seconds'], observed_seconds=0.,
                time_policy='record_only_v2', reserved_updates=0, completed_updates=0, skipped_updates=0,
                abandoned_updates=0, pending_attempt=None, formal=dict(attempts=0,successes=0,skips=0,abandoned=0),
                engineering=dict(attempts=0,successes=0,skips=0,abandoned=0), sections=[])
            value['max_seconds'] = None; value['max_updates'] = None; atomic_json(self.path,value)
        return self

    def _change(self, operation=None, kind=None, metadata=None, finish=None):
        with self._locked():
            value = read(self.path); active = value['active_stage']
            if self.closed or active is None or active['token'] != self.token: raise RuntimeError('Global model accounting owner differs')
            active['observed_seconds'] = max(active['observed_seconds'], time.monotonic()-self.started)
            value['charged_seconds'] = active['seconds_before']+active['observed_seconds']
            if operation == 'reserve':
                if kind not in ('formal','engineering') or active['pending_attempt'] is not None: raise RuntimeError('One explicit pending model attempt required')
                if active[kind]['attempts'] >= (1499 if kind == 'formal' else 2): raise RuntimeError('This released model phase attempt bound is exhausted')
                active[kind]['attempts'] += 1; active['reserved_updates'] += 1; value['charged_updates'] += 1
                active['pending_attempt'] = dict(kind=kind, number=active[kind]['attempts'])
            elif operation in ('complete','skip','abandon'):
                if active['pending_attempt'] is None or active['pending_attempt']['kind'] != kind: raise RuntimeError('Outcome requires exactly one matching reservation')
                field = dict(complete='successes',skip='skips',abandon='abandoned')[operation]
                active[kind][field] += 1
                if operation == 'complete': active['completed_updates'] += 1; value['completed_updates'] += 1
                elif operation == 'skip': active['skipped_updates'] += 1
                else: active['abandoned_updates'] += 1
                active['last_attempt_outcome'] = dict(operation=operation,kind=kind,metadata=metadata)
                active['pending_attempt'] = None
            elif operation == 'section': active['sections'].append(metadata)
            if finish is not None:
                active.update(status=str(finish), finished_unix=time.time(), pending_at_exit=active['pending_attempt'])
                value['stages'].append(active); value['active_stage'] = None; self.closed = True
            atomic_json(self.path,value); return value

    def heartbeat(self): return self._change()
    def reserve_attempt(self,kind): return self._change('reserve',kind)
    def complete_attempt(self,kind): return self._change('complete',kind)
    def skip_attempt(self,kind,metadata=None): return self._change('skip',kind,metadata)
    def abandon_attempt(self,kind,reason): return self._change('abandon',kind,dict(reason=reason))
    def snapshot(self):
        with self._locked(): return read(self.path)
    def finish(self,status='complete'):
        if not self.closed: return self._change(finish=status)
        return read(self.path)
    @contextmanager
    def model_section(self,label):
        started=time.monotonic(); self.heartbeat()
        try: yield
        finally: self._change('section',metadata=dict(label=str(label),seconds=time.monotonic()-started))


def fresh_progress():
    return dict(stage_attempts=0,stage_successes=0,AMP_skips=0,abandoned_attempts=0,formal_draws=0,
        lineage_updates=6000,complete_optimizer_boundary=True,pending_attempt=None,epoch_complete=False,
        consecutive_AMP_overflows=0,maximum_consecutive_AMP_overflows=0)


def validate_formal_progress(progress, order):
    cursor = order['batch_cursor']; draws = order['total_exposures']
    counters = ('stage_attempts','stage_successes','AMP_skips','abandoned_attempts','formal_draws','lineage_updates')
    if any(type(progress.get(k)) is not int or progress[k] < 0 for k in counters):
        raise ValueError('Formal counters must be nonnegative integers')
    if progress['stage_attempts'] != cursor or progress['formal_draws'] != draws or draws != 16*cursor:
        raise ValueError('Formal attempt/exposure cursor differs from checkpoint')
    if sum(progress[k] for k in ('stage_successes','AMP_skips','abandoned_attempts')) != cursor:
        raise ValueError('Incomplete/unknown checkpoint cannot silently replay or resume')
    if progress['lineage_updates'] != 6000+progress['stage_successes']:
        raise ValueError('Formal lineage includes only successful updates after6000')
    if not progress['complete_optimizer_boundary'] or progress['pending_attempt'] is not None:
        raise ValueError('Not a complete resumable optimizer boundary')
    if progress['epoch_complete'] and cursor != 1499:
        raise ValueError('Only all1499 attempts finish the first epoch')


def check_durable_cursor(durable, checkpoint_cursor):
    if durable and durable['order_batch_cursor'] > checkpoint_cursor:
        raise RuntimeError('Durable completed/consumed cursor leads last.pt; refusing to replay after an unclean hard kill')


class TrainingSession:
    def __init__(self,config,status_cb=None):
        import torch
        from .gpc_world_model import load_world_model_source,world_model_spec
        from .native_rgb_training import load_checkpoint_payload
        from .raw_world_model_training import make_generator,OPTIMIZER
        self.config=normalize_config(config);self.status_cb=status_cb;self.stage_root=Path(self.config['stage_root']).resolve()
        self.run=self.stage_root/'run';self.run.mkdir(parents=True,exist_ok=True);self.checkpoint_path=self.run/'last.pt'
        self.progress_path=self.run/'progress.json';self.loader=None;self.closed=False;self.clean=True;self.dirty=False;self.phase='formal_training'
        self.accounting=self.config.get('model_accounting');self.own_accounting=False
        if self.accounting is None:self.accounting=R3BModelLedger(self.config).begin();self.own_accounting=True
        self.dataset_kwargs,self.order=training_inputs(self.config)
        self.base_checkpoint=Path(self.config['resume_checkpoint']).resolve()
        if sha(self.base_checkpoint)!=PARENT_SHA:raise ValueError('Immutable full6000 parent changed')
        if not torch.cuda.is_available()or not str(self.config['device']).startswith('cuda'):raise RuntimeError('This released native model phase requires the local CUDA device')
        torch.set_num_threads(2)
        self.source=load_world_model_source(self.config['source_root']);inner=deepcopy(world_model_spec()['model'])
        denoiser={k:inner.pop(k)for k in('sigma_data','sigma_offset_noise','noise_previous_obs')}
        with torch.random.fork_rng(devices=[]):self.model=self.source.Denoiser(self.source.DenoiserConfig(inner_model=self.source.InnerModelConfig(**inner),**denoiser)).float()
        self.model.to(self.config['device']).requires_grad_(True)
        self.model.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.));self.model.train()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),**dict(OPTIMIZER,lr=1e-5))
        self.scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=128.)
        self.legacy_generator=make_generator(5478,next(self.model.parameters()).device)
        self._base_payload=load_checkpoint_payload(self.base_checkpoint)
        self.legacy_scheduler=deepcopy(self._base_payload['scheduler']);self.scheduler=None
        self.application_count=0;self._optimizer_hook=self.optimizer.register_step_post_hook(self._applied)
        self.counters={k:dict(forward_attempts=0,forward_completed=0,backward_microbatches=0,clean_microbatches=0,weak_microbatches=0)for k in('formal','engineering')}
        self._restore_base()
        self.identity=dict(kind='R3B-native-first-epoch-identity.v1',parent_checkpoint_sha256=PARENT_SHA,
            manifest_sha256=MANIFEST_SHA,index_plan_sha256=sha(self.config['index_plan']),window_order_sha256=ORDER_SHA,
            conditioning_policy_sha256=POLICY_SHA,conditioning_seeds=SEEDS,learning_rate=1e-5,
            selected_targets=23984,epoch_batches=1499,microbatch=4,effective_batch=16,AMP=True,
            LR='held1e-5; original scheduler retained only as provenance, never stepped')
        self.migration_provenance=dict(parent_identity=deepcopy(self._base_payload['identity']),
            parent_config=deepcopy(self._base_payload['config']),parent_progress=deepcopy(self._base_payload['progress']),
            original_prepared_order=deepcopy(self._base_payload['order']),original_global_and_noise_rng=deepcopy(self._base_payload['rng']))
        self.restore_proof=self._base_restore_proof()
        durable=read(self.progress_path)if self.progress_path.exists()else None
        if self.checkpoint_path.exists():
            if not self.config['resume_authorized']:raise RuntimeError('Existing R3B last.pt requires explicit resume authorization; never silently restart')
            payload=self._load_path(self.checkpoint_path)
            check_durable_cursor(durable,payload['order']['batch_cursor'])
            self._restore_own(payload,require_phase='formal_training')
        else:check_durable_cursor(durable,0)
        self._persist_progress();self.emit('training_session_ready',dict(progress=self.progress,restored_base=self.restore_proof))

    def _applied(self,*args):self.application_count+=1
    def emit(self,event,details):
        row=dict(at_utc=utc(),event=event,**details)
        with (self.run/'events.jsonl').open('a',encoding='utf-8')as stream:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        if self.status_cb:self.status_cb(event,details)
    def _digest(self,value):
        import torch
        from .native_rgb_r2c1r1_training import _tree_digest
        return _tree_digest(value,torch)
    def _restore_base(self):
        from .native_rgb_training import restore_rng
        from .native_rgb_r3a_conditioning import make_conditioning_rngs
        payload=self._base_payload
        if payload['progress'].get('updates')!=6000 or not payload['progress'].get('complete_optimizer_boundary'):
            raise ValueError('Only complete6000 lineage is released')
        self.model.load_state_dict(payload['model'],strict=True);self.optimizer.load_state_dict(payload['optimizer']);self.scaler.load_state_dict(payload['scaler'])
        if any(abs(group['lr']-1e-5)>1e-14 for group in self.optimizer.param_groups)or payload['scheduler']['last_epoch']!=3000:
            raise ValueError('Restore must preserve held1e-5 and legacy scheduler3000, without rebound/reset')
        restore_rng(payload['rng'],self.legacy_generator)
        self.conditioning_rngs=make_conditioning_rngs(**SEEDS,device=next(self.model.parameters()).device)
        self.order.batch_cursor=0;self.order.total_exposures=0;self.progress=fresh_progress()
        self.optimizer.zero_grad(set_to_none=True);self.model.train();self.clean=True;self.dirty=False
    def _base_restore_proof(self):
        from .native_rgb_training import capture_rng
        p=self._base_payload
        components={key:self._digest(value)==self._digest(p[key])for key,value in(
            ('model',self.model.state_dict()),('optimizer',self.optimizer.state_dict()),('scaler',self.scaler.state_dict()),
            ('scheduler',self.legacy_scheduler),('rng',capture_rng(self.legacy_generator)))}
        if not all(components.values())or not all(p.requires_grad for p in self.model.parameters()):raise RuntimeError('Full6000 restore/full-parameter verification failed')
        return dict(passed=True,components=components,parent_checkpoint_sha256=PARENT_SHA,new_cursor=0,
            old_order_and_rng_provenance_retained=True,new_three_conditioning_streams_seeded=True,
            training_all_parameters=all(p.requires_grad for p in self.model.parameters()))
    def _persist_progress(self):
        atomic_json(self.progress_path,dict(kind='R3B-durable-progress.v1',at_utc=utc(),**self.progress,
            order_batch_cursor=self.order.batch_cursor,total_exposures=self.order.total_exposures,counters=self.counters,
            checkpoint_last_saved_cursor=getattr(self,'last_saved_cursor',None)))
    def _storage(self,reserve=0):
        if hasattr(self,'model')and self.model is not None:
            import torch
            if torch.cuda.memory_reserved()>self.config['max_cuda_bytes']:raise RuntimeError('CUDA reserved memory exceeded the approved6GiB model ceiling')
        checker=self.config.get('storage_check')
        if checker:return checker(write_reserve_bytes=reserve)
        size=sum(p.stat().st_size for p in self.stage_root.rglob('*')if p.is_file()and not p.name.endswith('.tmp'))
        if size>self.config['stage_cap_bytes']:raise RuntimeError('R3B persistent stage exceeded320MiB')
        if shutil.disk_usage(self.stage_root).free<self.config['atomic_reserve_bytes']:raise RuntimeError('Atomic disk reserve unavailable')
    def _load_path(self,path):
        import torch
        value=torch.load(path,map_location='cpu',weights_only=False)
        if value.get('kind')!=CHECKPOINT_KIND or value['identity']!=self.identity:raise ValueError('R3B full checkpoint identity differs')
        return value
    def _payload(self,phase):
        from .native_rgb_training import _cpu_tree,capture_rng
        if not self.clean or not self.progress['complete_optimizer_boundary']:raise RuntimeError('Refusing resumable payload from uncertain optimizer state')
        return dict(kind=CHECKPOINT_KIND,phase=phase,identity=self.identity,model=_cpu_tree(self.model.state_dict()),
            optimizer=_cpu_tree(self.optimizer.state_dict()),scaler=_cpu_tree(self.scaler.state_dict()),
            legacy_scheduler_provenance=deepcopy(self.legacy_scheduler),global_rng=capture_rng(self.legacy_generator),
            migration_provenance=deepcopy(self.migration_provenance),
            conditioning_rngs=self.conditioning_rngs.state_dict(),order=self.order.state_dict(),progress=deepcopy(self.progress),
            counters=deepcopy(self.counters),global_ledger_snapshot=self.accounting.snapshot(),parent_lineage_updates=6000)
    def _save(self,path,phase):
        import torch
        self._storage(self.config['atomic_reserve_bytes']);payload=self._payload(phase)
        temp=Path(path).with_name(Path(path).name+'.'+uuid.uuid4().hex+'.tmp')
        torch.save(payload,temp)
        with temp.open('rb+')as stream:os.fsync(stream.fileno())
        deadline=time.monotonic()+2
        while True:
            try:os.replace(temp,path);break
            except PermissionError:
                if time.monotonic()>=deadline:raise
                time.sleep(.025)
        del payload;self.last_saved_cursor=self.order.batch_cursor;self.dirty=False
        result=dict(path=str(path),bytes=Path(path).stat().st_size,sha256=sha(path),cursor=self.order.batch_cursor,
            successes=self.progress['stage_successes'],phase=phase)
        if phase=='formal_training':self._persist_progress()
        self.emit('checkpoint_saved',result);self._storage();return result
    def _restore_own(self,payload,*,require_phase):
        from .native_rgb_training import restore_rng
        if payload['phase']!=require_phase:raise ValueError('Engineering checkpoint is never a formal resume source')
        if payload['legacy_scheduler_provenance']!=self.legacy_scheduler:raise ValueError('Legacy scheduler provenance changed')
        if self._digest(payload['migration_provenance'])!=self._digest(self.migration_provenance):raise ValueError('Original6000 migration provenance changed')
        validate_formal_progress(payload['progress'],payload['order'])
        if any(abs(group['lr']-1e-5)>1e-14 for group in payload['optimizer']['param_groups']):raise ValueError('R3B held learning rate differs')
        self.model.load_state_dict(payload['model'],strict=True);self.optimizer.load_state_dict(payload['optimizer']);self.scaler.load_state_dict(payload['scaler'])
        self.order.load_state_dict(payload['order']);self.progress=deepcopy(payload['progress']);self.counters=deepcopy(payload['counters'])
        restore_rng(payload['global_rng'],self.legacy_generator);self.conditioning_rngs.load_state_dict(payload['conditioning_rngs'])
        self.optimizer.zero_grad(set_to_none=True);self.clean=True;self.dirty=False;self.last_saved_cursor=self.order.batch_cursor
    def checkpoint_if_possible(self,reason='requested'):
        if self.phase!='formal_training':return dict(saved=False,reason='Engineering branch is not a formal checkpoint source')
        if not self.clean:return dict(saved=False,reason='uncertain optimizer boundary; previous checkpoint retained')
        if not self.dirty and self.checkpoint_path.exists():return dict(saved=False,reason='already saved complete boundary',path=str(self.checkpoint_path))
        return self._save(self.checkpoint_path,'formal_training')
    @contextmanager
    def preserve_training_state(self):
        from .native_rgb_training import capture_rng,restore_rng
        rng=capture_rng(self.legacy_generator);conditioning=self.conditioning_rngs.state_dict();order=self.order.state_dict();progress=deepcopy(self.progress);mode=self.model.training
        try:yield
        finally:
            changed=self.order.state_dict()!=order or self.progress!=progress
            restore_rng(rng,self.legacy_generator);self.conditioning_rngs.load_state_dict(conditioning);self.model.train(mode)
            if changed:raise RuntimeError('Evaluation changed formal order/progress; not hidden by restoring it')
    def _ensure_loader(self):
        if self.loader is None:
            from .native_rgb_r3a_data import PrefetchBatchLoader
            self.loader=PrefetchBatchLoader(self.dataset_kwargs,workers=2,prefetch_batches=2)
        return self.loader
    def close(self,status='complete'):
        if self.closed:return
        if self.loader is not None:self.loader.close();self.loader=None
        self._optimizer_hook.remove();self.closed=True
        self._base_payload=None;self.model=None;self.optimizer=None;self.scaler=None
        self.conditioning_rngs=None;self.legacy_generator=None;gc.collect()
        import torch
        torch.cuda.empty_cache()
        if self.own_accounting:self.accounting.finish(status)


def create_session(config,status_cb=None):return TrainingSession(config,status_cb)


def _optimizer_attempt(session,arrays,*,kind,indices):
    import torch
    from .native_rgb_r3a_conditioning import conditioning_residual_loss
    from .native_rgb_r2c3_training import scaled_optimizer_step
    from .native_rgb_r3a_data import assert_model_inputs
    assert_model_inputs(arrays,batch=True)
    if len(arrays['image'])!=16:raise ValueError('The actual optimizer draw must contain exactly16 samples')
    if kind=='formal'and(session.order.batch_cursor>=1499 or tuple(indices)!=session.order.batches[session.order.batch_cursor]):
        raise ValueError('Attempt is outside the frozen first-epoch cursor')
    session._storage();session.accounting.reserve_attempt(kind);began=time.monotonic();started_step=False
    before_counters=deepcopy(session.counters[kind]);total=0.;micros=[]
    try:
        if kind=='formal':
            session.order.consume(indices);session.progress['stage_attempts']+=1;session.progress['formal_draws']+=16
        session.clean=False;session.progress['complete_optimizer_boundary']=False
        session.progress['pending_attempt']=dict(kind=kind,batch_cursor=session.order.batch_cursor,backward_microbatches=0)
        session._persist_progress();session.optimizer.zero_grad(set_to_none=True)
        for micro in range(4):
            batch={key:torch.as_tensor(value[micro*4:(micro+1)*4],device=next(session.model.parameters()).device)for key,value in arrays.items()}
            session.counters[kind]['forward_attempts']+=1
            with torch.autocast('cuda',dtype=torch.float16,enabled=True):loss,details=conditioning_residual_loss(session.model,batch,rngs=session.conditioning_rngs)
            if loss.dtype!=torch.float32:raise TypeError('Residual MSE must stayFP32 under AMP')
            session.counters[kind]['forward_completed']+=1
            session.scaler.scale(loss*.25).backward();session.counters[kind]['backward_microbatches']+=1
            session.counters[kind][details['history_branch']+'_microbatches']+=1
            session.progress['pending_attempt']['backward_microbatches']+=1
            total+=float(loss.detach())*.25;micros.append(details)
        session.scaler.unscale_(session.optimizer)
        params=[p for p in session.model.parameters()if p.requires_grad]
        if any(p.grad is None for p in params):raise FloatingPointError('Missing native full-network gradients')
        if any(not bool(torch.isfinite(p).all())for p in params):raise FloatingPointError('Nonfinite model parameter')
        finite=all(bool(torch.isfinite(p.grad).all())for p in params)
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))if finite else None
        before_skip=session._digest(session.model.state_dict())if not finite else None
        started_step=True;outcome=scaled_optimizer_step(session.scaler,session.optimizer,application_count=lambda:session.application_count)
        if outcome['applied']:
            if any(not bool(torch.isfinite(p).all())for p in params):raise FloatingPointError('Nonfinite parameter after optimizer step')
            session.accounting.complete_attempt(kind)
            if kind=='formal':session.progress['stage_successes']+=1;session.progress['lineage_updates']+=1;session.progress['consecutive_AMP_overflows']=0
        else:
            if before_skip is not None and before_skip!=session._digest(session.model.state_dict()):raise FloatingPointError('AMP skip mutated model')
            session.accounting.skip_attempt(kind,outcome)
            if kind=='formal':
                session.progress['AMP_skips']+=1;session.progress['consecutive_AMP_overflows']+=1
                session.progress['maximum_consecutive_AMP_overflows']=max(session.progress['maximum_consecutive_AMP_overflows'],session.progress['consecutive_AMP_overflows'])
        session.optimizer.zero_grad(set_to_none=True);session.clean=True;session.dirty=True
        session.progress['pending_attempt']=None;session.progress['complete_optimizer_boundary']=True
        result=dict(kind=kind,loss=total,gradient_norm=norm,**outcome,microbatches=micros,
            actual_optimizer_applications=outcome['applied'],draws=16,indices=list(indices),seconds=time.monotonic()-began,
            learning_rate=[float(group['lr'])for group in session.optimizer.param_groups],
            microbatch_counter_delta={k:session.counters[kind][k]-before_counters[k]for k in before_counters},
            residual_MSE_dtype='float32',accumulation_weights=[.25]*4)
        if any(abs(lr-1e-5)>1e-14 for lr in result['learning_rate']):raise RuntimeError('Held LR changed')
        session._persist_progress();session.emit('optimizer_attempt_complete',dict(attempt=session.progress['stage_attempts'],**result))
        if kind=='formal'and session.progress['consecutive_AMP_overflows']>=16:raise FloatingPointError('16consecutive real AMP overflows; stop without new draw/order policy')
        return result
    except BaseException as exc:
        if not started_step:
            session.optimizer.zero_grad(set_to_none=True);session.accounting.abandon_attempt(kind,str(exc))
            if kind=='formal':session.progress['abandoned_attempts']+=1
            session.clean=True;session.dirty=True;session.progress['complete_optimizer_boundary']=True;session.progress['pending_attempt']=None
        elif not session.clean:
            atomic_json(session.run/'unsafe_checkpoint.json',dict(reason=str(exc),progress=session.progress,
                note='Optimizer application may be partial/unknown; do not replay a stale checkpoint',at_utc=utc()))
        session._persist_progress();raise


def engineering_probe(session,status_cb=None):
    """Actual one-step full-state save/reload check, then discard the branch."""
    if session.order.batch_cursor or session.progress['stage_attempts']:
        return dict(skipped=True,reason='Formal consumed cursor already exists; engineering never replays formal data')
    if (session.run/'engineering-probe.json').exists():
        old=read(session.run/'engineering-probe.json')
        if not old.get('passed'):raise RuntimeError('Previous engineering probe failed; no automatic branch replay')
        return old
    session.phase='engineering_probe'
    proof=dict(kind='R3B-actual-GPU-focused-integration.v1',passed=False,formal_draws=0,engineering_draws=0,engineering_attempts=0,
        engineering_successes=0,engineering_AMP_skips=0,microbatch4_accumulated4=True,parent_restore=session.restore_proof,results=[])
    base_digest=session._digest(session.model.state_dict());indices=session.order.batches[0]
    path=session.run/'engineering-probe.tmp.pt'
    try:
        batch=next(session._ensure_loader().iterate([indices]))
        for _ in range(2):
            result=_optimizer_attempt(session,batch['model_inputs'],kind='engineering',indices=indices)
            proof['engineering_attempts']+=1;proof['engineering_draws']+=16;proof['engineering_successes']+=result['applied'];proof['engineering_AMP_skips']+=int(not result['applied']);proof['results'].append(result)
            if result['applied']:break
        if not proof['engineering_successes']:raise RuntimeError('Focused engineering probe produced no real optimizer update; do not train')
        if any(r['microbatch_counter_delta']['backward_microbatches']!=4 or r['accumulation_weights']!=[.25]*4 for r in proof['results']):
            raise RuntimeError('Actual four-microbatch accumulation differs')
        changed=session._digest(session.model.state_dict())!=base_digest
        if not changed:raise RuntimeError('Actual engineering optimizer application did not change parameters')
        snapshot=session._payload('engineering_probe');expected={key:session._digest(snapshot[key])for key in('model','optimizer','scaler','global_rng','conditioning_rngs','order','progress')}
        del snapshot;session._save(path,'engineering_probe');session._restore_base()
        loaded=session._load_path(path);session._restore_own(loaded,require_phase='engineering_probe');del loaded
        restored=session._payload('engineering_probe');matches={key:session._digest(restored[key])==value for key,value in expected.items()};del restored
        if not all(matches.values()):raise RuntimeError('Actual engineering checkpoint reload differs')
        session._restore_base();base_restored=session._base_restore_proof()
        if session._digest(session.model.state_dict())!=base_digest or session.order.batch_cursor:raise RuntimeError('Engineering branch contaminated formal6000 initialization')
        proof.update(passed=True,actual_parameter_update_verified=changed,checkpoint_components_equal=matches,
            original6000_restored_after_probe=base_restored,formal_order_cursor=0,
            engineering_checkpoint_deleted_after_verification=True,peak_cuda_reserved_bytes=__import__('torch').cuda.max_memory_reserved())
        path.unlink();session.phase='formal_training';session.dirty=False;session._persist_progress();atomic_json(session.run/'engineering-probe.json',proof)
        session.emit('engineering_probe_complete',dict(passed=True,engineering_attempts=proof['engineering_attempts'],engineering_draws=proof['engineering_draws'],formal_draws=0))
        return proof
    except BaseException as exc:
        proof.update(failure=dict(type=type(exc).__name__,reason=str(exc)),temporary_checkpoint_retained=path.exists())
        atomic_json(session.run/'engineering-probe.json',proof);raise


def run_first_epoch(session,status_cb=None):
    """Consume the frozen1499batches once, including ordinary AMP skips."""
    if not read(session.run/'engineering-probe.json').get('passed'):raise RuntimeError('Actual focused GPU integration must pass before formal work')
    if session.progress['abandoned_attempts']:raise RuntimeError('A fatal abandoned formal attempt requires review, not silent continuation')
    session.model.train();session.checkpoint_if_possible('initial_complete6000_boundary')
    start=session.order.batch_cursor;began=time.monotonic()
    try:
        batches=session.order.remaining_batches()
        for rendered in session._ensure_loader().iterate(batches):
            result=_optimizer_attempt(session,rendered['model_inputs'],kind='formal',indices=rendered['indices'])
            if result['applied']and session.progress['stage_successes']%100==0:session.checkpoint_if_possible('100success_boundary')
            if session.order.batch_cursor%25==0:
                session.emit('formal_progress',dict(progress=deepcopy(session.progress),cursor=session.order.batch_cursor,
                    targets_consumed=session.order.total_exposures,latest_loss=result['loss'],loader_workers=len(session.loader.worker_statistics)))
            session.accounting.heartbeat()
        if session.order.batch_cursor!=1499 or session.order.total_exposures!=23984:raise RuntimeError('First epoch did not consume every frozen target once')
        session.progress['epoch_complete']=True;session.dirty=True;session._persist_progress();checkpoint=session.checkpoint_if_possible('epoch1_complete')
        result=dict(kind='R3B-complete-first-epoch.v1',complete=True,progress=deepcopy(session.progress),checkpoint=checkpoint,
            consumed_first_epoch_order_sha256=ORDER_SHA,selected_distinct_targets=23984,actual_training_draws=23984,
            repeated_formal_targets=0,unexposed_formal_targets=0,physics_steps=0,
            loader=dict(workers=2,prefetch_batches=2,maximum_in_flight=session.loader.maximum_in_flight,
                worker_statistics=session.loader.worker_statistics),new_attempts_in_this_process=session.order.batch_cursor-start,
            seconds_this_process=time.monotonic()-began,counters=deepcopy(session.counters),epoch2_authorized=False)
        atomic_json(session.run/'first-epoch-result.json',result);return result
    except BaseException as exc:
        try:saved=session.checkpoint_if_possible('exception_complete_boundary')
        except BaseException as saving:saved=dict(saved=False,save_error=repr(saving),previous_checkpoint_retained=session.checkpoint_path.exists())
        atomic_json(session.run/'training-failure.json',dict(type=type(exc).__name__,reason=str(exc),progress=session.progress,
            checkpoint=saved,automatic_restart=False,at_utc=utc()))
        raise
