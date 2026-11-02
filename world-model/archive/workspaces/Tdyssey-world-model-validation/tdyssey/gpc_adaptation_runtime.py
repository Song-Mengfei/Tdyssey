"""Small single-owner adaptation accounting and coherent boundary checkpoints.

Importing this module reads no resources and imports no tensor runtime. Durable
reservations are spent before computation. Recovery never rewinds charges or
the original wall/monotonic deadlines; inconsistent boundaries are rejected.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time
import uuid


class AdaptationRuntimeError(RuntimeError):
    pass


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _integer(value, minimum=1):
    return type(value) is int and value >= minimum


def _hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _training_schedule_identity(binding):
    keys=('training_schedule_sha256','training_schedule_order_sha256','training_schedule_rule_sha256')
    if not any(key in binding for key in keys):return None
    if not all(_hash(binding.get(key)) for key in keys):raise ValueError('Complete frozen training schedule identity required')
    from .gpc_training_schedule import RULE_SHA256
    if binding[keys[2]]!=RULE_SHA256:raise ValueError('Frozen training schedule rule differs')
    return dict(sha256=binding[keys[0]],order_sha256=binding[keys[1]],rule_sha256=binding[keys[2]])


def _checkpoint_schedule_extra(budget,extra):
    expected=budget.training_schedule_state()
    if expected is not None and (not isinstance(extra,dict) or extra.get('training_schedule')!=expected
            or extra.get('next_example')!=expected['cursor']):
        raise AdaptationRuntimeError('Checkpoint schedule identity/ordered cursor differs from spent highwater')


def _write_all(stream, value):
    view=memoryview(value)
    while view:
        count=stream.write(view)
        if type(count) is not int or not 0<count<=len(view):
            raise OSError('No progress while writing admitted runtime bytes')
        view=view[count:]


def _exclusive(path, value):
    with path.open('xb',buffering=0) as stream:
        _write_all(stream,_json(value));os.fsync(stream.fileno())


class AdaptationBudget:
    """One candidate, one live operation, append-only highwater accounting."""
    @classmethod
    def create(cls, ledger_dir, binding, *, max_updates, max_eval_batches, max_batch_size,
               deadline_seconds, max_train_examples=None, max_eval_examples=None,
               max_forecasts=3996, max_euler_steps=11988, max_model_loads=3,
               clock=time.time, monotonic=time.monotonic):
        if not isinstance(binding,dict) or not binding:
            raise ValueError('A nonempty immutable source/config/candidate binding is required')
        _json(binding)
        _training_schedule_identity(binding)
        values=dict(max_updates=max_updates,max_eval_batches=max_eval_batches,max_batch_size=max_batch_size,
            max_forecasts=max_forecasts,max_euler_steps=max_euler_steps,max_model_loads=max_model_loads)
        if any(not _integer(v) for v in values.values()) or not _finite(deadline_seconds) or deadline_seconds<=0:
            raise ValueError('Positive finite limits are required')
        if any(v is not None and not _integer(v) for v in (max_train_examples,max_eval_examples)):
            raise ValueError('Example caps must be positive integers or None')
        wall,mono=clock(),monotonic()
        if not _finite(wall) or not _finite(mono):
            raise ValueError('Finite clocks are required')
        path=Path(ledger_dir).absolute()
        attempt=dict(kind='gpc_adaptation_account.v1',budget_id=uuid.uuid4().hex,binding=deepcopy(binding),
            binding_sha256=_sha(_json(binding)),ledger_dir=str(path),
            limits=dict(values,deadline_seconds=deadline_seconds,max_train_examples=max_train_examples,max_eval_examples=max_eval_examples),
            started_unix=wall,deadline_unix=wall+deadline_seconds,
            started_monotonic=mono,deadline_monotonic=mono+deadline_seconds)
        path.mkdir(parents=True,exist_ok=False)
        _exclusive(path/'attempt.json',attempt)
        self=cls._initialize(path,attempt,clock,monotonic)
        self._append('created',{})
        return self

    @classmethod
    def _initialize(cls,path,attempt,clock,monotonic):
        self=cls.__new__(cls)
        self.directory,self.attempt,self.clock,self.monotonic=path,attempt,clock,monotonic
        self._training_schedule_identity=_training_schedule_identity(attempt['binding'])
        self._lock=threading.RLock();self._sequence=0;self._previous='0'*64;self._broken=False
        self._journal_bytes=0;self._requires_restore=False
        self._last_wall,self._last_mono=attempt['started_unix'],attempt['started_monotonic']
        self._state=dict(status='active',first_reason=None,pending=None,spent_reserved_updates=0,
            spent_reserved_eval_batches=0,spent_train_examples=0,spent_eval_examples=0,
            completed_updates=0,completed_eval_batches=0,checkpoint_committed_updates=0,
            checkpoint_committed_eval_batches=0,checkpoint=None,selection_count=0,selected_checkpoint_sha256=None,
            spent_reserved_forecasts=0,spent_reserved_euler_steps=0,completed_forecasts=0,completed_euler_steps=0,
            spent_reserved_model_loads=0,completed_model_loads=0,model_load_roles=[],active_model_role=None)
        return self

    @classmethod
    def resume(cls, ledger_dir, binding, *, clock=time.time, monotonic=time.monotonic):
        path=Path(ledger_dir).absolute();attempt=json.loads((path/'attempt.json').read_bytes())
        if (attempt.get('kind')!='gpc_adaptation_account.v1' or attempt.get('ledger_dir')!=str(path)
                or attempt.get('binding_sha256')!=_sha(_json(binding)) or attempt.get('binding')!=binding):
            raise AdaptationRuntimeError('Recovery account/source/config binding differs')
        self=cls._initialize(path,attempt,clock,monotonic)
        raw=(path/'highwater.jsonl').read_bytes()
        for line in raw.splitlines(keepends=True):
            try:row=json.loads(line)
            except (ValueError,UnicodeError) as error:
                raise AdaptationRuntimeError('Incomplete or invalid durable highwater journal') from error
            if (not line.endswith(b'\n') or row['sequence']!=self._sequence+1
                    or (self._sequence==0 and row['kind']!='created')
                    or row['previous_sha256']!=self._previous or not _finite(row['at_unix'])
                    or not _finite(row['at_monotonic']) or row['at_unix']<self._last_wall
                    or row['at_monotonic']<self._last_mono):
                raise AdaptationRuntimeError('Incomplete or inconsistent durable highwater journal')
            self._transition(row['kind'],row['data'],row['sequence'])
            self._sequence=row['sequence'];self._previous=_sha(line)
            self._last_wall,self._last_mono=row['at_unix'],row['at_monotonic']
            self._journal_bytes+=len(line)
        if not self._sequence or (path/'first-failure.json').exists():
            raise AdaptationRuntimeError('Incomplete or failed account cannot resume')
        self.check()
        if self._state['pending'] is not None:
            raise AdaptationRuntimeError('Reserved operation is not complete; charges retained, no replay')
        if self._state['checkpoint'] is None or self._state['checkpoint']['boundary']!=self._boundary():
            raise AdaptationRuntimeError('No exact committed checkpoint for this spent highwater; no rollback')
        self._requires_restore=True
        return self

    def _transition(self,kind,data,sequence):
        state=self._state;limits=self.attempt['limits']
        if kind=='created':
            if sequence!=1:raise AdaptationRuntimeError('Creation record is not first')
            return
        if state['status']!='active':
            raise AdaptationRuntimeError('Failed account cannot acquire another operation')
        if kind in ('reserve_update','reserve_eval'):
            train=kind=='reserve_update';size=data.get('batch_size')
            metadata=data.get('metadata',{})
            count='spent_reserved_updates' if train else 'spent_reserved_eval_batches'
            examples='spent_train_examples' if train else 'spent_eval_examples'
            cap='max_updates' if train else 'max_eval_batches'
            example_cap=limits['max_train_examples' if train else 'max_eval_examples']
            if (state['pending'] is not None or (train and state['selection_count']) or not _integer(size)
                    or state['active_model_role'] not in ('train_init','frozen','adapted')
                    or (train and state['active_model_role']!='train_init')
                    or not isinstance(metadata,dict)
                    or ('role' in metadata and metadata['role']!=state['active_model_role'])
                    or size>limits['max_batch_size'] or state[count]>=limits[cap]
                    or (example_cap is not None and state[examples]+size>example_cap)):
                raise AdaptationRuntimeError('Batch/update/evaluation reservation exceeds the remaining frozen budget')
            if train and self._training_schedule_identity is not None:
                expected=self.training_schedule_state();indices=metadata.get('dataset_indices')
                if (metadata.get('training_schedule')!=expected or metadata.get('next_cursor')!=expected['cursor']+size
                        or not isinstance(indices,list) or len(indices)!=size or any(type(i) is not int or i<0 for i in indices)):
                    raise AdaptationRuntimeError('Training reservation differs from its frozen schedule cursor/order')
            state[count]+=1;state[examples]+=size
            state['pending']=dict(operation_id=sequence,kind='update' if train else 'eval',batch_size=size,
                metadata=deepcopy(data.get('metadata',{})),role=state['active_model_role'])
        elif kind in ('complete_update','complete_eval'):
            wanted='update' if kind=='complete_update' else 'eval';pending=state['pending']
            if (not _integer(data.get('operation_id')) or pending is None or pending['kind']!=wanted
                    or pending['operation_id']!=data['operation_id']):
                raise AdaptationRuntimeError('Completion differs from its single pending reservation')
            state['completed_updates' if wanted=='update' else 'completed_eval_batches']+=1;state['pending']=None
        elif kind=='reserve_forecast':
            count,steps=data.get('actual_count'),data.get('euler_per_image');meta=data.get('metadata')
            if (state['pending'] is not None or not _integer(count) or count>limits['max_batch_size']
                    or steps!=3 or type(steps) is not int or not isinstance(meta,dict)
                    or meta.get('role') not in ('frozen','adapted') or meta['role']!=state['active_model_role']
                    or (meta['role']=='adapted' and not state['selection_count'])
                    or state['spent_reserved_forecasts']+count>limits['max_forecasts']
                    or state['spent_reserved_euler_steps']+count*steps>limits['max_euler_steps']):
                raise AdaptationRuntimeError('Forecast role/image/Euler reservation exceeds its independent budget')
            state['spent_reserved_forecasts']+=count;state['spent_reserved_euler_steps']+=count*steps
            state['pending']=dict(operation_id=sequence,kind='forecast',actual_count=count,euler_steps=count*steps,metadata=deepcopy(meta))
        elif kind=='complete_forecast':
            pending=state['pending']
            if (not _integer(data.get('operation_id')) or pending is None or pending['kind']!='forecast'
                    or pending['operation_id']!=data['operation_id']):
                raise AdaptationRuntimeError('Forecast completion differs from its pending reservation')
            state['completed_forecasts']+=pending['actual_count'];state['completed_euler_steps']+=pending['euler_steps'];state['pending']=None
        elif kind=='reserve_model_load':
            role=data.get('role');roles=('train_init','frozen','adapted')
            if (state['pending'] is not None or state['spent_reserved_model_loads']>=min(limits['max_model_loads'],3)
                    or role!=roles[state['spent_reserved_model_loads']]
                    or (role in ('frozen','adapted') and not state['selection_count'])):
                raise AdaptationRuntimeError('Only the three declared serial model-load roles may be reserved once')
            state['spent_reserved_model_loads']+=1;state['model_load_roles'].append(role)
            state['pending']=dict(operation_id=sequence,kind='model_load',role=role)
        elif kind=='complete_model_load':
            pending=state['pending']
            if (not _integer(data.get('operation_id')) or pending is None or pending['kind']!='model_load'
                    or pending['operation_id']!=data['operation_id']):
                raise AdaptationRuntimeError('Model-load completion differs from its pending reservation')
            state['completed_model_loads']+=1;state['active_model_role']=pending['role'];state['pending']=None
        elif kind=='checkpoint_committed':
            self._boundary()
            if (not _hash(data.get('sha256')) or data.get('boundary')!=self._boundary()
                    or (state['checkpoint'] is not None and state['checkpoint']['path']!=data.get('path'))
                    or state['selection_count']):
                raise AdaptationRuntimeError('Checkpoint is not this exact candidate/account boundary')
            state['checkpoint']=deepcopy(data)
            state['checkpoint_committed_updates']=state['completed_updates']
            state['checkpoint_committed_eval_batches']=state['completed_eval_batches']
        elif kind=='checkpoint_selected':
            current=state['checkpoint']
            if state['selection_count'] or current is None or data.get('sha256')!=current['sha256'] or current['boundary']!=self._boundary():
                raise AdaptationRuntimeError('Only the one current candidate checkpoint can be selected once')
            state['selection_count']=1;state['selected_checkpoint_sha256']=data['sha256']
        elif kind=='failed':
            state['status']='stopped';state['first_reason']=str(data['reason'])
        else:
            raise AdaptationRuntimeError('Unknown adaptation highwater operation')

    def _append(self,kind,data):
        _json(data)
        wall,mono=self.clock(),self.monotonic()
        if not _finite(wall) or not _finite(mono) or wall<self._last_wall or mono<self._last_mono:
            raise AdaptationRuntimeError('Runtime clock moved backward or became invalid')
        path=self.directory/'highwater.jsonl'
        if (path.stat().st_size if path.exists() else 0)!=self._journal_bytes:
            self._broken=True
            raise AdaptationRuntimeError('Highwater changed outside its single owner; no continuation')
        sequence=self._sequence+1;self._transition(kind,data,sequence)
        row=dict(sequence=sequence,previous_sha256=self._previous,kind=kind,data=data,at_unix=wall,at_monotonic=mono)
        payload=_json(row)
        try:
            with path.open('ab',buffering=0) as stream:
                if os.fstat(stream.fileno()).st_size!=self._journal_bytes:
                    raise AdaptationRuntimeError('Highwater append boundary changed')
                _write_all(stream,payload);os.fsync(stream.fileno())
        except BaseException:
            self._broken=True;raise
        self._sequence=sequence;self._previous=_sha(payload);self._last_wall,self._last_mono=wall,mono
        self._journal_bytes+=len(payload)

    def fail(self,reason):
        with self._lock:
            if self._state['first_reason'] is None:
                self._state['first_reason']=str(reason)
                if not self._broken and self._state['status']=='active':
                    try:self._append('failed',dict(reason=str(reason)))
                    except BaseException:self._broken=True
            self._state['status']='stopped'
            path=self.directory/'first-failure.json'
            if not path.exists():
                try:_exclusive(path,dict(reason=self._state['first_reason'],state=self._state,attempt=self.attempt))
                except Exception:pass
            return self.snapshot()

    def check(self):
        with self._lock:
            if self._state['status']!='active' or self._broken:
                raise AdaptationRuntimeError(self._state['first_reason'] or 'Journal failed; no continuation')
            wall,mono=self.clock(),self.monotonic()
            reason=('Runtime clock moved backward or became invalid' if not _finite(wall) or not _finite(mono)
                    or wall<self._last_wall or mono<self._last_mono else
                    'Original adaptation deadline includes all update/evaluation/checkpoint work' if
                    wall>=self.attempt['deadline_unix'] or mono>=self.attempt['deadline_monotonic'] else None)
            if reason:
                self.fail(reason);raise AdaptationRuntimeError(reason)
            self._last_wall,self._last_mono=wall,mono
            return self.snapshot()

    def _operation(self,kind,data):
        with self._lock:
            self.check()
            try:
                if kind.startswith('reserve_') and self._requires_restore:
                    raise AdaptationRuntimeError('Recovered account requires its exact coherent checkpoint restore before computation')
                self._append(kind,data);self.check()
            except BaseException as error:
                self.fail(type(error).__name__+': '+str(error));raise
            return self._sequence

    def reserve_update(self,batch_size,metadata=None):
        return self._operation('reserve_update',dict(batch_size=batch_size,metadata={} if metadata is None else metadata))
    def reserve_eval(self,batch_size,metadata=None):return self._operation('reserve_eval',dict(batch_size=batch_size,metadata={} if metadata is None else metadata))
    def complete_update(self,operation_id):return self._operation('complete_update',dict(operation_id=operation_id))
    def complete_eval(self,operation_id):return self._operation('complete_eval',dict(operation_id=operation_id))
    def select_checkpoint(self,sha256):return self._operation('checkpoint_selected',dict(sha256=sha256))
    def reserve_forecast(self,actual_count=1,euler_per_image=3,metadata=None):
        return self._operation('reserve_forecast',dict(actual_count=actual_count,euler_per_image=euler_per_image,metadata=metadata or {}))
    def complete_forecast(self,operation_id):return self._operation('complete_forecast',dict(operation_id=operation_id))
    def reserve_model_load(self,role):return self._operation('reserve_model_load',dict(role=role))
    def complete_model_load(self,operation_id):return self._operation('complete_model_load',dict(operation_id=operation_id))

    def _boundary(self):
        state=self._state
        if (state['pending'] is not None or state['spent_reserved_updates']!=state['completed_updates']
                or state['spent_reserved_eval_batches']!=state['completed_eval_batches']
                or state['spent_reserved_forecasts']!=state['completed_forecasts']
                or state['spent_reserved_model_loads']!=state['completed_model_loads']):
            raise AdaptationRuntimeError('Incomplete reservation cannot be paired with checkpoint weights')
        result=dict(budget_id=self.attempt['budget_id'],binding_sha256=self.attempt['binding_sha256'],
            spent_reserved_updates=state['spent_reserved_updates'],spent_reserved_eval_batches=state['spent_reserved_eval_batches'],
            spent_train_examples=state['spent_train_examples'],spent_eval_examples=state['spent_eval_examples'],
            spent_reserved_forecasts=state['spent_reserved_forecasts'],spent_reserved_euler_steps=state['spent_reserved_euler_steps'],
            spent_reserved_model_loads=state['spent_reserved_model_loads'])
        if self._training_schedule_identity is not None:result['training_schedule']=self.training_schedule_state()
        return result

    def training_schedule_state(self):
        return dict(self._training_schedule_identity,cursor=self._state['spent_train_examples']) if self._training_schedule_identity is not None else None

    def snapshot(self):
        result=dict(deepcopy(self._state),attempt=deepcopy(self.attempt),journal_sequence=self._sequence,
            journal_tip_sha256=self._previous,requires_checkpoint_restore=self._requires_restore)
        if self._training_schedule_identity is not None:result['training_schedule']=self.training_schedule_state()
        return result


def _tensor_copy(value,torch):
    if isinstance(value,torch.Tensor):
        result=value.detach().cpu().clone()
        if not torch.isfinite(result).all():raise AdaptationRuntimeError('Nonfinite checkpoint tensor')
        return result
    if isinstance(value,dict):return {k:_tensor_copy(v,torch) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(_tensor_copy(v,torch) for v in value)
    if value is None or type(value) in (str,int,bool) or _finite(value):return value
    raise AdaptationRuntimeError('Unsupported checkpoint state value')


def _capture_rng(torch,generators):
    import numpy as np
    ns=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(algorithm=ns[0],keys=ns[1].tolist(),position=ns[2],has_gauss=ns[3],cached_gaussian=ns[4]),
        torch_cpu=torch.get_rng_state().clone(),torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        generators={k:dict(device=str(v.device),state=v.get_state().clone()) for k,v in generators.items()})


def _restore_rng(value,torch,generators):
    import numpy as np
    if set(value['generators'])!=set(generators):raise AdaptationRuntimeError('Explicit RNG generators differ')
    if value['torch_cuda'] and not torch.cuda.is_initialized():raise AdaptationRuntimeError('Saved CUDA RNG requires an already initialized matching device')
    for k,v in generators.items():
        if str(v.device)!=value['generators'][k]['device']:raise AdaptationRuntimeError('Generator device differs')
    n=value['numpy'];random.setstate(value['python']);np.random.set_state((n['algorithm'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(value['torch_cpu'])
    if value['torch_cuda']:torch.cuda.set_rng_state_all(value['torch_cuda'])
    for k,v in generators.items():v.set_state(value['generators'][k]['state'])


def _optimizer_signature(model,optimizer):
    names={id(p):name for name,p in model.named_parameters()}
    groups=[]
    for group in optimizer.param_groups:
        if any(id(p) not in names for p in group['params']):
            raise AdaptationRuntimeError('Optimizer owns a parameter outside this candidate model')
        groups.append([names[id(p)] for p in group['params']])
    flattened=[p for group in groups for p in group]
    if len(flattened)!=len(set(flattened)):
        raise AdaptationRuntimeError('Optimizer parameter occurs more than once')
    return dict(kind=type(optimizer).__module__+'.'+type(optimizer).__qualname__,parameter_groups=groups)


class _BoundedWriter:
    def __init__(self,stream,maximum):self.stream,self.maximum=stream,maximum
    def write(self,value):
        if self.stream.tell()+len(value)>self.maximum:raise OSError('Checkpoint exceeds its admitted byte bound')
        _write_all(self.stream,value);return len(value)
    def flush(self):return self.stream.flush()
    def tell(self):return self.stream.tell()


class CheckpointStore:
    """One atomic latest checkpoint; no checkpoint search or silent rollback."""
    def __init__(self,path,budget,config_identity,*,max_bytes,generators=None):
        if not _integer(max_bytes):raise ValueError('Positive checkpoint byte bound required')
        self.path,self.budget,self.config_identity=Path(path).absolute(),budget,deepcopy(config_identity)
        _json(config_identity)
        self.max_bytes,self.generators=max_bytes,dict(generators or {})

    def save(self,model,optimizer,logical_steps=None,extra_state=None):
        try:return self._save(model,optimizer,logical_steps,extra_state)
        except BaseException as error:
            self.budget.fail('checkpoint_save_failed: '+type(error).__name__+': '+str(error));raise

    def _save(self,model,optimizer,logical_steps,extra_state):
        import torch
        with self.budget._lock:
            self.budget.check();boundary=self.budget._boundary();state=self.budget.snapshot()
            _checkpoint_schedule_extra(self.budget,extra_state)
            if state['requires_checkpoint_restore']:
                raise AdaptationRuntimeError('Recovered account cannot save fresh weights before coherent checkpoint restore')
            steps=state['completed_updates'] if logical_steps is None else logical_steps
            if type(steps) is not int or steps!=state['completed_updates']:raise AdaptationRuntimeError('Logical steps differ from durable completed updates')
            current=state['checkpoint']
            if current is not None and current['path']!=str(self.path):
                raise AdaptationRuntimeError('One account cannot publish a second checkpoint path')
            if self.path.exists() and (current is None or current['path']!=str(self.path) or _sha(self.path.read_bytes())!=current['sha256']):
                raise AdaptationRuntimeError('Refusing to overwrite an untracked or changed checkpoint')
            if state['selection_count']:raise AdaptationRuntimeError('The single selected checkpoint cannot change')
            temporary=self.path.with_name(self.path.name+'.pending-'+uuid.uuid4().hex)
            try:
                payload=dict(kind='gpc_adaptation_checkpoint.v1',model=_tensor_copy(model.state_dict(),torch),
                    optimizer=_tensor_copy(optimizer.state_dict(),torch),rng=_capture_rng(torch,self.generators),
                    optimizer_signature=_optimizer_signature(model,optimizer),
                    logical_steps=steps,budget=boundary,attempt=self.budget.attempt,config_identity=self.config_identity,
                    extra_state=_tensor_copy(extra_state,torch))
                self.path.parent.mkdir(parents=True,exist_ok=True)
                with temporary.open('xb',buffering=0) as stream:
                    torch.save(payload,_BoundedWriter(stream,self.max_bytes));os.fsync(stream.fileno())
                self.budget.check()
                if self.path.exists():os.replace(temporary,self.path)
                else:os.rename(temporary,self.path)
                digest=_sha(self.path.read_bytes())
                self.budget._operation('checkpoint_committed',dict(path=str(self.path),sha256=digest,boundary=boundary))
                return dict(path=str(self.path),sha256=digest,bytes=self.path.stat().st_size,logical_steps=steps)
            except BaseException as error:
                self.budget.fail('checkpoint_save_failed: '+type(error).__name__+': '+str(error));raise

    def restore(self,model,optimizer):
        try:return self._restore(model,optimizer)
        except BaseException as error:
            self.budget.fail('checkpoint_restore_failed: '+type(error).__name__+': '+str(error));raise

    def _restore(self,model,optimizer):
        import torch
        with self.budget._lock:
            self.budget.check();boundary=self.budget._boundary();current=self.budget.snapshot()['checkpoint']
            if (current is None or current['path']!=str(self.path) or current['boundary']!=boundary
                    or _sha(self.path.read_bytes())!=current['sha256']):
                raise AdaptationRuntimeError('Spent highwater and last committed checkpoint differ; no replay/refund')
            payload=torch.load(self.path,map_location='cpu',weights_only=True)
            if (payload.get('kind')!='gpc_adaptation_checkpoint.v1' or payload.get('budget')!=boundary
                    or payload.get('attempt')!=self.budget.attempt or payload.get('config_identity')!=self.config_identity
                    or payload.get('logical_steps')!=self.budget.snapshot()['completed_updates']
                    or payload.get('optimizer_signature')!=_optimizer_signature(model,optimizer)):
                raise AdaptationRuntimeError('Checkpoint/config/account/logical steps are not one coherent boundary')
            _checkpoint_schedule_extra(self.budget,payload.get('extra_state'))
            old_model=_tensor_copy(model.state_dict(),torch);old_optimizer=_tensor_copy(optimizer.state_dict(),torch);old_rng=_capture_rng(torch,self.generators)
            try:
                model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer']);_restore_rng(payload['rng'],torch,self.generators)
                self.budget.check()
            except BaseException as error:
                try:model.load_state_dict(old_model,strict=True);optimizer.load_state_dict(old_optimizer);_restore_rng(old_rng,torch,self.generators)
                except BaseException as secondary:error.add_note('Secondary restore rollback failure: '+str(secondary))
                self.budget.fail('checkpoint_restore_failed: '+type(error).__name__+': '+str(error));raise
            self.budget._requires_restore=False
            return dict(logical_steps=payload['logical_steps'],extra_state=payload['extra_state'],sha256=current['sha256'])

    def load_selected_for_evaluation(self,model):
        """Load selected model weights only; retain current counters, optimizer and RNG."""
        import torch
        with self.budget._lock:
            try:
                self.budget.check();state=self.budget.snapshot();current=state['checkpoint'];pending=state['pending']
                role=pending.get('role') if pending and pending['kind']=='model_load' else state['active_model_role']
                if (role!='adapted' or state['selection_count']!=1 or current is None
                        or current['path']!=str(self.path) or state['selected_checkpoint_sha256']!=current['sha256']
                        or _sha(self.path.read_bytes())!=current['sha256']):
                    raise AdaptationRuntimeError('Adapted evaluation requires the sole selected immutable checkpoint')
                payload=torch.load(self.path,map_location='cpu',weights_only=True)
                if (payload.get('kind')!='gpc_adaptation_checkpoint.v1' or payload.get('attempt')!=self.budget.attempt
                        or payload.get('config_identity')!=self.config_identity
                        or payload.get('logical_steps')!=state['checkpoint_committed_updates']):
                    raise AdaptationRuntimeError('Selected evaluation weights differ from their committed training identity')
                _checkpoint_schedule_extra(self.budget,payload.get('extra_state'))
                old=_tensor_copy(model.state_dict(),torch)
                try:model.load_state_dict(payload['model'],strict=True)
                except BaseException:
                    model.load_state_dict(old,strict=True);raise
                self.budget.check()
                return dict(logical_steps=payload['logical_steps'],sha256=current['sha256'],counts_restored=False,rng_restored=False)
            except BaseException as error:
                self.budget.fail('selected_evaluation_load_failed: '+type(error).__name__+': '+str(error));raise


def native_resource_preflight(output_dir, minima, *, device_index=0):
    """Future explicit Windows-only entry; never invoked by import or planning."""
    import sys
    from . import policy_memory
    if sys.platform!='win32':raise RuntimeError('This native resource preflight requires Windows')
    fields={'available_ram_bytes','commit_headroom_bytes','gpu_free_bytes','disk_free_bytes'}
    if set(minima)!=fields or any(not _integer(v) for v in minima.values()) or not _integer(device_index,0):
        raise ValueError('Explicit positive RAM/commit/GPU/disk minima and device index required')
    system=policy_memory.system_memory();gpu=policy_memory.gpu_memory(timeout=3);disk=policy_memory.disk_snapshot(output_dir)
    devices=[r for r in gpu.get('devices',[]) if r.get('index')==device_index] if gpu.get('ok') else []
    snapshot=dict(available_ram_bytes=system.get('physical',{}).get('available_bytes') if system.get('ok') else None,
        commit_headroom_bytes=system.get('commit',{}).get('headroom_bytes') if system.get('ok') else None,
        gpu_free_bytes=devices[0].get('free_bytes') if len(devices)==1 else None,
        disk_free_bytes=disk.get('free_bytes') if disk.get('ok') else None)
    failures=[name for name,minimum in minima.items() if type(snapshot.get(name)) is not int or snapshot[name]<minimum]
    return dict(passed=not failures,snapshot=snapshot,failures=failures,native=True,executed_by_import=False,
        errors=dict(system=system.get('error'),gpu=gpu.get('error'),disk=disk.get('error')))
