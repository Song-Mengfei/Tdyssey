"""One explicitly bounded raw-RGB training account; no resume or selection."""
from copy import deepcopy
import hashlib
from pathlib import Path
import time
import uuid

from .gpc_adaptation_runtime import AdaptationBudget, AdaptationRuntimeError, _json, _exclusive

ROLES = ('train_init', 'initial', 'raw500', 'raw2328')


class RawTrainingBudget(AdaptationBudget):
    @classmethod
    def create(cls, directory, binding, *, entries, case_keys, clock=None, monotonic=None):
        from .gpc_raw_fulltrain_entry import CANDIDATE, AUTHORIZATION
        if binding.get('candidate_id') != CANDIDATE or binding.get('authorization_id') != AUTHORIZATION:
            raise PermissionError('Exact raw fulltrain authority required')
        if len(entries) != 2328 or len({(r['trajectory_id'], r['anchor']) for r in entries}) != 2328:
            raise ValueError('Exactly2328 unique ordered train windows required')
        if len(case_keys) != 16 or len(set(case_keys)) != 16:
            raise ValueError('Exactly16 frozen development windows required')
        clock = clock or time.time; monotonic = monotonic or time.monotonic
        wall,mono = clock(),monotonic(); path = Path(directory).absolute()
        if wall >= binding['deadline_unix'] or mono >= binding['deadline_monotonic']:
            raise AdaptationRuntimeError('Original raw deadline expired')
        attempt = dict(kind='gpc_raw_fulltrain_account.v1',budget_id=uuid.uuid4().hex,
            binding=deepcopy(binding),binding_sha256=hashlib.sha256(_json(binding)).hexdigest(),
            ledger_dir=str(path),started_unix=wall,started_monotonic=mono,
            deadline_unix=binding['deadline_unix'],deadline_monotonic=binding['deadline_monotonic'],
            limits=dict(max_updates=2328,max_train_examples=2328,max_eval_batches=48,max_eval_examples=48,
                max_batch_size=1,deadline_seconds=1800,max_forecasts=48,max_euler_steps=144,max_model_loads=4))
        path.mkdir(parents=True,exist_ok=False); _exclusive(path/'attempt.json',attempt)
        self = cls._initialize(path,attempt,clock,monotonic)
        self.entries = deepcopy(entries); self.case_keys = tuple(case_keys)
        self._state.update(active_models=0, released_model_roles=[], exported_updates=[],
            training_complete=False, evaluation_complete=False,
            per_role={r: dict(loss=0, forecasts=0) for r in ROLES[1:]})
        self._append('created',{})
        return self

    @classmethod
    def resume(cls, *args, **kwargs):
        raise AdaptationRuntimeError('Raw fulltrain has no resume, retry or optimizer restore')

    def _append(self, kind, data):
        if len(_json(data)) > 8192 or self._journal_bytes+len(_json(data))+131072 > 16*1024**2:
            raise AdaptationRuntimeError('Raw ledger fixed16MiB admission exceeded')
        return super()._append(kind, data)

    def fail(self, reason):
        # The heartbeat keeps the original exception/traceback in its fixed slot.
        return super().fail(str(reason)[:4000])

    def _transition(self, kind, data, sequence):
        s = self._state
        if kind in ('created', 'failed'): return super()._transition(kind, data, sequence)
        if s['status'] != 'active': raise AdaptationRuntimeError('Stopped account cannot continue')
        if kind == 'reserve_model_load':
            index = s['spent_reserved_model_loads']; role = data.get('role')
            if (index >= 4 or s['pending'] is not None or s['active_models']
                    or role != ROLES[index] or s['released_model_roles'] != list(ROLES[:index])
                    or (index and not s['training_complete'])):
                raise AdaptationRuntimeError('Only four declared serial loads are admitted')
            s['spent_reserved_model_loads'] += 1; s['model_load_roles'].append(role)
            s['active_models'] = 1; s['active_model_role'] = role
            s['pending'] = dict(operation_id=sequence, kind='model_load', role=role)
            return
        if kind == 'complete_model_load':
            p = s['pending']
            if not p or p['kind'] != 'model_load' or p['operation_id'] != data.get('operation_id'):
                raise AdaptationRuntimeError('Load completion differs')
            s['completed_model_loads'] += 1; s['pending'] = None
            return
        if kind == 'reserve_update':
            cursor = s['spent_train_examples']; meta = data.get('metadata', {})
            if (s['active_model_role'] != 'train_init' or s['pending'] is not None
                    or data.get('batch_size') != 1 or cursor >= 2328 or s['training_complete']
                    or meta != dict(cursor=cursor, entry=self.entries[cursor])):
                raise AdaptationRuntimeError('Training differs from the frozen unique-window order')
            s['spent_reserved_updates'] += 1; s['spent_train_examples'] += 1
            s['pending'] = dict(operation_id=sequence, kind='update', batch_size=1, metadata=deepcopy(meta))
            return
        if kind in ('complete_update', 'complete_eval', 'complete_forecast'):
            p = s['pending']; wanted = kind.removeprefix('complete_')
            if not p or p['kind'] != wanted or p['operation_id'] != data.get('operation_id'):
                raise AdaptationRuntimeError('Completion differs from pending charged operation')
            if wanted == 'update': s['completed_updates'] += 1
            elif wanted == 'eval': s['completed_eval_batches'] += 1; s['per_role'][p['role']]['loss'] += 1
            else:
                s['completed_forecasts'] += 1; s['completed_euler_steps'] += 3
                s['per_role'][p['role']]['forecasts'] += 1
            s['pending'] = None
            return
        if kind in ('reserve_eval', 'reserve_forecast'):
            role = s['active_model_role']; meta = data.get('metadata', {})
            if role not in ROLES[1:] or s['pending'] is not None or s['active_models'] != 1:
                raise AdaptationRuntimeError('Development work requires its one active evaluation role')
            forecast = kind == 'reserve_forecast'; counter = 'forecasts' if forecast else 'loss'
            ordinal = s['per_role'][role][counter]
            if (ordinal >= 16 or meta != dict(role=role, example=ordinal, case_id=self.case_keys[ordinal])
                    or (forecast and (data.get('actual_count') != 1 or data.get('euler_per_image') != 3))
                    or (not forecast and data.get('batch_size') != 1)):
                raise AdaptationRuntimeError('Development reservation differs from fixed16 cases')
            if forecast:
                if s['per_role'][role]['loss'] != 16: raise AdaptationRuntimeError('Loss phase must finish first')
                s['spent_reserved_forecasts'] += 1; s['spent_reserved_euler_steps'] += 3
            else:
                if s['per_role'][role]['forecasts']: raise AdaptationRuntimeError('Loss cannot follow forecasts')
                s['spent_reserved_eval_batches'] += 1; s['spent_eval_examples'] += 1
            s['pending'] = dict(operation_id=sequence, kind='forecast' if forecast else 'eval', role=role)
            return
        if kind == 'exported':
            update = data.get('updates'); expected = (500, 2328)[len(s['exported_updates'])] if len(s['exported_updates']) < 2 else None
            if (s['pending'] is not None or s['active_model_role'] != 'train_init'
                    or update != expected or update != s['completed_updates']):
                raise AdaptationRuntimeError('Only immutable500/2328 export boundaries are admitted')
            s['exported_updates'].append(update)
            return
        if kind == 'training_finished':
            if (s['pending'] is not None or s['completed_updates'] != 2328
                    or s['spent_train_examples'] != 2328 or s['exported_updates'] != [500,2328]):
                raise AdaptationRuntimeError('Incomplete training cannot be final')
            s['training_complete'] = True
            return
        if kind == 'model_released':
            role = data.get('role')
            if (s['pending'] is not None or role != s['active_model_role'] or s['active_models'] != 1
                    or (role == 'train_init' and not s['training_complete'])
                    or (role in ROLES[1:] and s['per_role'][role] != dict(loss=16, forecasts=16))):
                raise AdaptationRuntimeError('Cannot release an incomplete role as successful')
            s['active_models'] = 0; s['active_model_role'] = None; s['released_model_roles'].append(role)
            return
        if kind == 'evaluation_finished':
            if (s['pending'] is not None or s['active_models'] or s['released_model_roles'] != list(ROLES)
                    or s['completed_updates'] != 2328 or s['completed_eval_batches'] != 48
                    or s['completed_forecasts'] != 48 or s['completed_euler_steps'] != 144):
                raise AdaptationRuntimeError('Partial raw run cannot be complete')
            s['evaluation_complete'] = True
            return
        raise AdaptationRuntimeError('Operation outside the fixed raw protocol: '+str(kind))

    def exported(self, updates): return self._operation('exported', dict(updates=updates))
    def training_finished(self): return self._operation('training_finished', {})
    def release_model(self, role): return self._operation('model_released', dict(role=role))
    def finish(self): return self._operation('evaluation_finished', {})
