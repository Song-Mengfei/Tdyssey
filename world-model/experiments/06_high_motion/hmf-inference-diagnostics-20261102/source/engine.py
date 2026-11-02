"""Bounded A/B inference only; the sole low-priority worker calls run.

No model or Torch operation occurs on import. Original native FP32 sampler,
batch slots, per-step seeds and raw recursive feedback are retained. A failure
ends this execution: no retry, replay, training or implicit warmup exists here.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

SAMPLER = dict(num_steps_denoising=3, sigma_min=.002, sigma_max=5., rho=7,
               order=1, s_churn=0., s_cond=0.)
PREDICTION_CAP = 3672
IMAGE_FORWARD_CAP = 11016
SECTION_CAPS = {'A': 3456, 'B': 216}
LOAD_ORDER = ('250', '500', '1000', '2000', 'E2', 'final')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def input_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def cpu_receipt(torch):
    """Observe only this worker. Do not change any other process."""
    if os.name != 'nt':
        raise RuntimeError('Approved local Windows worker required')
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetPriorityClass.argtypes = [ctypes.c_void_p]
    priority = int(kernel.GetPriorityClass(kernel.GetCurrentProcess()))
    names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
    env = {k: os.environ.get(k) for k in names}
    if priority != 0x4000 or any(v != '1' for v in env.values()):
        raise RuntimeError('Worker must actually be BelowNormal with CPU thread env1')
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError('Torch intra/inter-op threads must both equal1')
    return dict(pid=os.getpid(), priority_class=priority, priority='BelowNormal',
                thread_environment=env, torch_intra_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),
                cpu_model_fallback=False, multiprocessing=False)


class Budget:
    """Atomically reserve before sampling and before each denoiser forward.

    Reserved predictions are never relabelled completed after a partial error.
    Hook attempts/completions are separate and include every model forward.
    """
    def __init__(self, root, c):
        from common import atomic_json
        self.root = root; self.c = c; self.write = atomic_json
        self.path = root/'inference/budget.json'
        self.counts = Counter()
        self.sections = {k: Counter() for k in SECTION_CAPS}
        self.active = None; self.last_call = None; self.section = None
        self.journal = (root/'inference/reservations.jsonl').open('x', encoding='utf-8', buffering=1)
        self.persist()

    def state(self):
        return dict(run_id=self.c['run_id'], config_sha256=self.c['_runtime_config_sha256'],
                    caps=dict(predictions=PREDICTION_CAP, image_denoiser_forwards=IMAGE_FORWARD_CAP,
                              section_predictions=SECTION_CAPS), counts=dict(self.counts),
                    sections={k: dict(v) for k, v in self.sections.items()},
                    active_call=deepcopy(self.active), last_call=deepcopy(self.last_call),
                    optimizer_updates=0, backward_calls=0, physical_steps=0,
                    retry_or_replay_permitted=False)

    def persist(self):
        self.write(self.path, self.state())

    def begin(self, record, step, seed, n):
        if self.active is not None or n not in (3, 4):
            raise RuntimeError('Unclosed inference reservation or changed B4/B3 batch')
        section = record['section']
        if (self.counts['reserved_predictions']+n > PREDICTION_CAP or
            self.counts['reserved_image_forwards']+3*n > IMAGE_FORWARD_CAP or
            self.sections[section]['reserved_predictions']+n > SECTION_CAPS[section]):
            raise RuntimeError('Inference hard budget exhausted before prediction')
        self.section = section
        for counter in (self.counts, self.sections[section]):
            counter['reserved_predictions'] += n
            counter['reserved_image_forwards'] += 3*n
            counter['sampler_attempts'] += 1
        self.active = dict(call_id=self.counts['sampler_attempts'], section=section,
                           checkpoint=record['label'], batch_id=record['batch_id'],
                           mode=record['mode'], action_type=record.get('action_type'),
                           noise_id=record.get('noise_id'), step=step, seed=int(seed),
                           batch_size=n, forward_attempts=0, forward_completed=0,
                           stage='reserved_before_sampler')
        self.persist()
        self.journal.write(json.dumps(self.active, separators=(',', ':'))+'\n')

    def before_forward(self, module, args):
        a = self.active
        if a is None or len(args[0]) != a['batch_size'] or a['forward_attempts'] >= 3:
            self.counts['blocked_unreserved_forward_requests'] += 1
            self.persist()
            raise RuntimeError('Unexpected or unreserved denoiser forward')
        n = len(args[0])
        if self.counts['image_forward_attempts']+n > IMAGE_FORWARD_CAP:
            raise RuntimeError('Denoiser image-forward hard cap reached')
        a['forward_attempts'] += 1; a['stage'] = 'denoiser_forward_pending'
        for counter in (self.counts, self.sections[self.section]):
            counter['batch_forward_attempts'] += 1
            counter['image_forward_attempts'] += n
        # Persist BEFORE each actual inner_model forward, including its budget.
        self.persist()

    def after_forward(self, module, args, output):
        if self.active is None:
            raise RuntimeError('Completed unreserved model forward')
        self.active['forward_completed'] += 1
        self.active['stage'] = 'denoiser_forward_returned'
        for counter in (self.counts, self.sections[self.section]):
            counter['batch_forward_completed'] += 1
            counter['image_forward_completed'] += len(args[0])

    def complete(self):
        a = self.active
        if a is None or a['forward_attempts'] != 3 or a['forward_completed'] != 3:
            raise RuntimeError('Native sampler did not perform exactly3 denoiser forwards')
        for counter in (self.counts, self.sections[self.section]):
            counter['completed_predictions'] += a['batch_size']
            counter['sampler_completed'] += 1
        a['stage'] = 'finite_native_prediction_completed'
        self.last_call = a; self.active = None; self.persist()

    def close(self):
        self.journal.flush(); self.journal.close()


@contextmanager
def observe_native_epsilon(torch, sink):
    """Observe the actual native draw unchanged, with no replay/random draw."""
    original = torch.randn; captured = []
    def observed(*args, **kwargs):
        value = original(*args, **kwargs)
        if captured:
            raise RuntimeError('Native sampler unexpectedly drew more than one randn tensor')
        if value.dtype != torch.float32 or tuple(value.shape[1:]) != (3, 96, 96):
            raise RuntimeError('Initial epsilon shape/dtype changed')
        array = value.detach().cpu().contiguous().numpy()
        proof = dict(actual_initial_epsilon_sha256=array_sha(array),
                     per_case_initial_epsilon_sha256=[array_sha(row) for row in array],
                     shape=list(array.shape), dtype=str(array.dtype),
                     extra_random_draws=0, original_value_returned=True)
        captured.append(proof); sink.append(proof)
        return value
    torch.randn = observed
    try:
        yield
        if len(captured) != 1:
            raise RuntimeError('Native sampler initial epsilon draw missing')
    finally:
        torch.randn = original


class Session:
    def __init__(self, root, c):
        from common import atomic_json
        self.root = root; self.c = c; self.write = atomic_json
        self.started = time.perf_counter(); self.cpu = None; self.torch = None
        self.data = None; self.model = None; self.prehook = None; self.posthook = None
        self.model_construction_attempts = 0; self.model_constructions = 0
        self.loads = []; self.load_attempts = []; self.checkpoint = None
        self.version_snapshot = None; self.readonly_proofs = []
        self.budget = Budget(root, c)
        self.batches = []; self.first_batch = None; self.timings = Counter()
        self.epsilon_pairs = {}; self.B_input_pairs = {}; self.pending_partial = None
        self.source_stats = c['_validated_source_stats']

    def initialize(self):
        import torch
        from data import DiagnosticData
        from tdyssey.gpc_world_model import load_world_model_source, world_model_spec
        self.torch = torch; self.cpu = cpu_receipt(torch)
        if not torch.cuda.is_available():
            raise RuntimeError('Approved CUDA unavailable; no CPU-model fallback')
        self.device = torch.device('cuda'); self.device_index = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats()
        self.data = DiagnosticData(self.root, self.c)
        self.source = load_world_model_source(self.c['source_root'])
        inner = deepcopy(world_model_spec()['model'])
        outer = {k: inner.pop(k) for k in ('sigma_data','sigma_offset_noise','noise_previous_obs')}
        self.model_construction_attempts += 1
        self.write(self.root/'inference/construction.json',dict(attempts=1,completed=0))
        with torch.random.fork_rng(devices=[]):
            self.model = self.source.Denoiser(self.source.DenoiserConfig(
                inner_model=self.source.InnerModelConfig(**inner), **outer)).float()
        self.model.to(self.device).eval().requires_grad_(False)
        self.model_constructions = 1
        self.write(self.root/'inference/construction.json',dict(attempts=1,completed=1))
        self.sampler = self.source.DiffusionSampler(self.model, self.source.DiffusionSamplerConfig(**SAMPLER))
        self.prehook = self.model.inner_model.register_forward_pre_hook(self.budget.before_forward)
        self.posthook = self.model.inner_model.register_forward_hook(self.budget.after_forward)

    def source_stat(self, info):
        q = Path(info['path']).resolve(); expected = self.source_stats.get(str(q))
        if expected is None or expected['sha256'] != info['sha256']:
            raise RuntimeError('Checkpoint was not hash-validated by unique worker')
        stat = q.stat()
        if (stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns'] or
            ('ctime_ns' in expected and stat.st_ctime_ns != expected['ctime_ns'])):
            raise RuntimeError('Frozen checkpoint stat changed after worker validation')
        return dict(path=str(q), sha256=expected['sha256'], bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)

    def load(self, label, info):
        if len(self.loads) >= 6 or label != LOAD_ORDER[len(self.loads)]:
            raise RuntimeError('Unapproved checkpoint load/retry/order')
        self.verify_readonly()
        started = time.perf_counter(); identity = self.source_stat(info)
        self.load_attempts.append(dict(identity,label=label,completed=False))
        self.write(self.root/'inference/checkpoint-load-attempts.json',self.load_attempts)
        # mmap avoids reading/restoring E2 optimizer/scaler/RNG tensors.
        payload = self.torch.load(identity['path'], map_location='cpu', weights_only=False, mmap=True)
        kind = payload.get('kind')
        if label == 'E2':
            progress = payload.get('progress', {})
            if (kind != 'oc_native_rgb_complete.v1' or progress.get('epoch') != 2 or
                progress.get('lineage_updates') != 11367 or not progress.get('epoch_complete') or
                not progress.get('complete_optimizer_boundary') or progress.get('pending_attempt') is not None):
                raise RuntimeError('B baseline is not approved complete epoch2 model')
        elif kind != 'high_motion_fit_weights.v1':
            raise RuntimeError('Only actual HMF lightweight milestone model weights permitted')
        self.model.load_state_dict(payload['model'], strict=True)
        del payload
        self.model.eval().requires_grad_(False)
        if any(x.dtype != self.torch.float32 for x in self.model.parameters()):
            raise RuntimeError('Model parameters must remain native FP32')
        self.source_stat(info)
        self.checkpoint = dict(identity, label=label, model_state_only=True,
                               optimizer_loaded=False, RNG_restored=False)
        self.version_snapshot = self.versions()
        self.loads.append(dict(self.checkpoint, wall_seconds=time.perf_counter()-started))
        self.load_attempts[-1]['completed'] = True
        self.write(self.root/'inference/checkpoint-load-attempts.json',self.load_attempts)
        self.timings['checkpoint_load_seconds'] += self.loads[-1]['wall_seconds']
        self.write(self.root/'inference/checkpoint-loads.json', self.loads)

    def versions(self):
        # CPU metadata only: no per-parameter synchronization or tensor copy.
        return {name: value._version for name,value in
                list(self.model.named_parameters())+list(self.model.named_buffers())}

    def verify_readonly(self):
        if self.version_snapshot is None:
            return
        unchanged = self.versions() == self.version_snapshot
        proof = dict(checkpoint=self.checkpoint['label'],parameter_buffer_versions_unchanged=unchanged,
                     requires_grad_false=not any(x.requires_grad for x in self.model.parameters()),
                     model_eval_mode=not self.model.training,optimizer_objects_constructed=0,backward_calls=0)
        self.readonly_proofs.append(proof)
        self.write(self.root/'inference/model-readonly.json',self.readonly_proofs)
        if not unchanged or not proof['requires_grad_false'] or not proof['model_eval_mode']:
            raise RuntimeError('Inference modified model state/mode/grad contract')

    def input_evidence(self, inputs):
        initial = inputs['initial_histories']; absolute = inputs['absolute_actions']
        proof = dict(initial_history_sha256=input_sha(initial),
                     absolute_actions_sha256=input_sha(absolute),
                     normalized_actions_sha256=input_sha(absolute.astype(np.float32)/np.float32(255.5)-np.float32(1)),
                     shape_history=list(initial.shape), shape_actions=list(absolute.shape))
        if 'real_histories' in inputs:
            proof['per_target_true_histories_sha256'] = input_sha(inputs['real_histories'])
        return proof

    def predict(self, record, inputs, seeds):
        torch = self.torch; steps = len(seeds); n = len(record['rows'])
        initial = inputs['initial_histories']; absolute = inputs['absolute_actions']
        if (n not in (3,4) or initial.dtype != np.float32 or initial.shape != (n,4,3,96,96) or
            absolute.shape != (n,steps+3,2) or not np.isfinite(absolute).all() or
            absolute.min() < 0 or absolute.max() > 511):
            raise ValueError('Frozen RGB/action/batch contract differs')
        if not np.isfinite(initial).all() or initial.min() < 0 or initial.max() > 1:
            raise ValueError('True initial history outside original RGB contract')
        if (record['section'] == 'A' and steps != 16) or (record['section'] == 'B' and steps != 1):
            raise ValueError('Only frozen A16/B1 inference permitted')
        stamp = time.perf_counter()
        history = torch.as_tensor(initial.copy(), device=self.device, dtype=torch.float32)
        # Exact existing CPU float32 expression, never CUDA division substitution.
        normalized = absolute.astype(np.float32)/np.float32(255.5)-np.float32(1)
        actions = torch.as_tensor(normalized, device=self.device)
        real = None
        if 'real_histories' in inputs:
            if inputs['real_histories'].shape != (n,steps,4,3,96,96):
                raise ValueError('Real-history independent target slices differ')
            real = torch.as_tensor(inputs['real_histories'].copy(), device=self.device, dtype=torch.float32)
        if (record['mode'] == 'real_history') != (real is not None) and record['section'] == 'A':
            raise ValueError('RH/AR history contract differs')
        self.timings['input_H2D_submit_seconds'] += time.perf_counter()-stamp
        outputs = []; noises = []
        self.pending_partial = dict(record=record, gpu_outputs=outputs, noise_evidence=noises)
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            for step, seed in enumerate(seeds):
                if real is not None: history = real[:,step]
                self.budget.begin(record, step+1, int(seed), n)
                stamp = time.perf_counter(); observed = []
                with torch.random.fork_rng(devices=[self.device_index]):
                    torch.random.default_generator.manual_seed(int(seed))
                    torch.cuda.default_generators[self.device_index].manual_seed(int(seed))
                    with observe_native_epsilon(torch, observed):
                        raw, _ = self.sampler.sample(history.contiguous(), actions[:,step:step+4].contiguous())
                raw = raw.detach().float()
                if raw.shape != (n,3,96,96) or not bool(torch.isfinite(raw).all()):
                    raise FloatingPointError('Native raw prediction nonfinite/shape fault')
                self.budget.complete()
                elapsed = time.perf_counter()-stamp
                self.timings['native_sampler_and_noise_evidence_seconds'] += elapsed
                noise = dict(step=step+1, seed=int(seed), batch_size=n, **observed[0])
                noises.append(noise); outputs.append(raw)
                self.check_noise_pair(record, step, noise)
                if self.first_batch is None:
                    self.first_counted_batch(record, elapsed, n)
                if real is None and step+1 < steps:
                    history = torch.cat((history[:,1:], raw[:,None]), dim=1)
        return torch.stack(outputs, dim=1), noises

    def check_noise_pair(self, record, step, noise):
        if record['section'] == 'A':
            expected = record['expected_noise_evidence'][step]
            if (noise['actual_initial_epsilon_sha256'] != expected['actual_initial_epsilon_sha256'] or
                noise['per_case_initial_epsilon_sha256'] != expected['per_case_initial_epsilon_sha256'] or
                noise['seed'] != expected['seed']):
                raise RuntimeError('A actual native epsilon differs from frozen HMF baseline')
        else:
            key = (record['batch_id'], record['noise_id'])
            value = noise['actual_initial_epsilon_sha256']
            if key in self.epsilon_pairs and value != self.epsilon_pairs[key]:
                raise RuntimeError('B same-noise action/model pairing changed')
            self.epsilon_pairs[key] = value

    def first_counted_batch(self, record, elapsed, n):
        torch = self.torch
        allocated = torch.cuda.max_memory_allocated(); reserved = torch.cuda.max_memory_reserved()
        if max(allocated,reserved) > int(self.c.get('max_cuda_bytes',6*1024**3)):
            raise RuntimeError('CUDA memory envelope exceeded; no fallback/retry')
        remaining = PREDICTION_CAP-self.budget.counts['completed_predictions']
        now = time.time()
        self.first_batch = dict(run_id=self.c['run_id'], config_sha256=self.c['_runtime_config_sha256'],
            measured_at_unix=now, pid=os.getpid(), first_counted_sampler_batch=True,
            checkpoint=record['label'], section=record['section'], batch_id=record['batch_id'],
            batch_size=n, predicted_frames=n, image_denoiser_forwards=n*3,
            native_sampler_wall_seconds=elapsed, frames_per_second=n/elapsed,
            image_denoiser_forwards_per_second=3*n/elapsed,
            startup_through_first_batch_seconds=time.perf_counter()-self.started,
            peak_cuda_allocated_bytes=allocated, peak_cuda_reserved_bytes=reserved,
            CPU=self.cpu, I_O=dict(self.timings), budget=self.budget.state(),
            remaining_predictions=remaining,
            crude_remaining_native_sampler_seconds_range=[.5*remaining*elapsed/n,2*remaining*elapsed/n],
            uncertainty='Only first counted B4 draw, includes first-call/noise-hash overhead. No extra warmup. Range excludes later CPU qualification/topology/axis/PNG/report/I-O/checkpoint costs; not a completion promise or time cap.')
        self.write(self.root/'first-batch.json', self.first_batch)

    def batch(self, record, inputs, seeds):
        from common import save_raw, atomic_json
        from metrics import score_batch
        started = time.perf_counter()
        record = dict(record, checkpoint=deepcopy(self.checkpoint), seeds=list(map(int,seeds)),
                      batch_slots=list(range(len(record['rows']))), input_evidence=self.input_evidence(inputs))
        proof = record['input_evidence']
        proof.update(sample_ids=[x['sample_id'] for x in record['rows']],
                     family_ids=[x['family_id'] for x in record['rows']],seeds=record['seeds'])
        if record['section'] == 'A':
            expected = record['expected_input_evidence']
            if any(proof.get(k) != v for k,v in expected.items() if k not in ('mode','batch_id')):
                raise RuntimeError('A input/normalized action/history/seed differs from frozen HMF pairing')
        else:
            pair = (proof['initial_history_sha256'], input_sha(
                inputs['absolute_actions'][:,:3].astype(np.float32)/np.float32(255.5)-np.float32(1)))
            key = record['batch_id']
            if key in self.B_input_pairs and self.B_input_pairs[key] != pair:
                raise RuntimeError('B four-frame history or past3 model actions changed')
            self.B_input_pairs[key] = pair
        gpu, noise = self.predict(record, inputs, seeds)
        stamp = time.perf_counter(); raw = gpu.cpu().numpy()
        self.pending_partial['raw_numpy'] = raw
        self.timings['prediction_D2H_seconds'] += time.perf_counter()-stamp
        if raw.dtype != np.float32:
            raise RuntimeError('Original raw prediction dtype changed')
        stem = record['record_id']; path = self.root/'inference/raw'/f'{stem}.npz'
        stamp = time.perf_counter(); saved = save_raw(self.root,self.c,path,raw=raw)
        self.pending_partial['raw_saved'] = saved
        self.timings['raw_save_hash_seconds'] += time.perf_counter()-stamp
        record['raw_files'] = [dict(saved, sample_id=row['sample_id'], batch_slot=i,
                                    array_key='raw', raw_index=i) for i,row in enumerate(record['rows'])]
        record['noise_evidence'] = noise
        truth = self.data.truth_arrays(record['source_batch'], section=record['section'], mode=record['mode'])
        stamp = time.perf_counter(); scored = score_batch(self.root,self.c,record,raw,gpu=gpu,truth_arrays=truth)
        self.timings['scoring_seconds'] += time.perf_counter()-stamp
        record.pop('source_batch',None)
        scorepath = self.root/'inference/metrics'/f'{stem}.json'
        atomic_json(scorepath,scored)
        receipt = dict(record=record,score_path=str(scorepath),prediction=dict(saved,array_key='raw'),
                       metrics=str(scorepath),predicted_frames=int(raw.shape[0]*raw.shape[1]),
                       wall_seconds=time.perf_counter()-started)
        self.batches.append(receipt)
        # Write each batch once, rather than repeatedly serializing prior rows.
        atomic_json(self.root/'inference/records'/f'{stem}.json',receipt)
        atomic_json(self.root/'inference/progress.json',dict(
            run_id=self.c['run_id'],config_sha256=self.c['_runtime_config_sha256'],
            completed_batches=len(self.batches),last_record_id=stem,
            completed_predictions=self.budget.counts['completed_predictions'],
            image_forward_completed=self.budget.counts['image_forward_completed'],
            checkpoint_loads_completed=len(self.loads),timings=dict(self.timings)))
        self.budget.persist()
        self.pending_partial = None
        if max(self.torch.cuda.max_memory_allocated(),self.torch.cuda.max_memory_reserved()) > int(self.c.get('max_cuda_bytes',6*1024**3)):
            raise RuntimeError('CUDA envelope exceeded; saved partial retained')
        return receipt

    def close(self):
        if self.prehook is not None: self.prehook.remove()
        if self.posthook is not None: self.posthook.remove()
        self.budget.close()
        if hasattr(self.data,'close'): self.data.close()

    def save_partial_raw(self):
        """Best effort saved outputs from already-completed native steps only."""
        from common import save_raw
        pending = self.pending_partial
        if not pending or not pending['gpu_outputs']:
            return None
        record = pending['record']
        # This copies completed predictions; it makes no model/random call.
        raw = pending.get('raw_numpy')
        if raw is None:
            raw = np.stack([x.cpu().numpy() for x in pending['gpu_outputs']],axis=1)
        saved = pending.get('raw_saved')
        if saved is None:
            saved = save_raw(self.root,self.c,self.root/'inference/raw'/
                             (record['record_id']+'-PARTIAL.npz'),raw=raw)
        value = dict(record_id=record['record_id'],section=record['section'],
                     label=record['label'],shape=list(raw.shape),completed_steps=raw.shape[1],
                     prediction=saved,noise_evidence=pending['noise_evidence'],scored=False)
        self.write(self.root/'inference/partial-raw.json',value)
        return value


def run(root, c):
    """Unique worker entry. No automatic restart/retry is supported."""
    from common import atomic_json
    root = Path(root).resolve(); (root/'inference/raw').mkdir(parents=True,exist_ok=True)
    (root/'inference/metrics').mkdir(exist_ok=True)
    claim = root/'inference/execution-claim.json'
    with claim.open('x',encoding='utf-8') as f:
        json.dump(dict(run_id=c['run_id'],pid=os.getpid(),started_at_unix=time.time(),auto_retry=False),f)
    if not c.get('_validated_source_stats') or not c.get('_runtime_config_sha256'):
        raise RuntimeError('Only frozen-source-validated unique worker may invoke inference')
    a = read(root/'A-spec.json'); b = read(root/'B-spec.json')
    if a.get('sampler',SAMPLER) != SAMPLER or b.get('sampler',SAMPLER) != SAMPLER:
        raise ValueError('Only original native3 sampler allowed')
    if (a['checkpoint_labels'] != list(LOAD_ORDER[:4]) or a['input_modes'] != ['real_history','autoregressive'] or
        [x['batch_id'] for x in a['batches']] != list(range(12,19)) or
        [len(x['rows']) for x in a['batches']] != [4]*6+[3] or
        b['checkpoint_labels'] != ['E2','final'] or b['actions'] != ['recorded','hold','alternative'] or
        len(b['noise_sets']) != 3 or len(b['batches']) != 3 or any(len(x['rows']) != 4 for x in b['batches'])):
        raise ValueError('Frozen A/B case/checkpoint/action/noise/batch denominator differs')
    checkpoints = read(root/'checkpoint-map.json'); session = None; started = time.perf_counter()
    expected_inputs = {(x['mode'],x['batch_id']):x for x in a['expected_input_evidence']}
    expected_noise = {(x['mode'],x['batch_id']):x['steps'] for x in a['expected_noise_evidence']}
    try:
        session = Session(root,c)
        session.initialize()
        for label in LOAD_ORDER[:4]:
            session.load(label,checkpoints[label])
            for mode in ('real_history','autoregressive'):
                for batch in a['batches']:
                    stamp = time.perf_counter(); inputs = session.data.A_inputs(batch,mode)
                    session.timings['data_wait_seconds'] += time.perf_counter()-stamp
                    record = dict(section='A',label=label,mode=mode,rows=batch['rows'],
                                  batch_id=batch['batch_id'],source_batch=batch,
                                  expected_input_evidence=expected_inputs[(mode,batch['batch_id'])],
                                  expected_noise_evidence=expected_noise[(mode,batch['batch_id'])],
                                  record_id=f'A-{label}-{mode}-b{batch["batch_id"]:02d}')
                    session.batch(record,inputs,batch['seeds'])
        for label in LOAD_ORDER[4:]:
            session.load(label,checkpoints[label])
            for noise_id,noise_set in enumerate(b['noise_sets']):
                for batch_index,batch in enumerate(b['batches']):
                    for action_type in ('recorded','hold','alternative'):
                        stamp = time.perf_counter(); inputs = session.data.B_inputs(batch,action_type)
                        session.timings['data_wait_seconds'] += time.perf_counter()-stamp
                        seed = noise_set['batch_seeds'][str(batch['batch_id'])]
                        record = dict(section='B',label=label,mode='real_history',rows=batch['rows'],
                                      batch_id=batch['batch_id'],source_batch=batch,noise_id=noise_id,
                                      noise_name=noise_set['name'],
                                      action_type=action_type,
                                      counterfactual_GT_available=action_type=='recorded',
                                      record_id=f'B-{label}-n{noise_id}-b{batch["batch_id"]:02d}-{action_type}')
                        session.batch(record,inputs,[seed])
        state = session.budget.state(); counts = state['counts']
        session.verify_readonly()
        if (counts.get('completed_predictions') != PREDICTION_CAP or
            counts.get('image_forward_completed') != IMAGE_FORWARD_CAP or
            counts.get('image_forward_attempts') != IMAGE_FORWARD_CAP or
            counts.get('sampler_completed') != 950 or
            counts.get('batch_forward_completed') != 2850 or
            len(session.loads) != 6 or session.model_constructions != 1):
            raise RuntimeError('Actual completed inference denominator differs from frozen plan')
        result = dict(complete=True,run_id=c['run_id'],config_sha256=c['_runtime_config_sha256'],
                      budget=state,model_constructions=1,checkpoint_loads=session.loads,
                      model_construction_attempts=session.model_construction_attempts,
                      checkpoint_load_attempts=session.load_attempts,
                      model_readonly_proofs=session.readonly_proofs,
                      batches=session.batches,
                      A_batches=[x for x in session.batches if x['record']['section']=='A'],
                      B_batches=[x for x in session.batches if x['record']['section']=='B'],
                      first_batch=str(root/'first-batch.json'),
                      timings=dict(session.timings),CPU=session.cpu,
                      wall_seconds=time.perf_counter()-started,
                      reused=a['existing_HMF_metrics'],optimizer_updates=0,backward_calls=0,
                      physical_steps=0,new_data=0,automatic_next_experiment=False)
        atomic_json(root/'inference/result.json',result)
        return result
    except BaseException as exc:
        partial = None; partial_error = None
        if session is not None:
            try: partial = session.save_partial_raw()
            except BaseException as partial_exc: partial_error = repr(partial_exc)
        failure = dict(complete=False,run_id=c['run_id'],config_sha256=c.get('_runtime_config_sha256'),
                       error=repr(exc),wall_seconds=time.perf_counter()-started,
                       budget=session.budget.state() if session is not None else None,
                       checkpoint_loads=session.loads if session is not None else [],
                       checkpoint_load_attempts=session.load_attempts if session is not None else [],
                       model_construction_attempts=session.model_construction_attempts if session is not None else 0,
                       model_constructions=session.model_constructions if session is not None else 0,
                       completed_batches=session.batches if session is not None else [],
                       timings=dict(session.timings) if session is not None else {},
                       optimizer_updates=0,backward_calls=0,physical_steps=0,
                       uncertain_pending_call=bool(session is not None and session.budget.active),
                       partial_raw=partial,partial_save_error=partial_error,no_retry_or_replay=True)
        atomic_json(root/'inference/failure.json',failure)
        if session is not None: session.budget.persist()
        raise
    finally:
        if session is not None: session.close()
