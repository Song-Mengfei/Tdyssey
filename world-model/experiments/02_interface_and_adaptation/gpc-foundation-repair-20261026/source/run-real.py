"""One frozen diagnostic; no training, retries or numerical protocol changes."""
from collections import deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(ROOT))
START = time.monotonic()
PLAN = json.loads((BASE/'real-plan.json').read_bytes())
OUT = BASE/'real-01'
OUT.mkdir(exist_ok=False)
LOG = (OUT/'events.jsonl').open('x', encoding='utf-8')
account = dict(reset_reservations=0, control_reservations=0, control_returns=0,
               direct_forecast_reservations=0, direct_forecast_returns=0)
peaks = dict(private_bytes=0, working_set_bytes=0)
predictor = session = current_request = None

def write_json(name, value):
    with (OUT/name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

def log(kind, **value):
    LOG.write(json.dumps(dict(kind=kind, seconds=time.monotonic()-START, **value), allow_nan=False)+'\n')
    LOG.flush()

def resource(startup=False):
    from tdyssey.policy_memory import process_memory, system_memory, gpu_memory
    system, process = system_memory(), process_memory(os.getpid())
    if not system['ok'] or not process['ok']: raise RuntimeError('Host resource query failed')
    peaks['private_bytes'] = max(peaks['private_bytes'], process['private_bytes'], process['peak_pagefile_bytes'])
    peaks['working_set_bytes'] = max(peaks['working_set_bytes'], process['working_set_bytes'], process['peak_working_set_bytes'])
    if peaks['private_bytes'] > 3*1024**3 or peaks['working_set_bytes'] > 2*1024**3: raise MemoryError('Process ceiling')
    ram, commit = (4,6) if startup else (1.5,3)
    if system['physical']['available_bytes'] < ram*1024**3 or system['commit']['headroom_bytes'] < commit*1024**3:
        raise MemoryError('Host headroom floor')
    if startup:
        gpu = gpu_memory(timeout=5)
        if not gpu['ok'] or gpu['devices'][0]['free_bytes'] < 4*1024**3: raise MemoryError('Startup GPU floor/query')
        if shutil.disk_usage(ROOT).free < 2*1024**3: raise MemoryError('Startup disk floor')
        write_json('startup-resources.json', dict(system=system, process=process, gpu=gpu))
    elif 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        free, _ = sys.modules['torch'].cuda.mem_get_info()
        if free < 1.5*1024**3: raise MemoryError('Runtime GPU floor')
    if time.monotonic()-START > 280: raise TimeoutError('Diagnostic cleanup reserve reached')

try:
    resource(startup=True)
    freeze = json.loads((BASE/'real-freeze.json').read_bytes())
    for relative, digest in freeze['sources'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen diagnostic source changed: '+relative)
    paused = {'tdyssey/gpc_adaptation_guard.py','tdyssey/gpc_checkpoint_evaluation.py',
              'tdyssey/gpc_conditioning_contract.py','tests/test_gpc_conditioning_contract.py'}
    for entry in json.loads((BASE/'start.json').read_bytes())['dirty']:
        if entry['path'] in paused and hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Paused source changed')
    import numpy as np
    from tdyssey.manual_world_model import ManualSession, PinnedPredictor, make_environment, save_case_bundle, request_from_case
    from tdyssey.state import Snapshot
    from tdyssey.wm_compat_transform import transform_t1
    from tdyssey.wm_image_measurement_v2 import decode
    from tdyssey.wm_compat_engine import compact_measurement
    from PIL import Image, ImageDraw

    class CountedEnvironment:
        def __init__(self, env): self.env = env
        def __getattr__(self, name): return getattr(self.env, name)
        def step(self, action):
            if account['control_reservations'] >= 14: raise RuntimeError('Control reservation cap')
            account['control_reservations'] += 1
            result = self.env.step(action)
            account['control_returns'] += 1
            resource()
            return result

    def factory(shape):
        if account['reset_reservations'] >= 2: raise RuntimeError('Reset reservation cap')
        account['reset_reservations'] += 1
        env, obs = make_environment(shape)
        resource()
        return CountedEnvironment(env), obs

    records = [json.loads(line) for line in (ROOT/PLAN['source_log']).read_text(encoding='utf-8').splitlines()]
    session = ManualSession(factory)
    session.reset('L')
    predictor = PinnedPredictor()
    predictor.forecast_cap, predictor.model_load_cap = 16, 2
    original_check = predictor.check_cuda
    def check():
        original_check()
        resource()
    predictor.check_cuda = check
    cases, summary = {}, {}

    def transition(location):
        return deepcopy(records[location[0]-1]['steps'][location[1]])

    def rehydrate(case):
        declared = records[case['request_line']-1]
        if (declared['kind'] != 'prediction_requested' or declared['shape'] != 'L'
                or declared['seed'] != 5478 or declared['target'] != case['target']
                or declared['horizon'] != case['historical_horizon']):
            raise ValueError('Declared historical request differs')
        states = [transition(location)['after_state'] for location in case['history_state_steps']]
        steps = [transition(location) for location in case['history_transition_steps']]
        images = []
        for state in states:
            obs = session.env.restore_state(Snapshot.from_dict(state))
            images.append(obs['image'].copy())
        session.observation = obs
        session.images = deque(images, maxlen=4)
        session.actions = deque([np.asarray(step['action'],np.float32) for step in steps],maxlen=3)
        session.history_states = deque(states,maxlen=4)
        session.history_steps = deque(steps,maxlen=3)
        session.done = False
        session.invalidate()
        session.configure(role='frozen',horizon=4)
        session.select(case['target'])
        request = session.request()
        if request['initial_state'] != records[case['execution_line']-1]['steps'][0]['before_state']:
            raise ValueError('History endpoint differs from logged execution')
        log('rehydrated', case=case['name'], request_key=request['request_key'],
            historical_control_step=request['initial_state']['control_step'],
            historical_pixels_sealed=False, historical_horizon=case['historical_horizon'], requested_horizon=4)
        return request

    def measurements(raw, request):
        rows = []
        for index, frame in enumerate(raw):
            full = decode(frame, request['shape'], request['goal'])
            rows.append(dict(horizon=index+1, measurement=compact_measurement(full),pusher_diagnostics=full['pusher']))
            resource()
        return rows

    for case in PLAN['cases']:
        current_request = rehydrate(case)
        raw, measurement, seconds = predictor.predict(current_request)
        session.accept(current_request,raw,measurement)
        execution = session.execute()
        bundle = session.capture_case()
        cases[case['name']] = bundle
        summary[case['name']] = dict(request_key=current_request['request_key'],
            historical_horizon=case['historical_horizon'],source_lines=case,
            execution=execution,roles={'frozen':dict(seconds=seconds,steps=measurements(raw,current_request))})
        save_case_bundle(OUT/(case['name']+'-frozen'),bundle)
        log('case_frozen_completed',case=case['name'],usage=predictor.usage(),account=account)

    for case in PLAN['cases']:
        bundle = cases[case['name']]
        current_request = request_from_case(bundle,role='adapted')
        raw, measurement, seconds = predictor.predict(current_request)
        index = len(bundle['metadata']['predictions'])
        name = f'prediction_{index:03d}'
        bundle['arrays'][name] = raw.copy()
        entry = deepcopy(bundle['metadata']['predictions'][0])
        entry.update(role='adapted',array=name,comparison_only=True,measurement=measurement,runner_offline_comparison=True)
        for field in ('prediction_trace','prediction_usage','checkpoint_identity'): entry[field] = deepcopy(current_request[field])
        bundle['metadata']['predictions'].append(entry)
        bundle['metadata']['latest_accepted_role'] = 'adapted'
        summary[case['name']]['roles']['adapted'] = dict(seconds=seconds,steps=measurements(raw,current_request))
        first, second = bundle['metadata']['predictions'][0]['prediction_trace'],entry['prediction_trace']
        summary[case['name']]['pairing'] = dict(h1_input_equal=first[0]['input_sha256']==second[0]['input_sha256'],
            all_action_windows_equal=all(a['action_sha256']==b['action_sha256'] for a,b in zip(first,second)),
            all_rng_entries_equal=all(a['rng_entry_sha256']==b['rng_entry_sha256'] for a,b in zip(first,second)))
        if not all(summary[case['name']]['pairing'].values()): raise ValueError('Matched role contract differs')
        save_case_bundle(OUT/(case['name']+'-paired'),bundle)
        log('case_paired_completed',case=case['name'],usage=predictor.usage(),account=account)

    import torch
    chosen = cases['L-double-pusher']
    current_request = request_from_case(chosen,role='adapted')
    context = np.stack([transform_t1(frame)[0] for frame in current_request['images']])
    actions = np.asarray(list(current_request['past_actions'])+[current_request['target']],np.float32)/np.float32(511)*np.float32(2)-np.float32(1)
    if predictor.forecast_reservations+account['direct_forecast_reservations'] >= 17: raise RuntimeError('Combined forecast cap')
    account['direct_forecast_reservations'] += 1
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
        predictor.sampler.seed(5478)
        entry_rng = hashlib.sha256(predictor.sampler.capture()).hexdigest()
        direct, _ = predictor.sampler.sampler.sample(torch.tensor(context,device='cuda')[None],torch.tensor(actions,device='cuda')[None])
        torch.cuda.synchronize()
        account['direct_forecast_returns'] += 1
        check()
        direct = direct[0].cpu().numpy()
    reference = chosen['arrays']['prediction_001'][0]
    parity = dict(byte_identical=np.array_equal(direct,reference),maximum_absolute_difference=float(np.abs(direct-reference).max()),
        entry_rng_equal=entry_rng==chosen['metadata']['predictions'][1]['prediction_trace'][0]['rng_entry_sha256'],
        extra_forecasts=1,extra_euler_reservations=3)
    write_json('native-parity.json',parity)
    if not parity['byte_identical'] or not parity['entry_rng_equal']: raise ValueError('Direct upstream sampler differs')

    for name,bundle in cases.items():
        sheet = Image.new('RGB',(3*288,4*312),'#eeeeee')
        draw = ImageDraw.Draw(sheet)
        for row in range(4):
            for col,(label,array) in enumerate((('actual',bundle['arrays']['actual_t1']),
                    ('frozen',bundle['arrays']['prediction_000']),('adapted',bundle['arrays']['prediction_001']))):
                pixels = np.rint(np.clip(array[row],0,1).transpose(1,2,0)*255).astype('uint8')
                sheet.paste(Image.fromarray(pixels).resize((288,288),Image.Resampling.NEAREST),(col*288,row*312))
                draw.text((col*288+5,row*312+291),f'{name} {label} H{row+1}',fill='black')
        sheet.save(OUT/(name+'-horizons.png'))
    write_json('summary.json',summary)
    log('completed',usage=predictor.usage(),account=account,training_updates=0)
except BaseException:
    error = traceback.format_exc()
    (OUT/'first-error.txt').write_text(error,encoding='utf-8')
    if current_request is not None and 'np' in globals():
        np.savez_compressed(OUT/'failed-request-arrays.npz',images=current_request['images'],past_actions=current_request['past_actions'],
            target=current_request['target'],partial_prediction=np.stack(predictor.last_partial_predictions) if predictor and predictor.last_partial_predictions else np.empty((0,3,96,96),np.float32))
        write_json('failed-request.json',{key:value for key,value in current_request.items() if key not in ('images','past_actions','target','goal')})
    log('failed',error=error,account=account,usage=predictor.usage() if predictor else None)
    raise
finally:
    if session is not None: session.close()
    if predictor is not None: predictor.close()
    output_bytes = sum(path.stat().st_size for path in BASE.rglob('*') if path.is_file())
    write_json('final.json',dict(seconds=time.monotonic()-START,account=account,peaks=peaks,
        usage=predictor.usage() if predictor else None,cuda_allocated_peak=predictor.peak_allocated if predictor else 0,
        cuda_reserved_peak=predictor.peak_reserved if predictor else 0,output_bytes=output_bytes,
        training_updates=0,failed=(OUT/'first-error.txt').exists()))
    LOG.close()
    if output_bytes > 128*1024**2: raise OSError('Output cap exceeded')
