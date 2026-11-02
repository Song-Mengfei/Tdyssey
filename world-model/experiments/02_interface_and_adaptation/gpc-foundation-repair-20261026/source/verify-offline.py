"""Replay saved trace/array identities and summarize fixed gates; no Torch/physics."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
sys.path.insert(0,str(ROOT))
from tdyssey.manual_world_model import load_case_bundle

summary = json.loads((BASE/'real-01/summary.json').read_bytes())
output = dict(cases={},model_calls=0,physics_controls=0,training_updates=0)
for name,value in summary.items():
    bundle = load_case_bundle(BASE/'real-01'/(name+'-paired'))
    meta,arrays = bundle['metadata'],bundle['arrays']
    initial = np.asarray(meta['request']['initial_state']['pusher_body']['pose']['position'])
    truth = np.asarray([step['after_state']['pusher_body']['pose']['position'] for step in meta['execution']['steps']])
    object_initial = np.asarray(meta['request']['initial_state']['object_body']['pose']['position'])
    object_truth = np.asarray([step['after_state']['object_body']['pose']['position'] for step in meta['execution']['steps']])
    case = dict(request_key=meta['request']['request_key'],pairing=value['pairing'],
        initial_pusher=initial.tolist(),actual_pusher_h1_h4=truth.tolist(),
        physical_contacts_h1_h4=[step['contacts'] for step in meta['execution']['steps']],
        maximum_true_object_translation=float(np.linalg.norm(object_truth-object_initial,axis=1).max()),roles={})
    for prediction in meta['predictions']:
        raw = arrays[prediction['array']]
        trace = prediction['prediction_trace']
        assert len(raw)==len(trace)==4
        context = arrays['history_t1'].copy()
        past = list(arrays['past_actions'])
        target = np.asarray(meta['request']['target'],np.float32)
        for index,row in enumerate(trace):
            window = np.asarray(past+[target],np.float32)/np.float32(511)*np.float32(2)-np.float32(1)
            assert row['horizon']==index+1 and row['status']=='validated'
            assert hashlib.sha256(context.tobytes()).hexdigest()==row['input_sha256']
            assert hashlib.sha256(window.tobytes()).hexdigest()==row['action_sha256']
            assert hashlib.sha256(raw[index].tobytes()).hexdigest()==row['raw_sha256']
            context = np.concatenate((context[1:],raw[index][None]),axis=0)
            past = past[1:]+[target]
        measurements=value['roles'][prediction['role']]['steps']
        rows=[]
        for index,record in enumerate(measurements):
            pusher=record['measurement']['pusher']
            position=np.asarray(pusher['position']) if pusher['valid'] else None
            rows.append(dict(horizon=index+1,pusher=pusher,
                valid_pusher_error=(float(np.linalg.norm(position-truth[index])) if position is not None else None),
                shape=record['measurement']['shape'],
                meaningful_blue_components=record['pusher_diagnostics'].get('meaningful_components')))
        final=measurements[-1]['measurement']['pusher']
        distances=(np.linalg.norm(truth-np.asarray(final['position']),axis=1) if final['valid'] else None)
        case['roles'][prediction['role']]=dict(all_four_calls_and_raw_feedback_verified=True,steps=rows,
            h4_distance_to_actual_h1_h4=distances.tolist() if distances is not None else None,
            h4_nearest_actual_horizon=int(distances.argmin()+1) if distances is not None else None)
    components=value['roles']['adapted']['steps'][0]['pusher_diagnostics'].get('meaningful_components',[])
    case['adapted_h1_blue_distances']=[dict(position=item['position'],
        distance_to_input_pusher=float(np.linalg.norm(np.asarray(item['position'])-initial)),
        distance_to_true_h1_pusher=float(np.linalg.norm(np.asarray(item['position'])-truth[0]))) for item in components]
    output['cases'][name]=case
assert 'torch' not in sys.modules
with (BASE/'diagnostic-summary.json').open('x',encoding='utf-8') as stream:
    json.dump(output,stream,indent=2,allow_nan=False)
    stream.write('\n')
print('Verified two bundles, sixteen native trace/array/action records; no model or physics calls.')
