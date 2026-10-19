"""Bounded cold-restore simulator check of visual-model candidate rankings."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def candidate_actions(position, object_position, recorded):
    import numpy as np
    position=np.asarray(position,dtype=np.float64)
    direction=np.asarray(object_position,dtype=np.float64)-position
    direction/=max(float(np.linalg.norm(direction)),1e-9)
    offsets=np.arange(1,len(recorded)+1)[:,None]*12*direction
    return {'recorded':np.asarray(recorded,dtype=np.float64),
            'hold':np.repeat(position[None],len(recorded),axis=0),
            'toward':np.clip(position+offsets,25,486),'away':np.clip(position-offsets,25,486)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=ROOT/'third_party/diffusion_policy')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--anchors',type=int,default=4)
    parser.add_argument('--seconds',type=int,default=120)
    args=parser.parse_args()
    if not 1<=args.anchors<=4 or not 1<=args.seconds<=300:parser.error('Use 1–4 anchors and at most 300 seconds')
    if args.output.exists():parser.error('Output exists; choose a new file')
    for key,value in {'SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy','PYGAME_HIDE_SUPPORT_PROMPT':'1'}.items():
        os.environ.setdefault(key,value)
    import numpy as np
    import torch
    from tdyssey.dataset import digest,json_bytes
    from tdyssey.envs.official import make_official_shape_env_class
    from tdyssey.official import load_official_source
    from tdyssey.training_data import WindowSpec
    from tdyssey.world_model_data import VisualDynamicsDataset
    from tdyssey.world_model_training import cuda_setup,load_checkpoint,to_device

    started=time.monotonic();device=cuda_setup(0)
    model,payload=load_checkpoint(args.checkpoint,device)
    if digest(args.catalog.read_bytes())!=payload['catalog_sha256']:parser.error('Frozen catalog differs from checkpoint')
    future=payload['config']['future']
    dataset=VisualDynamicsDataset(args.catalog,'validation',WindowSpec(2,future))
    report={'schema_version':1,'kind':'validation_candidate_diagnostic_not_policy_improvement','status':'running',
            'checkpoint_sha256':digest(args.checkpoint.read_bytes()),'catalog_sha256':payload['catalog_sha256'],
            'selection':'first disjoint validation windows with no previous contact and some upcoming contact',
            'block_score':'maximum clipped upstream reward in the future-action block; stop on original success',
            'candidate_prior':'toward/away use privileged object position; no state is input to the visual predictor',
            'restore':'all candidates use the same full saved state in a fresh solver space; cached contact impulses absent',
            'model_test_used':False,'counts':{'control_steps':0,'control_physics_substeps':0,'reset_physics_substeps':0,
                                         'state_restores':0,'model_forecast_steps':0},'anchors':[]}
    env=None
    try:
        cls=make_official_shape_env_class(load_official_source(args.source_root))
        env=cls(shape='L',legacy=True);env.seed(0);env.reset();report['counts']['reset_physics_substeps']+=1
        last={}
        for index in range(len(dataset)):
            if len(report['anchors'])>=args.anchors:break
            if time.monotonic()-started>=args.seconds:raise TimeoutError('Candidate probe time cap')
            sample=dataset[index]
            if (sample['labels']['previous_contact'] or not (sample['labels']['n_contacts']>0).any()
                    or sample['anchor']<last.get(sample['episode_id'],-future)+future):continue
            last[sample['episode_id']]=sample['anchor']
            trajectory=dataset._cache[0];state=trajectory.transitions[sample['anchor']].before
            candidates=candidate_actions(sample['observations']['agent_pos'][-1],state.object_body.pose.position,sample['actions'])
            row={'episode_id':sample['episode_id'],'anchor':sample['anchor'],'initial_state':asdict(state),'candidates':[]}
            report['anchors'].append(row)
            for name,actions in candidates.items():
                if time.monotonic()-started>=args.seconds:raise TimeoutError('Candidate probe time cap')
                observation=env.restore_state(state);report['counts']['state_restores']+=1
                image_error=float(abs(observation['image']-sample['observations']['image'][-1]).max())
                if image_error>.5/255+1e-6:raise ValueError('Restored initial image differs from the saved current frame')
                batch=torch.utils.data.default_collate([sample]);gpu=to_device(batch,device)
                with torch.no_grad():prediction=model(gpu['observations'],torch.as_tensor(actions,dtype=torch.float32,device=device)[None])
                report['counts']['model_forecast_steps']+=future
                rewards=prediction['reward'][0].cpu().tolist()
                actual=[]
                for action in actions:
                    _,reward,done,info=env.step(action)
                    report['counts']['control_steps']+=1
                    report['counts']['control_physics_substeps']+=env.sim_hz//env.control_hz
                    actual.append({'reward':float(reward),'success':bool(done),'n_contacts':int(info['n_contacts']),
                                   'state':asdict(env.capture_state())})
                    if done:break
                row['candidates'].append({'name':name,'actions':actions.tolist(),'predicted_rewards':rewards,
                    'predicted_block_score':max(rewards),'actual_block_score':max(x['reward'] for x in actual),
                    'initial_image_max_error':image_error,'actual':actual,
                    'predicted_object_displacement':prediction['motion'][0,:,2:].cumsum(0).cpu().tolist()})
            predicted=max(row['candidates'],key=lambda r:r['predicted_block_score'])
            actual=max(row['candidates'],key=lambda r:r['actual_block_score'])
            row.update(selected=predicted['name'],best_physical=actual['name'],
                       regret=actual['actual_block_score']-predicted['actual_block_score'],
                       selected_optimism=predicted['predicted_block_score']-predicted['actual_block_score'],
                       predicted_score_range=max(c['predicted_block_score'] for c in row['candidates'])-min(c['predicted_block_score'] for c in row['candidates']),
                       physical_score_range=max(c['actual_block_score'] for c in row['candidates'])-min(c['actual_block_score'] for c in row['candidates']))
        if not report['anchors']:raise ValueError('No eligible pre-contact validation windows')
        report['status']='complete';report['elapsed_seconds']=time.monotonic()-started
    except BaseException as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if env is not None:env.close()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('xb') as stream:stream.write(json_bytes(report))
    print(json.dumps({'status':'complete','anchors':len(report['anchors']),'counts':report['counts'],
                      'mean_regret':sum(r['regret'] for r in report['anchors'])/len(report['anchors'])}))


if __name__=='__main__':main()
