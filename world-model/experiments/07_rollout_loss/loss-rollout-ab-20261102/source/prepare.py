"""Saved-only one-time freeze. No Torch, model, simulation or new data."""
from common import *
import random
def main():
    if (ROOT/'config.json').exists():raise FileExistsError('No refreeze of executed plan')
    index=read(HMF/'selection/index.json');order=read(HMF/'selection/order.json');spec=read(HMF/'evaluation-spec.json')
    # Native H1 order is an exact prefix, not a rebalanced/new pool.
    native=order['batches'][:256]
    blocks=[dict(segment_index=i,phase=p,segment_id=row['segment_id'],letter=row['letter']) for i,row in enumerate(index['train48']) for p in (0,4,8,12)]
    rng=random.Random(2026110201);schedule=[]
    while len(schedule)<256:
        cycle=list(range(len(blocks)));rng.shuffle(cycle);schedule.extend(cycle)
    schedule=schedule[:256]
    batches=[b for b in spec['batches'] if b['batch_id'] in spec['panel_batch_ids'] or b['rows'][0]['selection_role']!='train']
    assert sum(len(b['rows']) for b in batches)==39
    write(ROOT/'schedule.json',dict(native_h1_batches=native,aux_blocks=blocks,aux_block_indices=schedule,aux_seed=2026110202))
    write(ROOT/'evaluation-spec.json',dict(batches=batches,source=str(HMF/'evaluation-spec.json'),source_sha256=sha(HMF/'evaluation-spec.json'),scope='unchanged fixed train12 + all seen_val24 + old failures3; not all train48'))
    paths=[HMF/p for p in ['training_core.py','data.py','evaluation.py','evaluation-spec.json','selection/index.json','selection/order.json','selection/summary.json','fit-standard.json','GT-measurement-calibration.json']]
    paths += [ROOT.parent/'oc-continuation-20261031/training/epoch2.pt',ROOT.parent/'region-weighted-cd-20261101/evaluation.py',ROOT.parent/'generated-history-ab-20261101/evaluation.py',ROOT.parent/'accelerated-rgb-run-20261031/train_eval.py',ROOT.parent/'native-rgb-20261028/frozen-renderer-calibration.json']
    source=REPO.parent/'Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion'
    paths+=list(source.glob('*.py'))
    paths += [Path(r['raw_path']) for r in index['train48']+index['seen_val24']+index['failures3']]
    paths += [REPO/p for p in ['tdyssey/gpc_world_model.py','tdyssey/native_rgb_r3a_conditioning.py','tdyssey/native_rgb_evaluation.py','tdyssey/native_rgb_r2c3_training.py','tdyssey/raw_world_model_training.py','scripts/native_rgb_stage3_score.py']]
    paths += list(ROOT.glob('*.py'))+[ROOT/'PLAN.txt',ROOT/'schedule.json',ROOT/'evaluation-spec.json']
    bindings=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(paths))]
    c=dict(run_id='LOSS-AR4-20261102',approved_authorization=dict(thread='__LOCAL_CHAT__',turn='__LOCAL_CHAT__',userMessage='__LOCAL_CHAT__'),
      parent=str(ROOT.parent/'oc-continuation-20261031/training/epoch2.pt'),parent_sha256='c3ba4099b42f054b4cc60a675e93265d8aee8649e9a7825b3ad70a269e1fb187',
      source_root=str(REPO.parent/'Tdyssey/third_party/gpc_code'),attempts_per_arm=256,milestones=[128,256],microbatch=4,effective_h1_batch=16,aux_sequences_per_B_attempt=1,
      formula_A='mean_16,3,96,96((F_theta(z,h,a,cs) - (y-c_skip*z)/c_out)^2)',formula_B='L_A + 4*mean_sequence1,horizon4,3,96,96((S_theta^h(history,actions,epsilon)-GT_h)^2)',
      unique_variable='auxiliary differentiable four-step native-sampler reconstruction loss, including disclosed STE backward; original H1 branch unchanged',
      matching='same optimizer attempts and native H1 targets/noise, NOT equal total targets/compute; B adds4 GT exposures and12 forward+12 recompute image calls per attempt',
      sampler=dict(num_steps_denoising=3,sigma_min=.002,sigma_max=5.,rho=7,order=1,s_churn=0.,s_cond=0.),
      caps=dict(attempts=512,successful_updates=512,native_target_exposures=8192,aux_target_exposures=1024,train_image_forwards=14336,train_batch_forwards=8192,training_backward_calls=2304,
                eval_predictions=6240,eval_image_forwards=18720,eval_batch_forwards=4800,preflight_image_forwards=64,preflight_gradient_calls=2,all_image_forwards=33120,all_batch_forwards=13056,new_root_bytes=CAP,shared_bytes=SHARED_CAP),
      preflight='one frozen B1 four-step sequence; native/STE exact forward parity + H4-to-H1 gradient + whole objective backward; at most64 image forwards,2 gradient/backward calls,0 optimizer updates; fail stops, no automatic retry',
      stop='256 attempts each; any nonfinite/AMP skip/guard/parity failure stops with partial accounting, no extra attempts. No arbitrary elapsed-time cutoff. No next experiment.',
      resources='CUDA only; native FP16 AMP/FP32 loss, aux and eval FP32; one low-priority process, all CPU numerical threads1; no video encoding',
      acceptance='Existing HMF thresholds unchanged on frozen39 subset; no claim all48 fit. Direction/topology/shape unknown stays unknown. For directional support require per-letter seen-val AR4 meanIoU B>A and position B<=A, RH-strong IoU B>=A and position B<=A, no worse topology failures at1/4/8/16; report each separately plus old failures and all16 curves. One seed/budget cannot establish general efficacy.',
      eta_minutes=[15,30],wall_time_is_measurement_only=True,bindings=bindings,created_utc=utc())
    write(ROOT/'config.json',c);write(ROOT/'freeze.json',dict(config_sha256=sha(ROOT/'config.json'),storage=guard(full=True),new_model_calls=0,new_physics=0,at_utc=utc()))
    print(json.dumps(dict(config_sha256=sha(ROOT/'config.json'),bindings=len(bindings),storage=guard(),budget=c['caps'])))
if __name__=='__main__':main()
