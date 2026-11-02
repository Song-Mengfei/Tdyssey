"""Build/check an actual-data R3B proposal on CPU; no GPU trainer is provided.

Reads compact metadata, numeric window references and file hashes only. It never
loads model tensors, renders RGB, steps physics, or imports Torch.
"""
from __future__ import annotations
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
LETTERS=tuple('TLOCXB')
PROTOCOL='58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6'
CHECKPOINT_SHA='c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148'
MIB=1024**2


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def bound(path):return dict(path=str(Path(path).resolve()),sha256=sha(path),bytes=Path(path).stat().st_size)
def safe(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    if Path(name).is_absolute()or not path.is_relative_to(root):raise ValueError('Data reference escapes source root')
    return path


def budget():
    # Full108 + raw20.25 + code/log/fig8 =136.25MiB; no endpoint image copies.
    raw=16*4*3*96*96*4*3
    return dict(stage_cap_bytes=144*MIB,one_full_checkpoint_cap_bytes=108*MIB,
        raw_float32_frames=192,raw_float32_frame_bytes=110592,raw_float32_logical_bytes=raw,
        code_logs_metadata_figures_cap_bytes=8*MIB,
        planned_persistent_upper_bound_bytes=108*MIB+raw+8*MIB,
        growth_headroom_bytes=144*MIB-(108*MIB+raw+8*MIB),
        shared_cap_bytes=3*1024**3,atomic_reserve_bytes=256*MIB,
        duplicate_float64_endpoint_images=False,endpoint_persistence='immutable raw step references; lossless float64 view only in RAM',
        checkpoint_policy='One rolling complete last.pt; atomic replace every100 successful updates and final boundary; no model-only duplicates',
        raw_budget_includes_container_overhead=False,
        raw_container_metadata_uses_growth_headroom=True,
        bound_is_independent_of_compression_ratio=True)


def require_gate(manifest_path,gate_path):
    manifest_path=Path(manifest_path).resolve();gate_path=Path(gate_path).resolve()
    if not manifest_path.exists()or not gate_path.exists():raise RuntimeError('Wait for completed912 manifest and actual data-interface/interface-gate.json; no proposal config was generated')
    manifest=read(manifest_path);gate=read(gate_path)
    if (manifest.get('kind')!='R3A-compact-state-reference.v1' or manifest.get('collection_closed')is not True
            or manifest.get('protocol_sha256')!=PROTOCOL or manifest.get('control_dt')!=.1):raise ValueError('Closed selected-physics R3A compact library required')
    entries=manifest['entries'];counts=Counter((e['letter'],e['split'])for e in entries)
    if (len(entries)!=912 or counts!={(l,s):n for l in LETTERS for s,n in(('train',128),('seen_val',24))}
            or len({e['trajectory_id']for e in entries})!=912 or len({e['family_id']for e in entries})!=912
            or len({e['seed']for e in entries})!=912 or any(e['controls']!=48 or e['protocol_sha256']!=PROTOCOL for e in entries)):
        raise ValueError('Actual912 independent successful48-control scenes and128/24perletter required')
    if (gate.get('passed')is not True or gate.get('manifest_sha256')!=sha(manifest_path)
            or gate.get('trajectories')!=912 or gate.get('TRAIN_trajectories')!=768 or gate.get('seen_val_trajectories')!=144
            or gate.get('model_calls')!=0 or gate.get('checkpoint_loads')!=0 or gate.get('physics_controls')!=0
            or gate.get('Torch_imported')is not False or gate.get('model_inputs')!=['image','action']
            or gate.get('state_letter_geometry_masks_are_not_model_inputs')is not True
            or gate.get('sampler_rng_and_exposure_resume_exact')is not True):
        raise ValueError('Actual new-library CPU interface proof does not bind this manifest')
    if not all(k in gate for k in('two_worker_throughput_file','two_worker_throughput_sha256')):
        raise RuntimeError('Interface catalogue exists but actual two-worker throughput is not yet closed; wait before generating the proposal')
    throughput=safe(gate_path.parent,gate['two_worker_throughput_file'])
    if sha(throughput)!=gate['two_worker_throughput_sha256']:raise ValueError('Completed two-worker throughput evidence missing/changed')
    speed=read(throughput)
    if (speed.get('source_manifest_sha256')!=sha(manifest_path)or speed.get('workers_observed')!=2
            or speed.get('legacy_R2B_benchmark_only')is not False or speed.get('new_R3A_library_admission_claimed')is not True
            or speed.get('model_input_keys')!=['image','action']or speed.get('maximum_in_flight_observed',3)>2
            or speed.get('model_calls')!=0 or speed.get('checkpoint_loads')!=0 or speed.get('physics_controls')!=0
            or speed.get('Torch_imported')is not False or speed.get('samples_per_second_including_start',0)<=0):
        raise ValueError('Actual completed two-worker new-library throughput contract differs')
    source=(manifest_path.parent/manifest.get('source_root','.')).resolve()
    for entry in entries:
        for field,digest_field in(('file','sha256'),('metadata_file','metadata_sha256')):
            if sha(safe(source,entry[field]))!=entry[digest_field]:raise ValueError('Actual compact data/metadata SHA changed: '+entry[field])
    return manifest,gate,throughput


def reference_rows(manifest,gate,gate_path):
    """Reconstruct exact catalogue metadata from its existing numeric archive."""
    import numpy as np
    from tdyssey.native_rgb_r3a_data import select_training_pool
    compact=gate['compact_window_reference'];path=safe(Path(gate_path).resolve().parent,compact['file'])
    if sha(path)!=compact['sha256']:raise ValueError('Frozen compact window archive changed')
    with np.load(path,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
    entries=manifest['entries'];result={};codes={int(k):v for k,v in compact['category_codes'].items()}
    for horizon,key in((1,'h1'),(4,'h4')):
        rows=[];n=len(arrays[key+'_anchor'])
        for i in range(n):
            ti=int(arrays[key+'_trajectory_index'][i]);anchor=int(arrays[key+'_anchor'][i])
            if not 0<=ti<len(entries)or not 3<=anchor<=48-horizon:raise ValueError('Actual window indices escape complete48-control trajectory')
            entry=entries[ti]
            translation=float(arrays[key+'_translation_image96_px'][i]);rotation=float(arrays[key+'_signed_rotation_deg'][i])
            trans=translation>=2;rot=abs(rotation)>=5;category=codes[int(arrays[key+'_category'][i])]
            if category!=('both'if trans and rot else'translation_only'if trans else'rotation_only'if rot else'other'):raise ValueError('Frozen motion threshold/category mismatch')
            rows.append(dict(dataset_index=i,window_id=f"{entry['trajectory_id']}:{anchor}:H{horizon}",
                trajectory_index=ti,trajectory_id=entry['trajectory_id'],family_id=entry['family_id'],letter=entry['letter'],
                split=entry['split'],anchor=anchor,horizon=horizon,translation_image96_px=translation,
                signed_rotation_deg=rotation,abs_rotation_deg=abs(rotation),translation=trans,rotation=rot,moving=trans or rot,
                category=category,valid_window=True,control_dt=.1))
        if canonical(rows)!=compact['catalogue_identity'][key+'_sha256']:raise ValueError('Numeric catalogue does not reconstruct original exact window metadata')
        result[key]=rows
    selected=arrays['selected_h1_training_indices'].astype(np.int32)
    if len(set(map(int,selected)))!=len(selected)or not len(selected)or selected.min()<0 or selected.max()>=len(result['h1']):raise ValueError('Selected pool contains duplicate/out-of-range windows')
    selected_rows=[result['h1'][int(i)]for i in selected]
    if any(r['split']!='train'for r in selected_rows):raise ValueError('Selected training pool includes validation')
    if len(selected)!=gate['epoch_samples'] or len(selected)!=gate['selected_pool']['selected_windows']:raise ValueError('Actual selected/epoch counts disagree')
    catalogue=dict(h1=result['h1'],identity=compact['catalogue_identity'])
    recomputed=select_training_pool(catalogue)
    if (recomputed['selected_h1_training_indices']!=selected.tolist()
            or {k:v for k,v in recomputed.items()if k!='selected_h1_training_indices'}!=gate['selected_pool']
            or compact['selected_pool']!=gate['selected_pool']):
        raise ValueError('Actual subset must retain every moving window and balanced distinct negatives under the unchanged public policy')
    train=sum(r['split']=='train'for r in result['h1']);val=len(result['h1'])-train
    if (len(result['h1'])!=gate['fulllegal_H1_windows']or len(result['h4'])!=gate['fulllegal_H4_windows']
            or train!=gate['fulllegal_TRAIN_H1']or val!=gate['fulllegal_seen_val_H1']
            or compact['h1_windows']!=len(result['h1'])or compact['h4_windows']!=len(result['h4'])
            or compact['catalogue_identity']['manifest_sha256']!=gate['manifest_sha256']):
        raise ValueError('Actual legal counts or catalogue manifest identity differ')
    return result,selected_rows,selected,path


def select_evaluation(rows,manifest):
    from tdyssey.native_rgb_r2c0_evaluation import paired_seed
    h4={(r['trajectory_id'],r['anchor']):r for r in rows['h4']}
    eligible=[r for r in rows['h1']if r['split']=='seen_val'and(r['trajectory_id'],r['anchor'])in h4]
    selected=[];used_families=set();deficits=[]
    def choose(candidates,domain):
        candidates=[r for r in candidates if r['family_id']not in used_families]
        if not candidates:return None
        row=min(candidates,key=lambda r:canonical([domain,5478,r['window_id']]))
        used_families.add(row['family_id']);return row
    def record(row,group):
        final=h4[row['trajectory_id'],row['anchor']];entry=manifest['entries'][row['trajectory_index']]
        return dict(group=group,letter=row['letter'],window_id=f"{row['trajectory_id']}:{row['anchor']}",
            trajectory_id=row['trajectory_id'],family_id=row['family_id'],trajectory_index=row['trajectory_index'],split='seen_val',
            anchor=row['anchor'],h1_index=row['dataset_index'],h4_index=final['dataset_index'],
            source_file=entry['file'],source_sha256=entry['sha256'],H1_translation_image96_px=row['translation_image96_px'],
            H1_signed_rotation_deg=row['signed_rotation_deg'],H4_translation_image96_px=final['translation_image96_px'],
            H4_signed_rotation_deg=final['signed_rotation_deg'],history_state_indices=list(range(row['anchor']-3,row['anchor']+1)),
            executed_action_audit_indices=list(range(row['anchor']-3,row['anchor']+4)),
            prediction_input='4RGB histories and4 normalized actions only; truth/state/letter remain renderer/audit data')
    for group,domain,moving in(('natural','R3B-seen-natural-v1',False),('motion','R3B-seen-motion-v1',True)):
        for letter in LETTERS:
            row=choose([r for r in eligible if r['letter']==letter and(not moving or r['moving'])],domain)
            if row is None:deficits.append(dict(group=group,letter=letter,missing=1,reason='No unused independent-family legalH4 window meets the fixed group rule'))
            else:selected.append(record(row,group))
    for _ in range(4):
        row=choose([r for r in eligible if r['translation_image96_px']<.1 and r['abs_rotation_deg']<.5],'R3B-seen-nearstatic-v1')
        if row is None:break
        selected.append(record(row,'nearstatic'))
    missing=4-sum(r['group']=='nearstatic'for r in selected)
    if missing:deficits.append(dict(group='nearstatic',missing=missing,reason='Strict H1 translation<.1image96px and abs unwrapped rotation<.5deg with a new family; no threshold relaxation or copied window'))
    calls=[]
    for batch_index,start in enumerate(range(0,len(selected),4)):
        ids=[r['window_id']for r in selected[start:start+4]]
        for step in range(4):calls.append(dict(batch_index=batch_index,step=step,window_ids=ids,
            seed=paired_seed(5478,'R3B_new_seen16_v1',batch_index,ids,step)))
    return dict(kind='R3B-frozen-new-seen16-evaluation-v1',seed=5478,planned_windows=16,actual_windows=len(selected),
        rows=selected,deficits=deficits,group_counts=dict(Counter(r['group']for r in selected)),
        paired_raw_feedback_calls=calls,call_seeds_reused_across_points_and_denoising_settings=True,
        planned_groups=dict(natural=6,motion=6,nearstatic=4),all_families_distinct=len(used_families)==len(selected),
        natural_rule='one independently hash-selected legalH4 new seen-validation window per letter; no motion filter',
        motion_rule='one per letter with H1translation>=2image96px OR abs rotation>=5deg; different families from natural',
        static_rule='four strictnearstatic windows from further distinct families; shortfall remains explicit',
        batch_pairing='Concatenate groups in saved order, batch4 across all16; same R2C0 paired_seed domain for every checkpoint/setting; group metrics remain separate',
        cohort_seed_domain='R3B_new_seen16_v1',unseen_shape_claim=False)


def epoch_plan(selected_rows,epochs=8):
    import numpy as np
    from tdyssey.native_rgb_r3a_data import StratifiedEpochSampler
    sampler=StratifiedEpochSampler(selected_rows,seed=5478,batch_size=16)
    flat=[];offsets=[0];batch_epochs=[];summary=[];n=len(selected_rows)
    for epoch in range(1,epochs+1):
        batches=[];first_batch=len(batch_epochs);start_draw=len(flat)
        if epoch>1:batches.append(sampler.take_batch())
        remaining=len(sampler.batches)-len(batches)
        batches.extend(sampler.take_batch()for _ in range(remaining))
        values=[i for batch in batches for i in batch]
        if sorted(values)!=list(range(n)):raise ValueError('Exact epoch plan repeats/drops a selected window')
        for batch in batches:
            if not 1<=len(batch)<=16 or len({selected_rows[i]['family_id']for i in batch})!=len(batch):raise ValueError('Planned batch violates nominal16/family separation')
            flat.extend(batch);offsets.append(len(flat));batch_epochs.append(epoch)
        summary.append(dict(epoch=epoch,attempt_batches=len(batches),draws=len(values),unique_windows=n,repeated_within_epoch=0,
            batch_size_histogram=dict(Counter(str(len(b))for b in batches)),batch_start=first_batch,batch_end=len(batch_epochs),
            draw_start=start_draw,draw_end=len(flat),order_sha256=canonical([selected_rows[i]['window_id']for i in values])))
    return dict(flat_local_indices=np.asarray(flat,np.int32),batch_offsets=np.asarray(offsets,np.int32),
        batch_epochs=np.asarray(batch_epochs,np.uint8)),dict(epochs=epochs,selected_windows_per_epoch=n,
        total_planned_attempt_batches=len(batch_epochs),total_planned_draws=len(flat),epoch_summary=summary,
        each_window_expected_exposures_if_full_plan_delivered=epochs,attempts_are_not_guaranteed_successes=True,
        AMP_skip_consumes_batch=True,no_duplicate_or_dropped_tail=True,
        successful_update_counter='increase only on observed optimizer application; an AMP skip is a charged attempt/draw with zero success and zero scheduler advance',
        epoch_scope='eight complete selected-window exposures epochs, not an implicit attempt/time lifetime allowance')


def micro_weights(batch_samples):
    if type(batch_samples)is not int or not 1<=batch_samples<=16:raise ValueError('Actual batch1..16 required')
    return [min(4,batch_samples-i)/batch_samples for i in range(0,batch_samples,4)]


def source_paths():
    result=[ROOT/p for p in('scripts/native_rgb_r3b_proposal_check.py','tdyssey/native_rgb_r3a_data.py',
        'tdyssey/native_rgb_r3a_conditioning.py','tdyssey/raw_world_model_training.py','tdyssey/native_rgb_r2c0_evaluation.py',
        'tdyssey/native_rgb_r2c3_evaluation_io.py','tdyssey/native_rgb_r2c3_training.py','tdyssey/native_rgb_r2c3_continuation.py',
        'tdyssey/native_rgb_stage_budget.py','tdyssey/native_rgb_training.py','tdyssey/envs/official.py',
        'scripts/native_rgb_r3a_data_check.py','scripts/native_rgb_r3a_renderer_check.py','manifests/gpc_world_model.json',
        'artifacts/native-rgb-supervised-20261029/r3a/prepare-r3b-proposal.py')]
    result.append(ROOT.parent/'Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/denoiser.py')
    result.append(ROOT.parent/'Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/diffusion_sampler.py')
    if any(not p.is_file()for p in result):raise FileNotFoundError('An explicit current source dependency is missing')
    return result


def assemble(manifest_path,gate_path):
    from tdyssey.native_rgb_r3a_conditioning import POLICY,POLICY_SHA
    manifest,gate,throughput=require_gate(manifest_path,gate_path)
    rows,selected,selected_ids,archive=reference_rows(manifest,gate,gate_path)
    evaluation=select_evaluation(rows,manifest);arrays,training=epoch_plan(selected)
    arrays['selected_h1_indices']=selected_ids
    if (training['epoch_summary'][0]['order_sha256']!=gate['first_epoch_window_order_sha256']
            or training['epoch_summary'][0]['attempt_batches']!=gate['same_batch_family_separation_checked']
            or training['epoch_summary'][0]['batch_size_histogram']!=gate['batch_size_histogram']):
        raise ValueError('Exact first epoch does not reproduce the actual public data-interface gate')
    checkpoint=ROOT/'artifacts/native-rgb-supervised-20261029/r2c3/run/last.pt'
    if sha(checkpoint)!=CHECKPOINT_SHA or checkpoint.stat().st_size!=105331438:raise ValueError('Preserved complete6000 checkpoint identity differs')
    dependencies=[Path(manifest_path),Path(gate_path),archive,throughput,checkpoint,
        ROOT/'artifacts/native-rgb-supervised-20261029/r3a/conditioning-contracts.json',
        ROOT/'artifacts/native-rgb-supervised-20261029/user-model-time-policy.json',ROOT/'configs/native_rgb_r3a.json']
    contracts=read(dependencies[-3]);policy=read(dependencies[-2])
    if contracts.get('passed')is not True or contracts.get('policy_sha256')!=POLICY_SHA:raise ValueError('Passing CPU conditioning contracts must bind the actual current policy')
    if policy.get('time_policy')!='record_only_v2'or policy.get('hard_time_limits_enabled')is not False:raise ValueError('Existing record-only time policy must remain active')
    sources=source_paths()
    seed=lambda name:int.from_bytes(hashlib.sha256(json.dumps(['R3B-history-v1',5478,name],separators=(',',':')).encode()).digest()[:8],'little')%(2**63)
    proposal=dict(kind='R3B-actual-new-library-eight-epoch-proposal.v1',is_proposal=True,execution_authorized=False,
        training_authorized=False,GPU_training_implemented=False,integration_implemented=False,training_executed=False,
        CPU_proposal_builder_implemented=True,CPU_validation_entry='scripts/native_rgb_r3b_proposal_check.py',
        release_condition='Parent reviews actual R3A movement/videos and finalizes proposal before separate GPU integration/execution; server full training is human-run',
        starting_model=dict(complete_checkpoint=bound(checkpoint),old_model_lineage_updates=6000,
            full_state_load_executed_in_R3A=False,model_Adam_scaler_choice='A proposed: restore full6000 model/AdamW moments/scaler; subject to parent finalization',
            optimizer=dict(lr=1e-5,betas=[.9,.999],eps=1e-8,weight_decay=.01),
            LR_migration='preserve held eta_min1e-5, native scheduler last_epoch3000, true oldlineage6000; no cosine rebound or fresh unlogged reset',
            later_R3B_successes_are_new_data_stage_updates=True,
            expected_total_parameter_updates_if_every_planned_attempt_succeeds=6000+training['total_planned_attempt_batches']),
        data=dict(manifest=bound(manifest_path),interface_gate=bound(gate_path),window_archive=bound(archive),
            trajectories=912,TRAIN_trajectories=768,seen_validation_trajectories=144,
            legal_H1=gate['fulllegal_H1_windows'],legal_H4=gate['fulllegal_H4_windows'],
            legal_TRAIN_H1=gate['fulllegal_TRAIN_H1'],legal_seen_validation_H1=gate['fulllegal_seen_val_H1'],
            selected_pool=gate['selected_pool'],protocol_sha256=PROTOCOL,control_dt=.1,
            migration='Explicit new-manifest selected-pool cursor0/epoch0 and named RNG states; retain old861-order/oldnoise RNG as provenance, not same-distribution exact resume',
            source_sha256=sha(ROOT/'tdyssey/native_rgb_r3a_data.py'),labels_are_renderer_audit_only=True),
        training_plan=training,batching=dict(nominal_effective_batch=16,microbatch_max=4,
            AMP=True,gradient_clip_norm=1.0,restore_scaler_state=True,
            short_batch_policy='Use actual N and scale each micro mean by micro_N/N; no dropped samples, repeated padding or accidental /16 attenuation',
            sample_weights_by_actual_batch_size={str(n):micro_weights(n)for n in range(1,17)}),
        conditioning=dict(policy=POLICY,policy_sha256=POLICY_SHA,
            proposed_rng_seeds={name:seed(name)for name in('branch','target','weak')},
            RNG_device='branch CPU; independent target/weak generators on inputdevice; all states saved/restored with policy+seed+device identity',
            objective='unchanged native H1 FP32 residualMSE; once normalized image/action, original target sigma/offset distribution',
            normal_AMP_skip='real scaler skip lowers scale; no success/LR advance, consume current batch once and continue; separate attempt/draw/skip/success counters',
            proposed_numerical_stop=dict(nonfinite_loss_or_parameters='fatal',max_consecutive_AMP_overflows=16,success_resets_consecutive_overflows=True)),
        evaluation=evaluation,observation_schedule=dict(fixed_noise_loss_epochs=[1,2],
            fixed_loss='frozen16window inputs/branch choices and three noise stream seeds, identical at both points; training RNG/order captured/restored; clean/weak loss separately reported',
            native_evaluations=[dict(epoch=4,denoise_steps=3),dict(epoch=8,denoise_steps=3),dict(epoch=8,denoise_steps=16)],
            raw_feedback_steps=4,H1_reuses_first_H4_call=True,actual_window_count=evaluation['actual_windows'],
            worst_case_new_raw_frames=192,no_300_point_evaluation=True,
            sampler=dict(sigma_min=.002,sigma_max=5,rho=7,order=1,s_churn=0,s_cond=0),
            planned_sampler_batches_if_full16=48,planned_sampling_denoiser_forwards_if_full16=352,
            metrics='same frozen grayIoU/ROI-MAE/contour/centroid/drift and bluepusher separately; shape failure/unknown, holes/fragments and strictnearstatic full denominators'),
        resource_plan=budget(),time_policy=dict(kind='record_only_v2',user_set_model_time_budget=False,hard_time_limit=None),
        proposed_R3B_output_directory=str(ROOT/'artifacts/native-rgb-supervised-20261029/r3b'),
        dependencies=[bound(p)for p in dependencies],source_bindings=[bound(p)for p in sources],
        required_unimplemented_integration=['CompleteGPUtrainer with fullmodel/Adam/scaler restore verification and explicit data/noise/LR migrations',
            'Variable tail-batch accumulation weighted by actual samples and separate attempts/skips/successes/epoch exposure counters',
            'Save full model/optimizer/scaler/LR/epoch sampler plus all three conditioning RNG streams in one rolling last.pt',
            'Public16window metric adapter using unchanged endpoint math and rawreference loader; old fixed12/cohort scorer envelope cannot be falsely reused as already integrated',
            'Actual whole-source/input freeze and full-stage144MiB+256MiB atomic/shared3GiB GPU preflight after separate parent release'],
        limits=['Proposal CPU validity is not GPU execution or empirical prediction acceptance',
            'Sixseen letters only; natural/motion/static groups selected from newvalidation families, no unseen-shape claim',
            'Newdata plus clean/weakhistory jointly changes training; cannot assign improvement to one unique factor',
            'Eightcomplete epochs fixes exposures, not guaranteed successful optimizer updates if AMP skips occur'])
    proposal['storage_roots']=read(ROOT/'configs/native_rgb_r3a.json')['storage_roots']+[str(p)for p in sources]
    return proposal,arrays


def check_storage(proposal,config_path):
    from tdyssey.native_rgb_stage_budget import storage_footprint
    roots=proposal['storage_roots']+[str(Path(config_path).resolve()),str(Path(proposal['proposal_directory']))]
    actual=storage_footprint(roots)
    prior=read(ROOT/'configs/native_rgb_r3a.json');r3a=storage_footprint([prior['output']])
    if prior['R3A_cap_bytes']!=96*MIB or r3a['bytes']>96*MIB:raise ValueError('Current R3A data/report/proposal artifacts exceed the reserved96MiB stage')
    projected=actual['bytes']-r3a['bytes']+96*MIB+144*MIB+256*MIB
    if projected>3*1024**3:raise ValueError('Actual nonR3A shared footprint plus whole R3A96MiB/R3B144MiB and atomic256MiB reservation exceeds3GiB')
    return dict(actual_shared_bytes=actual['bytes'],future_whole_R3B_stage_reserved_bytes=144*MIB,
        R3A_actual_bytes=r3a['bytes'],R3A_whole_stage_reserved_bytes=96*MIB,
        atomic_reserve_bytes=256*MIB,conservative_projected_bytes=projected,shared_cap_bytes=3*1024**3,
        actual_all_paths_counted=True)


def build_proposal(manifest_path,gate_path,outdir,config_path):
    import numpy as np
    outdir=Path(outdir).resolve();config_path=Path(config_path).resolve()
    if config_path.exists():raise FileExistsError('Proposal already exists; preserve it and explicitly version any later change')
    proposal,arrays=assemble(manifest_path,gate_path)
    outdir.mkdir(parents=True,exist_ok=False)
    path=outdir/'eight-epoch-index-plan.npz';np.savez_compressed(path,**arrays)
    proposal['training_plan']['index_plan']=bound(path)
    evaluation_path=outdir/'evaluation-indices.json';write(evaluation_path,proposal['evaluation'])
    proposal['evaluation_indices']=bound(evaluation_path)
    snapshots=outdir/'source-snapshot';snapshots.mkdir()
    for row in proposal['source_bindings']:
        source=Path(row['path']);destination=snapshots/source.name;destination.write_bytes(source.read_bytes())
        row['snapshot']=str(destination);assert sha(destination)==row['sha256']
    proposal['config_file']=str(config_path);proposal['proposal_directory']=str(outdir)
    write(config_path,proposal)
    return validate(config_path)


def validate(config_path):
    import numpy as np
    config_path=Path(config_path).resolve();p=read(config_path)
    if p.get('is_proposal')is not True or any(p.get(k)is not False for k in('execution_authorized','training_authorized','GPU_training_implemented','integration_implemented','training_executed')):
        raise ValueError('CPU proposal must not claim GPU implementation/execution or authorization')
    for row in p['dependencies']+p['source_bindings']:
        if Path(row['path']).stat().st_size!=row['bytes']or sha(row['path'])!=row['sha256']:raise ValueError('Bound source/input changed: '+row['path'])
        if row.get('snapshot')and sha(row['snapshot'])!=row['sha256']:raise ValueError('Source snapshot changed')
    if p['resource_plan']!=budget():raise ValueError('Reserved144MiB/onefull/raw budget differs')
    if p['resource_plan']['planned_persistent_upper_bound_bytes']>=144*MIB:raise ValueError('Planned persistent budget is not below144MiB')
    if p['time_policy']!=dict(kind='record_only_v2',user_set_model_time_budget=False,hard_time_limit=None):raise ValueError('No time stopping budget is permitted')
    fresh,arrays=assemble(p['data']['manifest']['path'],p['data']['interface_gate']['path'])
    for key in('starting_model','data','batching','conditioning','evaluation','observation_schedule','proposed_R3B_output_directory','required_unimplemented_integration'):
        if p[key]!=fresh[key]:raise ValueError('Actual data/evaluation/noise/plan mismatch: '+key)
    planned=p['training_plan'];expected=fresh['training_plan'];reference=planned['index_plan']
    if sha(reference['path'])!=reference['sha256']:raise ValueError('Exact eight-epoch plan archive changed')
    with np.load(reference['path'],allow_pickle=False)as z:
        if set(z.files)!=set(arrays)or any(not np.array_equal(z[k],v)for k,v in arrays.items()):raise ValueError('Eight epochs do not match the deterministic actual selected-pool stream')
    if {k:v for k,v in planned.items()if k!='index_plan'}!=expected:raise ValueError('Planned exact epoch counts differ')
    if sha(p['evaluation_indices']['path'])!=p['evaluation_indices']['sha256']or read(p['evaluation_indices']['path'])!=p['evaluation']:raise ValueError('Frozen16-window index sidecar changed')
    if 'torch'in sys.modules:raise RuntimeError('Proposal CPU check unexpectedly imported Torch')
    storage=check_storage(p,config_path)
    return dict(kind='R3B-CPU-only-proposal-validation.v1',passed=True,config_file=str(config_path),config_sha256=sha(config_path),
        trajectories=912,selected_windows=p['training_plan']['selected_windows_per_epoch'],epochs=8,
        exact_attempt_batches=p['training_plan']['total_planned_attempt_batches'],exact_planned_draws=p['training_plan']['total_planned_draws'],
        actual_evaluation_windows=p['evaluation']['actual_windows'],evaluation_deficits=p['evaluation']['deficits'],
        planned_persistent_upper_bound_bytes=p['resource_plan']['planned_persistent_upper_bound_bytes'],
        storage=storage,GPU_integration_implemented=False,GPU_training_executed=False,checkpoint_loads=0,model_calls=0,physics_steps=0,Torch_imported=False)


def self_test():
    import numpy as np
    assert budget()['raw_float32_logical_bytes']==21233664
    assert budget()['planned_persistent_upper_bound_bytes']==136.25*MIB
    for n in range(1,17):assert abs(sum(micro_weights(n))-1)<1e-15
    rows=[]
    for l in LETTERS:
        for family in range(3):
            for anchor in range(3):rows.append(dict(letter=l,family_id=l+str(family),trajectory_id=l+str(family),
                split='train',category='both'if family==0 and anchor==0 else'other',window_id=f'{l}{family}:{anchor}'))
    arrays,plan=epoch_plan(rows)
    assert plan['total_planned_draws']==len(rows)*8 and set(arrays['batch_epochs'])==set(range(1,9))
    assert all(1<=int(n)<=16 for n in np.diff(arrays['batch_offsets']))
    h1=[];h4=[];entries=[]
    for letter in LETTERS:
        for family in range(4):
            index=len(entries);tid=f'{letter}-fixture-{family}'
            entries.append(dict(file=tid+'.npz',sha256='0'*64))
            row=dict(dataset_index=index,trajectory_index=index,trajectory_id=tid,family_id=tid,
                letter=letter,split='seen_val',anchor=3,translation_image96_px=3. if family<2 else 0.,
                signed_rotation_deg=0.,abs_rotation_deg=0.,moving=family<2)
            h1.append(dict(row,window_id=tid+':3:H1'));h4.append(dict(row,window_id=tid+':3:H4'))
    evaluation=select_evaluation(dict(h1=h1,h4=h4),dict(entries=entries))
    assert evaluation['group_counts']==dict(natural=6,motion=6,nearstatic=4)
    assert evaluation['actual_windows']==16 and evaluation['all_families_distinct']and not evaluation['deficits']
    assert len(evaluation['paired_raw_feedback_calls'])==16
    assert evaluation==select_evaluation(dict(h1=h1,h4=h4),dict(entries=entries))
    # Keep the original motion threshold but remove every strict-static candidate.
    nonstatic=[dict(r,translation_image96_px=max(.2,r['translation_image96_px']))for r in h1]
    short=select_evaluation(dict(h1=nonstatic,h4=h4),dict(entries=entries))
    assert short['actual_windows']==12 and short['deficits'][0]['group']=='nearstatic'and short['deficits'][0]['missing']==4
    assert 'torch'not in sys.modules
    return dict(passed=True,tests=['136.25MiB worst logical persistent arithmetic','all actual sample1..16 micro weights sum1',
        'eight exact withoutreplacement family-separated selected epochs','deterministic16 total natural6/motion6/static4 independent-family selection and paired seeds',
        'strict-static deficiency stays explicit with no repeated/fake windows','no Torch/model/checkpoint/physics'],fixture_only=True,
        source_file=str(Path(__file__).resolve()),source_sha256=sha(__file__),
        Torch_imported=False,checkpoint_loads=0,model_calls=0,physics_steps=0,final_actual_data_config_generated=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/native_rgb_r3b_proposal.json')
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--out',type=Path)
    args=parser.parse_args();result=self_test()if args.self_test else validate(args.config)
    if args.out:write(args.out,result)
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
