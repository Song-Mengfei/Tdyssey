"""One training pass, two model-only exports and three serial raw H1 reloads."""
import gc
import json
from pathlib import Path
import time


def run_worker(path):
    from . import gpc_adaptation_guard as guard
    from .gpc_raw_fulltrain_entry import verify_worker, ROOT
    from .gpc_raw_fulltrain_runtime import RawTrainingBudget
    from .gpc_evaluation_status import Writer
    from .gpc_adaptation import OutputWriter, batch
    handoff,prepared = verify_worker(path); cfg = prepared['config']; output = Path(handoff['output_dir'])
    entries = prepared['inputs']['full_schedule']['entries']; specs = prepared['inputs']['development']['entries']
    keys = [r['case_id'] for r in specs]
    budget = RawTrainingBudget.create(output/'ledger',handoff['binding'],entries=entries,case_keys=keys)
    # Exporter owns five128KiB JSON slots and two64MiB weight slots; prepay both
    # outside this writer so the combined writers cannot exceed the same caps.
    writer_limits = dict(cfg['limits'],metadata_bytes=cfg['limits']['metadata_bytes']-5*128*1024,
        checkpoint_bytes=64*1024**2,output_bytes=cfg['limits']['output_bytes']-69*1024**2)
    writer = OutputWriter(output,writer_limits); trace = guard.StartupTrace(path)
    runtime = dict(torch=None,model_instances=0)
    model = optimizer = selected = generator = inputs = loss = norm = sampler = None
    def refresh():
        value = dict(model_instances=runtime['model_instances'])
        if runtime['torch'] is not None: value.update(guard.cuda_allocator_sample(runtime['torch']))
        return value
    with trace, Writer(Path(handoff['status_path']),pid=handoff['actual_worker_pid'],
            nonce=handoff['nonce'],raw_fulltrain=True) as journal:
        with guard.Heartbeat(path,guard.status_template(raw_fulltrain=True),refresh=refresh,
                publisher=lambda _path,value:journal.publish(value)) as heartbeat:
            def check(phase=None):
                state = budget.check(); heartbeat.check()
                if not state['training_complete'] and (time.time() >= handoff['binding']['training_deadline_unix']
                        or time.monotonic() >= handoff['binding']['training_deadline_monotonic']):
                    raise TimeoutError('Original1200s training deadline exceeded')
                torch = runtime['torch']
                if torch is not None:
                    torch.cuda.synchronize(0)
                    if (torch.cuda.max_memory_allocated(0)>cfg['limits']['cuda_allocated_bytes']
                            or torch.cuda.max_memory_reserved(0)>cfg['limits']['cuda_reserved_bytes']):
                        raise MemoryError('Unchanged CUDA peak ceiling exceeded')
                fields = dict(model_loads=state['spent_reserved_model_loads'],optimizer_updates=state['completed_updates'],
                    forecasts=state['spent_reserved_forecasts'],euler_steps=state['spent_reserved_euler_steps'],
                    eval_batches=state['spent_reserved_eval_batches'],eval_examples=state['spent_eval_examples'],
                    training_complete=state['training_complete'])
                if phase is not None: fields['phase'] = phase
                heartbeat.update(**fields)
            try:
                torch,_ = guard.initialize_sdk(heartbeat,trace,cpu_threads=2); runtime['torch'] = torch
                import numpy as np
                from .gpc_raw_fulltrain_data import RawDataset,RawScheduleConsumer
                from .gpc_supervised_adaptation import (make_real_denoiser,make_optimizer,make_generator,
                    supervised_denoising_loss,make_noise_plan,validation_loss)
                from .official_world_model import model_profile,load_denoiser_sampler,predict_next_frame,RAW_CONTRACT
                from .gpc_raw_fulltrain_outputs import export_checkpoint,write_panels
                dataset = RawDataset(ROOT/cfg['data_manifest'],'train')
                if len(dataset) != 2328: raise ValueError('Train window count differs')
                schedule = RawScheduleConsumer(prepared['inputs']['full_schedule'],
                    prepared['plan_identities']['full_schedule']['sha256'],dataset)
                provenance = dict(data_identity=dict(path=cfg['data_manifest'],sha256=cfg['data_manifest_sha256'],
                    mode='object-directed',train_trajectories=24,train_shapes=['T','L']),
                    list_identity=prepared['plan_identities']['full_schedule'],
                    source_identity=dict(commit=handoff['binding']['source_commit'],config_sha256=prepared['config_sha256'],
                        upstream_commit='a96146e568826ef46134f73c22fdf59a675e7264'),
                    authorization_id=handoff['binding']['authorization_id'],objective=cfg['train'],seed=5478)
                operation = budget.reserve_model_load('train_init'); runtime['model_instances'] = 1; check('model_loading')
                model = make_real_denoiser(handoff['gpc_root'],handoff['checkpoint'],'cuda:0',dict(execute=True))
                optimizer = make_optimizer(model); selected = [p for p in model.parameters() if p.requires_grad]
                generator = make_generator(5478,'cuda:0'); budget.complete_model_load(operation); check('training')
                metrics = []; checkpoints = {}; training_started = time.monotonic()
                for cursor,entry in enumerate(entries):
                    check('training'); inputs = batch(dataset,schedule.indices(cursor,1),'cuda:0')
                    operation = budget.reserve_update(1,metadata=dict(cursor=cursor,entry=entry)); check('training')
                    optimizer.zero_grad(set_to_none=True)
                    loss,values = supervised_denoising_loss(model,inputs,generator=generator)
                    loss.backward()
                    grads = [p.grad for p in selected if p.grad is not None]
                    if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                        raise FloatingPointError('Missing or nonfinite intended parameter gradients')
                    del grads
                    norm = torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True)
                    optimizer.step()
                    if any(not bool(torch.isfinite(p).all()) for p in selected):
                        raise FloatingPointError('Nonfinite updated parameter')
                    check('training'); budget.complete_update(operation)
                    step = cursor+1
                    metrics.append(dict(step=step,shape=entry['shape'],trajectory_id=entry['trajectory_id'],
                        anchor=entry['anchor'],loss=values['loss_denoising'],gradient_norm=float(norm)))
                    if step in (500,2328):
                        check('export'); checkpoints[str(step)] = export_checkpoint(model,output,step,provenance)
                        budget.exported(step); check('training')
                training_seconds = time.monotonic()-training_started
                budget.training_finished(); check('export')
                writer.json('training-metrics.json',dict(rows=metrics,updates=2328,train_examples=2328,
                    shape_examples=dict(T=1164,L=1164),training_loop_seconds=training_seconds))
                optimizer.zero_grad(set_to_none=True)
                optimizer = selected = generator = inputs = loss = norm = model = dataset = schedule = None
                gc.collect(); torch.cuda.empty_cache(); runtime['model_instances'] = 0
                budget.release_model('train_init'); check('evaluation')
                report = _evaluate(prepared,handoff,budget,writer,checkpoints,check,runtime)
                writer.json('diagnostic.json',report)
                writer.json('completion.json',{key:report[key] for key in ('kind','status','forecast_records',
                    'optimizer_updates','train_examples','model_loads','eval_loss_batches','train_shape_counts',
                    'checkpoints','interpretation','physics','collection','policy_training')})
                check('complete')
            except BaseException as error:
                budget.fail(type(error).__name__+': '+str(error)); heartbeat.record_error(error,step='raw_fulltrain.worker')
                raise
            finally:
                model = optimizer = selected = generator = inputs = loss = norm = sampler = None
                gc.collect()
                if runtime['torch'] is not None: runtime['torch'].cuda.empty_cache()
                runtime['model_instances'] = 0
    print(json.dumps(dict(status='complete',forecast_records=48,optimizer_updates=2328)))
    return 0


def _evaluate(prepared,handoff,budget,writer,checkpoints,check,runtime):
    import numpy as np
    from .gpc_raw_fulltrain_data import RawDataset
    from .gpc_raw_fulltrain_entry import ROOT
    from .gpc_supervised_adaptation import make_generator,make_noise_plan,validation_loss
    from .official_world_model import model_profile,load_denoiser_sampler,predict_next_frame,RAW_CONTRACT
    from .gpc_raw_fulltrain_outputs import write_panels
    torch = runtime['torch']; cfg = prepared['config']; specs = prepared['inputs']['development']['entries']
    development = RawDataset(ROOT/cfg['data_manifest'],'development')
    cases = []; samples = []
    for spec in specs:
        sample = development[spec['dataset_index']]
        if sample['episode_id'] != spec['trajectory_id'] or sample['anchor'] != spec['anchor']:
            raise ValueError('Frozen development sample identity differs')
        samples.append(sample)
        cases.append(dict(case_id=spec['case_id'],latest=sample['image'][3],target=sample['image'][4]))
        writer.raw('raw/'+spec['case_id']+'-target.npy',sample['image'][4])
    development = None; gc.collect()
    results = {spec['case_id']:{} for spec in specs}; rows = []; loss_rows = []; paired_noise = {}
    profiles = dict(initial=model_profile('initial',handoff['checkpoint']))
    for role,step in (('raw500','500'),('raw2328','2328')): profiles[role] = checkpoints[step]['profile']
    model = sampler = None
    for role in ('initial','raw500','raw2328'):
        operation = budget.reserve_model_load(role); runtime['model_instances'] = 1; check('model_loading')
        profile = profiles[role]
        sampler,device = load_denoiser_sampler(handoff['gpc_root'],profile['checkpoint'],profile=profile,execute=True)
        model = sampler.denoiser; model.requires_grad_(False)
        budget.complete_model_load(operation); check('evaluation')
        for ordinal,(spec,sample) in enumerate(zip(specs,samples)):
            operation = budget.reserve_eval(1,metadata=dict(role=role,example=ordinal,case_id=spec['case_id']))
            check('evaluation')
            inputs = {key:torch.from_numpy(sample[key][None]).to(device) for key in ('image','action')}
            noise_generator = make_generator(5478+ordinal,device); noise = make_noise_plan(inputs,noise_generator)
            noise_hash = noise.generator_after_sha256
            if role == 'initial': paired_noise[spec['case_id']] = noise_hash
            elif paired_noise[spec['case_id']] != noise_hash: raise ValueError('Paired fixed-noise identity differs')
            value,details = validation_loss(model,inputs,noise_plan=noise)
            check('evaluation'); budget.complete_eval(operation)
            loss_rows.append(dict(role=role,case_id=spec['case_id'],shape=spec['shape'],loss=value,
                noise_sha256=noise_hash,model_output_calls=details['model_output_calls']))
            del noise,noise_generator,inputs
        for ordinal,(spec,sample) in enumerate(zip(specs,samples)):
            operation = budget.reserve_forecast(1,3,metadata=dict(role=role,example=ordinal,case_id=spec['case_id']))
            check('evaluation')
            prediction = predict_next_frame(sampler,sample['image'][:4],sample['absolute_action'],device,seed=5478)
            if prediction.shape != (3,96,96) or not np.isfinite(prediction).all():
                raise FloatingPointError('Invalid raw H1 forecast')
            mae = float(np.abs(prediction-sample['image'][4]).mean())
            writer.raw('raw/'+spec['case_id']+'-'+role+'.npy',prediction)
            results[spec['case_id']][role] = prediction
            rows.append(dict(role=role,case_id=spec['case_id'],shape=spec['shape'],contact=spec['contact'],
                raw_mae=mae,seed=5478,euler_steps=3,sigma_cond=None,
                measurement=dict(status='not_measured',pose=None,coverage=None)))
            check('evaluation'); budget.complete_forecast(operation)
        model = sampler = None; gc.collect(); torch.cuda.empty_cache(); runtime['model_instances'] = 0
        budget.release_model(role); check('evaluation')
    check('export'); panels = write_panels(Path(handoff['output_dir']),cases,results)
    budget.finish(); check('export')
    return dict(kind='gpc_raw_tl_fulltrain_result.v1',status='complete',forecast_records=48,
        optimizer_updates=2328,train_examples=2328,model_loads=4,eval_loss_batches=48,
        train_shape_counts=dict(T=1164,L=1164),checkpoints=checkpoints,profiles=profiles,
        raw_contract=RAW_CONTRACT,development=prepared['inputs']['development'],
        forecast_rows=rows,loss_rows=loss_rows,panels=panels,budget=budget.snapshot(),
        interpretation='Fixed development reload/diagnostic only; no dynamics or policy acceptance',
        physics=0,collection=0,policy_training=0)
