"""One authorized R2C1 native-GPC run with explicit saved-array dependencies.

Import and pure contracts do not import Torch, construct a model or run a GPU.
The calling CLI exclusively owns the single model-work ledger stage and clock.
"""
from __future__ import annotations
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
COHORTS=("train_visual","val_natural","val_motion")
ORIGINAL_WEIGHT_SHA="4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892"


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def safe_json(value):
    if isinstance(value,float) and not math.isfinite(value):return "NONFINITE:"+repr(value)
    if isinstance(value,dict):return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [safe_json(v) for v in value]
    return value


def atomic_json(path,value):
    import os
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+".tmp")
    with temp.open("w",encoding="utf-8") as stream:
        json.dump(safe_json(value),stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def atomic_npz(path,**arrays):
    import os
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def event(log_path,kind,**values):
    import os
    with Path(log_path).open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(safe_json(dict(kind=kind,**values)),ensure_ascii=False,allow_nan=False)+"\n");stream.flush();os.fsync(stream.fileno())


def validate_predictor_inputs(histories,actions,*,step,seed,window_ids):
    import numpy as np
    h,a=np.asarray(histories),np.asarray(actions);n=len(h)
    if (h.shape!=(n,4,3,96,96) or a.shape!=(n,4,2) or not np.issubdtype(h.dtype,np.floating)
            or not np.issubdtype(a.dtype,np.floating) or not np.isfinite(h).all() or not np.isfinite(a).all()
            or a.min()< -1 or a.max()>1 or len(window_ids)!=n or type(step)is not int or not 0<=step<4
            or type(seed)is not int or not 0<=seed<2**32):raise ValueError("Native normalized-history/action contract failed")
    if step==0 and (h.min()<0 or h.max()>1):raise ValueError("Initial real RGB must be0..1")
    return h,a


def normalized_predictor(sample_batch,*,before_call,raw_sink):
    """Pure injectable bridge; no action rescale, RGB clipping or output repair."""
    def predict(histories,normalized_actions,*,seed,cohort,step,window_ids):
        h,a=validate_predictor_inputs(histories,normalized_actions,step=step,seed=seed,window_ids=window_ids)
        before_call(cohort=cohort,step=step,window_ids=list(window_ids))
        raw=sample_batch(h.copy(),a.copy(),seed=seed)
        raw_sink(raw,histories=h,normalized_actions=a,seed=seed,cohort=cohort,step=step,window_ids=list(window_ids))
        return raw
    return predict


def torch_native_sample(model,source,sampler_settings):
    """Called only inside the authorized clock. Upstream sampler/math unchanged."""
    import numpy as np
    import torch
    sampler=source.DiffusionSampler(model,source.DiffusionSamplerConfig(**sampler_settings))
    device=next(model.parameters()).device
    def sample(histories,actions,*,seed):
        devices=[device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices),torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
            image=torch.as_tensor(np.asarray(histories,dtype=np.float32).copy(),device=device)
            action=torch.as_tensor(np.asarray(actions,dtype=np.float32).copy(),device=device)
            result,_=sampler.sample(image,action)
        return result.detach().float().cpu().numpy()
    return sample


def micro_loss_weight(microbatch=4,effective_batch=16):
    if (microbatch,effective_batch)!=(4,16):raise ValueError("R2C1 fixes micro4/effective16")
    return microbatch/effective_batch


def reserve_then_draw(work_budget,prepared,*,on_reserved):
    """Durable attempt reservation precedes all update draws and microbatches."""
    work_budget.reserve_update()
    on_reserved()
    return prepared.begin_update()


def checkpoint_contract(payload,config,identity):
    required={"model","optimizer","scaler","scheduler","rng","order","progress","config","identity","ledger"}
    if payload.get("kind")!="native_gpc_rgb_training.v1" or not required.issubset(payload):raise ValueError("Incomplete full checkpoint")
    if payload["config"]!=config or payload["identity"]!=identity:raise ValueError("Checkpoint identity/config differs")
    if payload["order"].get("pending_indices") is not None:raise ValueError("Midmicro data state is not a model checkpoint boundary")
    if not payload["progress"].get("complete_optimizer_boundary"):raise ValueError("Checkpoint is not a complete optimizer boundary")
    return True


def evaluation_inventory(root):
    root=Path(root)
    return [dict(file=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".npz",".json")]


def _tree_digest(value,torch):
    import numpy as np
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        elif isinstance(x,np.ndarray):h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=str):h.update(repr(key).encode());visit(x[key])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for item in x:visit(item)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def initial_tensor_summary(model,torch):
    rows=[]
    for name,tensor in model.state_dict().items():
        cpu=tensor.detach().cpu().contiguous();a=cpu.numpy();n=cpu.numel()
        rows.append(dict(name=name,shape=list(cpu.shape),dtype=str(cpu.dtype),numel=n,sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            finite=bool(torch.isfinite(cpu).all()),minimum=None if not n else float(cpu.min()),maximum=None if not n else float(cpu.max()),
            mean=None if not n else float(cpu.double().mean())))
    parameters=[dict(name=n,requires_grad=p.requires_grad,numel=p.numel()) for n,p in model.named_parameters()]
    return dict(state_tensors=rows,parameters=parameters,trainable_parameter_count=sum(p["numel"] for p in parameters if p["requires_grad"]),
        parameter_count=sum(p["numel"] for p in parameters),all_parameters_trainable=all(p["requires_grad"] for p in parameters))


def train(config,*,work_budget,execute=False):
    """Only root's once-claimed CLI invokes this real runner; no legacy train()."""
    if execute is not True or config.get("execution_authorized") is not True:raise PermissionError("Explicit R2C1 execution required")
    if (work_budget.max_attempts,work_budget.max_seconds)!=(300,1172.641) or config["budget"]!={"max_attempts":300,"max_model_seconds":1172.641,"latest_new_attempt_seconds":872.641,"tail_reserve_seconds":300.}:raise ValueError("Exactly one authorized R2C1R1 residual300/1172.641/872.641/300 external budget required")
    import numpy as np
    from .native_rgb_training import (construct_model,capture_rng,restore_rng,tensor_hashes,save_training_checkpoint,load_checkpoint_payload)
    from .raw_world_model_training import supervised_denoising_loss,make_generator,OPTIMIZER
    from .native_rgb_r2c0_data import R2C0Dataset,metadata_only_sampler_factory
    from .native_rgb_r2c0_interfaces import PreparedInputs
    from .native_rgb_r2c0_evaluation import rollout_raw
    from scripts.native_rgb_r2c0_evaluation import load_eval_cohort
    from scripts.native_rgb_r2c1_report import evaluate_saved_predictions,metrics_placeholder
    stage=Path(config["output"]);run=stage/"run";run.mkdir(exist_ok=False);events=run/"events.jsonl"
    model=optimizer=scaler=scheduler=generator=dataset=prepared=source=sample=None
    params=[];batch=loss=norm=None
    progress=dict(updates=0,samples=0,attempts=0,complete_optimizer_boundary=True,initial_parameter_sha256=None)
    result=dict(status="initializing",failure=None,progress=progress,checkpoint=None,reload_verified=False,evaluation_inventory=[],model_calls=0,model_call_attempts=0,
        model_calls_scope="completed native sampler batches only; training H1 forwards separately counted",native_sampler_batch_attempts=0,native_sampler_batch_completed=0,
        training_forward_calls=0,training_forward_completed_calls=0,training_backward_completed_microbatches=0,new_physics_controls=0,environment_constructions=0)
    clean=True;step_started=False;boundary_rng=boundary_data=None;last_saved_update=None;update_seconds=[]
    identity=dict(kind="R2C1-native-full-parameter.v1",config_sha256=sha(config["config_file"]),source_identity_sha256=sha(config["source_identity"]),initial_checkpoint_sha256=ORIGINAL_WEIGHT_SHA,
        frozen_inputs=deepcopy(config["input_hashes"]),objective="original_phase_two_H1_residual_mse")
    cohorts={};evalroot=run/"evaluation";evalroot.mkdir()
    calibration=json.loads(Path(config["calibration"]).read_text(encoding="utf-8"))
    def check():
        work_budget.heartbeat()
        if (stage/"STOP").exists() or (run/"STOP").exists():raise RuntimeError("STOP requested")
        if model is not None:
            import torch
            if torch.cuda.max_memory_reserved()>config["max_cuda_bytes"]:raise RuntimeError("6GiB CUDA reserved ceiling")
    def save_checkpoint(label):
        nonlocal last_saved_update
        if not clean or prepared.pending is not None or not progress["complete_optimizer_boundary"]:raise RuntimeError("No checkpoint from partial optimizer/microbatch")
        check();work_budget.before_checkpoint(run/"last.pt",estimated_seconds=30.)
        info=save_training_checkpoint(run/"last.pt",model,optimizer,scaler,scheduler,generator,prepared,progress,config,identity,ledger_state=work_budget.ledger.snapshot())
        last_saved_update=progress["updates"];result["checkpoint"]=info;event(events,"checkpoint",label=label,**info,model_seconds=work_budget.elapsed);check()
    def initialize_predictions():
        (evalroot/"copy_last").mkdir()
        for method in ("baseline","final"):
            folder=evalroot/method;folder.mkdir()
            for name,a in cohorts.items():
                n=len(a["window_ids"]);path=folder/("cohort-"+name)
                atomic_npz(path.with_suffix(".npz"),window_ids=np.asarray(a["window_ids"]),h1=np.full((n,3,96,96),np.nan,np.float32),h4=np.full((n,3,96,96),np.nan,np.float32),status_h1=np.full(n,"not_attempted",dtype="U128"),status_h4=np.full(n,"not_attempted",dtype="U128"))
                atomic_json(path.with_suffix(".json"),dict(cohort_identity=a["identity"],window_ids=a["window_ids"],step_records=[],hard_stop=False,evaluation_complete=False,attempted=False,raw_step_files=[],method=method))
        for method in ("copy_last","baseline","final"):
            for name,a in cohorts.items():
                label="original_GPC" if method=="baseline" else "trained" if method=="final" else "copy_last"
                atomic_json(evalroot/method/("cohort-"+name+".metrics.json"),metrics_placeholder(a,method=label,cohort=name))
    def evaluate(method):
        import torch
        old_mode=model.training;rng=capture_rng(generator);started=time.monotonic();folder=evalroot/method
        try:
            model.eval()
            for name,a in cohorts.items():
                check();rawfiles=[];batch_ids={tuple(a["window_ids"][i:i+4]):i//4 for i in range(0,len(a["window_ids"]),4)}
                raw_steps=[[None]*4 for _ in a["window_ids"]];positions={key:i for i,key in enumerate(a["window_ids"])}
                if method=="baseline":
                    work_budget.require_operation(40.,"copy-last endpoint/rawstep CPU scores");last=a["histories"][:,-1]
                    hold=dict(window_ids=a["window_ids"],h1=last.copy(),h4=last.copy(),status_h1=["ok"]*len(last),status_h4=["ok"]*len(last),planned_windows=len(last),hard_stop=False)
                    held=[[last[i].copy() for _ in range(4)] for i in range(len(last))]
                    metrics=evaluate_saved_predictions(a,hold,held,calibration,method="copy_last",cohort=name)
                    atomic_json(evalroot/"copy_last"/("cohort-"+name+".metrics.json"),metrics);check()
                def before_call(**context):
                    work_budget.require_operation(config["native_call_reserve_seconds"],"native H1/H4 batch");check()
                    result["model_call_attempts"]+=1
                    result["native_sampler_batch_attempts"]+=1
                    event(events,"native_call_start",method=method,**context,model_seconds=work_budget.elapsed)
                def sink(raw,**context):
                    raw=np.asarray(raw);batch=batch_ids[tuple(context["window_ids"])];rel=f"steps/{name}-batch{batch:02d}-step{context['step']}.npz";path=folder/rel
                    atomic_npz(path,raw=raw,window_ids=np.asarray(context["window_ids"]))
                    if raw.shape==(len(context["window_ids"]),3,96,96):
                        for i,key in enumerate(context["window_ids"]):raw_steps[positions[key]][context["step"]]=raw[i].copy()
                    else:
                        for key in context["window_ids"]:raw_steps[positions[key]][context["step"]]=raw.copy()
                    rawfiles.append(dict(file=rel,sha256=sha(path),bytes=path.stat().st_size,batch_index=batch,step=context["step"],window_ids=context["window_ids"],seed=context["seed"],raw_shape=list(raw.shape),raw_dtype=str(raw.dtype),input_history_sha256=hashlib.sha256(context["histories"].tobytes()).hexdigest(),normalized_actions_sha256=hashlib.sha256(context["normalized_actions"].tobytes()).hexdigest()))
                    result["model_calls"]+=1;result["native_sampler_batch_completed"]+=1;check()
                predictor=normalized_predictor(sample,before_call=before_call,raw_sink=sink)
                r=rollout_raw(predictor,a["histories"],a["absolute_actions"],window_ids=a["window_ids"],seed=5478,cohort=name,batch_size=4)
                path=folder/("cohort-"+name)
                atomic_npz(path.with_suffix(".npz"),window_ids=np.asarray(r["window_ids"]),h1=r["h1"],h4=r["h4"],status_h1=np.asarray(r["status_h1"],dtype="U128"),status_h4=np.asarray(r["status_h4"],dtype="U128"))
                atomic_json(path.with_suffix(".json"),dict(cohort_identity=a["identity"],window_ids=r["window_ids"],step_records=r["step_records"],hard_stop=r["hard_stop"],evaluation_complete=not r["hard_stop"],attempted=True,raw_step_files=rawfiles,method=method,checkpoint_sha256=None if result["checkpoint"] is None else result["checkpoint"]["sha256"],finite_OOR_retained_unmodified=True))
                event(events,"native_cohort",method=method,cohort=name,hard_stop=r["hard_stop"],statuses_h1=r["status_h1"],statuses_h4=r["status_h4"],calls=r["callback_calls"],model_seconds=work_budget.elapsed)
                if r["hard_stop"]:raise RuntimeError("Native rollout hard stop; raw/missing denominators retained")
                work_budget.require_operation(40.,"H1/H4 endpoint and all-four-step CPU scoring");check()
                metrics=evaluate_saved_predictions(a,r,raw_steps,calibration,method="original_GPC" if method=="baseline" else "trained",cohort=name)
                atomic_json(path.with_name(path.name+".metrics.json"),metrics);check()
        finally:
            restore_rng(rng,generator);model.train(old_mode)
        check();return time.monotonic()-started
    try:
        check();dataset=R2C0Dataset(config["data_manifest"],split="train",cache_bytes=1048576);order=metadata_only_sampler_factory(dataset,5478)
        for name in COHORTS:cohorts[name]=load_eval_cohort(config["data_manifest"],config["evaluation_indices"],name)
        prepared=PreparedInputs(dataset,order,dataset_identity=dataset.identity(),sampling_identity=order.identity(),evaluation_identity={n:a["identity"] for n,a in cohorts.items()})
        initialize_predictions();work_budget.require_operation(60.,"original full model construction/weight load")
        import torch
        if not torch.cuda.is_available():raise RuntimeError("Authorized native run requires CUDA; no fallback")
        torch.set_num_threads(2);random.seed(5478);np.random.seed(5478);torch.manual_seed(5478);torch.cuda.manual_seed_all(5478)
        torch.cuda.reset_peak_memory_stats()
        result["runtime"]=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda)
        model,source=construct_model(config["source_root"],config["init_checkpoint"],trainable=True)
        params=list(model.named_parameters());count=sum(p.numel() for _,p in params)
        if count!=8729827 or len(params)!=296 or any(not p.requires_grad for _,p in params):raise RuntimeError("Original full trainable parameter set differs")
        summary=initial_tensor_summary(model,torch)
        if not all(row["finite"] for row in summary["state_tensors"]):raise FloatingPointError("Original model has nonfinite state tensor")
        atomic_json(run/"initial-tensors.json",summary);result["initial_tensors_sha256"]=sha(run/"initial-tensors.json")
        progress["initial_parameter_sha256"]=tensor_hashes(model);result.update(parameter_count=count,parameter_tensors=len(params))
        optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=3000,eta_min=1e-5);scaler=torch.amp.GradScaler("cuda",enabled=True,init_scale=128.);generator=make_generator(5478,next(model.parameters()).device)
        work_budget.require_operation(10.,"native sampler setup");sample=torch_native_sample(model,source,config["sampler"])
        save_checkpoint("initial_t0");baseline=evaluate("baseline");work_budget.record_baseline(baseline);result["baseline_seconds"]=baseline
        final_estimate=max(120.,1.5*baseline)
        result["tail_plan"]=dict(final_predictions_and_CPUscore_seconds=final_estimate,save_seconds=30.,reload_seconds=60.,cleanup_seconds=15.,
            initial_minimum_tail_seconds=300.,overall_tail_estimate=max(300.,final_estimate+105.),no_repeated_save_reload_reserve_after_done=True)
        while progress["updates"]<300:
            check();estimate=max(4.,1.5*max(update_seconds[-10:],default=0.))
            if work_budget.elapsed+estimate>=work_budget.update_cutoff:
                progress["stop_reason"]="reserved_final_tail";break
            def reserved():
                nonlocal boundary_rng,boundary_data,clean,step_started
                progress["attempts"]+=1;boundary_rng=capture_rng(generator);boundary_data=prepared.state_dict();clean=False;step_started=False;progress["complete_optimizer_boundary"]=False
            try:indices=reserve_then_draw(work_budget,prepared,on_reserved=reserved)
            except Exception as exc:
                if getattr(exc,"updates_only_stop",False):progress["stop_reason"]=str(exc);break
                raise
            began=time.monotonic();event(events,"attempt_reserved_before_draw",attempt=progress["attempts"],indices=indices,windows=[dataset.window_rows[i] for i in indices],model_seconds=work_budget.elapsed)
            optimizer.zero_grad(set_to_none=True);loss_total=0.
            for micro_index in range(4):
                check();chunk=prepared.next_microbatch();batch={k:torch.as_tensor(v,device=next(model.parameters()).device) for k,v in chunk["model_inputs"].items()}
                result["training_forward_calls"]+=1
                with torch.autocast("cuda",dtype=torch.float16,enabled=True):loss,_=supervised_denoising_loss(model,batch,generator=generator)
                result["training_forward_completed_calls"]+=1
                if loss.dtype!=torch.float32 or not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite/nonFP32 original residual loss")
                scaler.scale(loss*micro_loss_weight()).backward();loss_total+=float(loss.detach())*.25
                result["training_backward_completed_microbatches"]+=1
                event(events,"microbatch_backward_complete",attempt=progress["attempts"],micro_index=micro_index,indices=chunk["indices"],loss=float(loss.detach()),model_seconds=work_budget.elapsed)
            check();scaler.unscale_(optimizer)
            if any(p.grad is None for _,p in params):raise FloatingPointError("Missing full-network gradients")
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if "first_gradient_norms" not in progress:progress["first_gradient_norms"]={n:float(p.grad.float().norm()) for n,p in params}
            old_scale=scaler.get_scale();step_started=True;scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<old_scale:raise FloatingPointError("AMP skipped update; attempt charged; no retry")
            if any(not bool(torch.isfinite(p).all()) for _,p in params):raise FloatingPointError("Nonfinite parameter after optimizer step")
            work_budget.complete_update();scheduler.step();progress["updates"]+=1;progress["samples"]+=16;progress["complete_optimizer_boundary"]=True;clean=True;step_started=False
            update_seconds.append(time.monotonic()-began)
            event(events,"update_complete",update=progress["updates"],attempt=progress["attempts"],loss=loss_total,gradient_norm=float(norm),lr=optimizer.param_groups[0]["lr"],scaler=scaler.get_scale(),cuda_reserved=torch.cuda.max_memory_reserved(),model_seconds=work_budget.elapsed)
            if progress["updates"]%25==0:print(json.dumps(dict(update=progress["updates"],attempt=progress["attempts"],loss=loss_total,model_seconds=work_budget.elapsed)),flush=True)
        if "stop_reason" not in progress:progress["stop_reason"]="stage_attempt_target"
        work_budget.require_operation(final_estimate+105.,"single complete tail: save+reload+final evaluation+cleanup")
        save_checkpoint("final_complete_boundary")
        work_budget.require_operation(final_estimate+75.,"remaining tail: reload+final evaluation+cleanup")
        before=dict(model=tensor_hashes(model),optimizer=_tree_digest(optimizer.state_dict(),torch),scaler=_tree_digest(scaler.state_dict(),torch),scheduler=_tree_digest(scheduler.state_dict(),torch),rng=_tree_digest(capture_rng(generator),torch),prepared=prepared.state_dict())
        payload=load_checkpoint_payload(run/"last.pt");checkpoint_contract(payload,config,identity)
        model.load_state_dict(payload["model"],strict=True);optimizer.load_state_dict(payload["optimizer"]);scaler.load_state_dict(payload["scaler"]);scheduler.load_state_dict(payload["scheduler"]);prepared.load_state_dict(payload["order"]);restore_rng(payload["rng"],generator);restored=deepcopy(payload["progress"]);del payload
        after=dict(model=tensor_hashes(model),optimizer=_tree_digest(optimizer.state_dict(),torch),scaler=_tree_digest(scaler.state_dict(),torch),scheduler=_tree_digest(scheduler.state_dict(),torch),rng=_tree_digest(capture_rng(generator),torch),prepared=prepared.state_dict())
        if before!=after or restored!=progress:raise RuntimeError("Full checkpoint reload state differs")
        result["reload_verified"]=True;event(events,"checkpoint_reload_verified",update=progress["updates"],model_seconds=work_budget.elapsed)
        work_budget.require_operation(final_estimate+15.,"remaining tail: final36 H1/H4 predictions/CPU scores and cleanup")
        result["final_evaluation_seconds"]=evaluate("final");result["final_parameter_sha256"]=tensor_hashes(model);result["parameter_tensors_changed"]=sum(result["final_parameter_sha256"][n]!=h for n,h in progress["initial_parameter_sha256"].items());result["status"]="complete"
    except BaseException as exc:
        result["failure"]=dict(type=type(exc).__name__,reason=str(exc),during_optimizer_step=step_started,clean_boundary=clean);result["status"]="stopped_on_error"
        event(events,"fatal_stop",**result["failure"],model_seconds=work_budget.elapsed)
        atomic_json(run/"partial-update.json",dict(failure=result["failure"],attempted_draws=prepared.delivered_draws if prepared is not None else 0,
            complete_updates=progress["updates"],charged_attempts=progress["attempts"],not_a_resume_checkpoint=True,prior_complete_payload_preserved=result["checkpoint"]))
        if model is not None and not clean and not step_started and boundary_rng is not None:
            try:
                restore_rng(boundary_rng,generator);prepared.load_state_dict(boundary_data);optimizer.zero_grad(set_to_none=True);clean=True;progress["complete_optimizer_boundary"]=True
            except BaseException as restore_exc:result["boundary_rewind_error"]=repr(restore_exc)
        if model is not None and clean and optimizer is not None and last_saved_update!=progress["updates"]:
            try:save_checkpoint("last_complete_after_error")
            except BaseException as save_exc:result["checkpoint_error"]=repr(save_exc)
    finally:
        result["evaluation_inventory"]=evaluation_inventory(evalroot);result["progress"]=deepcopy(progress)
        if dataset is not None:dataset.close()
        torch_module=sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_initialized():
            result["peak_cuda_reserved"]=torch_module.cuda.max_memory_reserved();result["peak_cuda_allocated"]=torch_module.cuda.max_memory_allocated()
            sample=model=optimizer=scaler=scheduler=generator=source=batch=loss=norm=None;params=[];gc.collect()
            result["cleanup_errors"]=[]
            for operation in (torch_module.cuda.synchronize,torch_module.cuda.empty_cache):
                try:operation()
                except BaseException as cleanup_exc:result["cleanup_errors"].append(repr(cleanup_exc))
            if result["cleanup_errors"]:
                result["status"]="stopped_on_error"
                if result["failure"] is None:
                    result["failure"]=dict(type="CleanupError",reason="; ".join(result["cleanup_errors"]),during_optimizer_step=False,clean_boundary=clean)
        result["budget_before_finish"]=work_budget.report();atomic_json(run/"training-result.json",result)
    return result
