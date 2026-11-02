"""Explicit full-state continuation of the same native GPC learning lineage.

Only root's once-claimed caller executes this module. No model work occurs on
import; old checkpoints, frozen source and training/evaluation math stay intact.
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
import traceback
from types import SimpleNamespace

from .native_rgb_r2c1r1_training import (atomic_json,atomic_npz,event,sha,
    normalized_predictor,torch_native_sample,initial_tensor_summary,evaluation_inventory,_tree_digest)

COHORTS=("train_visual","val_natural","val_motion")
TARGETS=(1000,3000,6000)
RESUME_SHA="3c397035857fe98b896ea5083b52f32c129f1787c1be6884d4c740a9e58e5be0"
R1_CONFIG_SHA="65551468b89763ecfa30ee8a67232d87813a843f8975f5f38706f96fccbb379a"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def legacy_prepared_snapshot(prepared):
    """Compare restored semantic state without hiding the new versioned scope."""
    state=deepcopy(prepared.state_dict());state["order"]=prepared.order.legacy_state_dict()
    state.pop("state_sha256",None)
    state["state_sha256"]=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return state


def restore_payload(payload,model,optimizer,scaler,scheduler,generator,prepared,*,restore_rng_fn):
    """Every saved training component is loaded; no fresh-order/noise restart."""
    if payload.get("kind")!="native_gpc_rgb_training.v1":raise ValueError("Complete native checkpoint required")
    progress=deepcopy(payload["progress"])
    if progress.get("updates")!=300 or progress.get("samples")!=4800 or not progress.get("complete_optimizer_boundary"):
        raise ValueError("Only the complete300-update lineage checkpoint is authorized")
    model.load_state_dict(payload["model"],strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    scheduler.load_state_dict(payload["scheduler"])
    prepared.load_state_dict(deepcopy(payload["order"]))
    restore_rng_fn(payload["rng"],generator)
    return progress


def fixed_loss_batch_arrays(prepared):
    """Saved truth is uint8; use the identical once-/255 training conversion."""
    import numpy as np
    from .native_rgb_r2c0_evaluation import normalize_actions
    history=np.asarray(prepared["histories"]);truth=np.asarray(prepared["truth_h1"])
    if history.dtype!=np.float32 or truth.dtype!=np.uint8:raise ValueError("Fixed-loss saved-array dtypes differ")
    target=truth.astype(np.float32)/np.float32(255)
    image=np.concatenate((history,target[:,None]),axis=1)
    actions=normalize_actions(np.asarray(prepared["absolute_actions"])[:,:4])
    if image.dtype!=np.float32 or image.min()<0 or image.max()>1:raise ValueError("Invalid normalized fixed-loss images")
    return image,actions


def scaled_optimizer_step(scaler,optimizer,*,application_count):
    """Observe real optimizer post-hook applications, never AdamW's None return."""
    old_scale=scaler.get_scale()
    if not math.isfinite(old_scale) or old_scale<=0:raise FloatingPointError("GradScaler scale nonfinite/nonpositive")
    before=application_count();scaler.step(optimizer);scaler.update()
    new_scale=scaler.get_scale();applied=application_count()-before
    if not math.isfinite(new_scale) or new_scale<=0:raise FloatingPointError("GradScaler scale nonfinite/nonpositive")
    if applied==0:
        if not new_scale<old_scale:raise FloatingPointError("Unapplied optimizer without real scale reduction")
    elif applied!=1 or new_scale<old_scale:raise FloatingPointError("Inconsistent real optimizer application/scale state")
    return dict(applied=applied,scale_before=old_scale,scale_after=new_scale)


def train(config,*,work_budget,execute=False):
    if execute is not True or config.get("execution_authorized") is not True or config.get("training_authorized") is not True:
        raise PermissionError("Directly authorized C3 training continuation required")
    if (config.get("time_policy")!="record_only_v2" or config.get("cap_time",False) is not None
            or work_budget.max_seconds is not None or work_budget.update_cutoff is not None):
        raise ValueError("Model elapsed time is statistics only, never a stopping cap")
    import numpy as np
    from .gpc_world_model import load_world_model_source,world_model_spec
    from .native_rgb_training import (capture_rng,restore_rng,tensor_hashes,
        save_training_checkpoint,load_checkpoint_payload)
    from .raw_world_model_training import supervised_denoising_loss,make_generator,OPTIMIZER
    from .native_rgb_r2c0_data import R2C0Dataset
    from .native_rgb_r2c0_interfaces import PreparedInputs
    from .native_rgb_r2c0_evaluation import rollout_raw,normalize_actions
    from .native_rgb_r2c3_continuation import make_training_order,advance_lr_after_success
    from scripts.native_rgb_r2c0_evaluation import load_eval_cohort
    from scripts.native_rgb_r2c1_report import evaluate_saved_predictions,metrics_placeholder
    stage=Path(config["output"]);run=stage/"run";run.mkdir(exist_ok=False);events=run/"events.jsonl"
    evalroot=run/"evaluation";evalroot.mkdir();milestones=run/"milestones";milestones.mkdir()
    model=optimizer=scaler=scheduler=generator=dataset=prepared=source=payload=None
    sample=sampler=measured_sample=batch=loss=norm=None;handles=[];params=[]
    progress=None;boundary_rng=boundary_data=None;clean=True;optimizer_step_started=False;last_saved_update=None;checkpoint_dirty=False
    counters=dict(attempts=0,completed=0,optimizer_applications=0);sampling_active=False;draw_indices=[]
    overall_peak=dict(reserved=0,allocated=0)
    result=dict(status="initializing",failure=None,progress=None,checkpoint=None,milestones=[],observations={},
        resume_checkpoint_loads=0,final_checkpoint_reload_verified=False,training_forward_calls=0,
        training_forward_completed_calls=0,training_backward_completed_microbatches=0,
        native_sampler_batch_attempts=0,native_sampler_batch_completed=0,denoiser_forward_attempts=0,
        denoiser_forward_completed=0,fixed_loss_forward_calls=0,new_physics_controls=0,environment_constructions=0,
        automatic_retry=False,time_policy="record_only_v2",enforced_time_limit=None,evaluation_inventory=[])
    identity=dict(kind="native-GPC-R2C3-full-state-continuation.v1",config_sha256=sha(config["config_file"]),
        source_identity_sha256=sha(config["source_identity"]),parent_checkpoint_sha256=RESUME_SHA,
        parent_resume_config_sha256=R1_CONFIG_SHA,resume_lineage_updates=300,target_lineage_updates=6000,
        native_objective="unchanged_phase_two_H1_residual_mse",frozen_inputs=deepcopy(config["input_hashes"]))
    cohorts={};calibration=read(config["calibration"])

    def peak():
        torch_module=sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_initialized():
            overall_peak["reserved"]=max(overall_peak["reserved"],torch_module.cuda.max_memory_reserved())
            overall_peak["allocated"]=max(overall_peak["allocated"],torch_module.cuda.max_memory_allocated())
        return overall_peak

    def check():
        work_budget.heartbeat();peak()
        if (stage/"STOP").exists() or (run/"STOP").exists():raise RuntimeError("STOP requested")
        if overall_peak["reserved"]>config["max_cuda_bytes"]:raise RuntimeError("6GiB CUDA reserved ceiling")

    def save_complete(label):
        nonlocal last_saved_update,checkpoint_dirty
        if not clean or prepared.pending is not None or not progress["complete_optimizer_boundary"]:
            raise RuntimeError("Refusing full checkpoint from incomplete optimizer boundary")
        check();work_budget.before_checkpoint(run/"last.pt",estimated_seconds=0.)
        info=save_training_checkpoint(run/"last.pt",model,optimizer,scaler,scheduler,generator,prepared,progress,config,identity,
            ledger_state=work_budget.ledger.snapshot())
        if info["bytes"]>112*1024**2:raise RuntimeError("Full persistent payload exceeds reserved112MiB slot")
        result["checkpoint"]=info;last_saved_update=progress["updates"];checkpoint_dirty=False
        event(events,"checkpoint",label=label,**info,model_seconds=work_budget.elapsed);check()

    def save_model_only():
        import torch
        path=milestones/f"model-{progress['updates']:06d}.pt"
        if path.exists():raise FileExistsError("Model-only milestone already exists")
        temp=path.with_suffix(".pt.tmp");state={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
        try:
            torch.save(state,temp)
            if temp.stat().st_size>36*1024**2:raise RuntimeError("Model-only milestone exceeds36MiB reserved slot")
            temp.replace(path)
        finally:state.clear()
        info=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),updates=progress["updates"],
            kind="R2C3-model-only-milestone.v1",not_a_resume_checkpoint=True,parent_checkpoint_sha256=RESUME_SHA,
            full_payload_at_observation=deepcopy(result["checkpoint"]),identity=identity,
            progress=dict(updates=progress["updates"],samples=progress["samples"],attempts=progress["attempts"]))
        atomic_json(path.with_suffix(".json"),info);result["milestones"].append(info);check()

    def initialize_outputs():
        for update in TARGETS:
            result["observations"][str(update)]={}
            for setting in (3,16):
                folder=evalroot/f"update{update}"/f"steps{setting}";folder.mkdir(parents=True)
                result["observations"][str(update)][str(setting)]=dict(sampler_batch_attempts=0,sampler_batch_completed=0,
                    raw_frames_saved=0,endpoint_rows_scored=0,denoiser_forward_attempts=0,denoiser_forward_completed=0,
                    sampler_seconds=0.,CPU_scoring_seconds=0.,calls=[])
                for name,a in cohorts.items():
                    n=len(a["window_ids"])
                    atomic_json(folder/f"cohort-{name}.json",dict(cohort_identity=a["identity"],window_ids=a["window_ids"],planned_windows=n,
                        status_h1=["not_attempted"]*n,status_h4=["not_attempted"]*n,step_records=[],hard_stop=False,
                        evaluation_complete=False,attempted=False,raw_step_files=[],raw_endpoint_refs={i:{"H1":None,"H4":None} for i in a["window_ids"]},
                        method="trained",lineage_updates=update,num_steps_denoising=setting))
                    atomic_json(folder/f"cohort-{name}.metrics.json",metrics_placeholder(a,method="trained",cohort=name))

    def fixed_loss(update):
        import torch
        old_mode=model.training;rng=capture_rng(generator);before=prepared.state_dict();rows=[];started=time.monotonic()
        try:
            model.eval()
            for name,a in cohorts.items():
                images,actions=fixed_loss_batch_arrays(a)
                for start in range(0,len(images),4):
                    check();ids=a["window_ids"][start:start+4]
                    domain=["R2C3-fixed-loss-v1",5478,name,start//4,ids]
                    seed=int(hashlib.sha256(json.dumps(domain,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()[:8],16)
                    local_generator=make_generator(seed,next(model.parameters()).device)
                    fixed_batch=dict(image=torch.as_tensor(images[start:start+4],device="cuda"),action=torch.as_tensor(actions[start:start+4],device="cuda"))
                    result["fixed_loss_forward_calls"]+=1
                    with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16,enabled=True):
                        measured,details=supervised_denoising_loss(model,fixed_batch,generator=local_generator)
                    if measured.dtype!=torch.float32 or not bool(torch.isfinite(measured)):raise FloatingPointError("Invalid fixed-window native loss")
                    rows.append(dict(cohort=name,batch=start//4,window_ids=ids,seed=seed,loss=float(measured),**details))
                    fixed_batch=measured=local_generator=None;check()
        finally:
            restore_rng(rng,generator);model.train(old_mode)
        if prepared.state_dict()!=before:raise RuntimeError("Fixed loss changed PreparedInputs order")
        value=dict(kind="R2C3-fixed-H1-residual-loss.v1",lineage_updates=update,rows=rows,
            cohort_means={name:sum(r["loss"] for r in rows if r["cohort"]==name)/3 for name in COHORTS},
            seconds=time.monotonic()-started,fixed_noise_across_observations=True,training_rng_restored=True,
            checkpoint_sha256=result["checkpoint"]["sha256"])
        atomic_json(run/f"fixed-loss-{update}.json",value);return value

    def evaluate(update,setting):
        nonlocal sampling_active,sampler,measured_sample
        import torch
        folder=evalroot/f"update{update}"/f"steps{setting}";stats=result["observations"][str(update)][str(setting)]
        old_mode=model.training;rng=capture_rng(generator);order_before=prepared.state_dict();started=time.monotonic()
        peak();torch.cuda.reset_peak_memory_stats()
        try:
            model.eval();sampler=torch_native_sample(model,source,dict(config["sampler"],num_steps_denoising=setting));sampling_active=True
            def measured_sample(history,actions,*,seed):
                torch.cuda.synchronize();began=time.monotonic();before=deepcopy(counters)
                try:return sampler(history,actions,seed=seed)
                finally:
                    torch.cuda.synchronize();seconds=time.monotonic()-began
                    delta={key:counters[key]-before[key] for key in counters}
                    stats["denoiser_forward_attempts"]+=delta["attempts"];stats["denoiser_forward_completed"]+=delta["completed"]
                    stats["sampler_seconds"]+=seconds
                    stats["calls"].append(dict(seed=seed,batch_size=len(history),seconds=seconds,
                        denoiser_forward_attempts=delta["attempts"],denoiser_forward_completed=delta["completed"]))
            for name,a in cohorts.items():
                check();rawfiles=[];raw_steps=[[None]*4 for _ in a["window_ids"]]
                positions={key:i for i,key in enumerate(a["window_ids"])}
                batches={tuple(a["window_ids"][i:i+4]):i//4 for i in range(0,len(a["window_ids"]),4)}
                refs={key:{"H1":None,"H4":None} for key in a["window_ids"]}
                def before_call(**context):
                    check();stats["sampler_batch_attempts"]+=1;result["native_sampler_batch_attempts"]+=1
                    event(events,"native_call_start",lineage_updates=update,setting=setting,**context,model_seconds=work_budget.elapsed)
                def sink(raw,**context):
                    raw=np.asarray(raw);index=batches[tuple(context["window_ids"])]
                    relative=f"steps/{name}-batch{index:02d}-step{context['step']}.npz";path=folder/relative
                    atomic_npz(path,raw=raw,window_ids=np.asarray(context["window_ids"]))
                    good_shape=raw.shape==(len(context["window_ids"]),3,96,96);digest=sha(path)
                    for i,key in enumerate(context["window_ids"]):
                        raw_steps[positions[key]][context["step"]]=raw[i].copy() if good_shape else raw.copy()
                        if good_shape and context["step"] in (0,3):
                            refs[key]["H1" if context["step"]==0 else "H4"]=dict(file=relative,row=i,step=context["step"],sha256=digest)
                    rawfiles.append(dict(file=relative,sha256=digest,bytes=path.stat().st_size,batch_index=index,step=context["step"],
                        window_ids=context["window_ids"],seed=context["seed"],raw_shape=list(raw.shape),raw_dtype=str(raw.dtype),
                        input_history_sha256=hashlib.sha256(context["histories"].tobytes()).hexdigest(),
                        normalized_actions_sha256=hashlib.sha256(context["normalized_actions"].tobytes()).hexdigest()))
                    stats["sampler_batch_completed"]+=1;result["native_sampler_batch_completed"]+=1
                    if good_shape:stats["raw_frames_saved"]+=len(raw)
                    if raw.dtype!=np.float32:raise TypeError("Native nonfloat32 raw preserved; stopping")
                    check()
                bridge=normalized_predictor(measured_sample,before_call=before_call,raw_sink=sink)
                def predictor(history,actions,**context):
                    try:return bridge(history,actions,**context)
                    except BaseException as exc:
                        event(events,"predictor_exception",lineage_updates=update,setting=setting,**context,
                            type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),model_seconds=work_budget.elapsed)
                        raise
                generated=rollout_raw(predictor,a["histories"],a["absolute_actions"],window_ids=a["window_ids"],seed=5478,cohort=name,batch_size=4)
                sidecar=dict(cohort_identity=a["identity"],window_ids=generated["window_ids"],planned_windows=len(a["window_ids"]),
                    status_h1=generated["status_h1"],status_h4=generated["status_h4"],step_records=generated["step_records"],
                    hard_stop=generated["hard_stop"],evaluation_complete=not generated["hard_stop"],attempted=True,
                    raw_step_files=rawfiles,raw_endpoint_refs=refs,method="trained",lineage_updates=update,
                    num_steps_denoising=setting,checkpoint_sha256=result["checkpoint"]["sha256"],finite_OOR_retained_unmodified=True)
                atomic_json(folder/f"cohort-{name}.json",sidecar)
                if generated["hard_stop"]:raise RuntimeError("Native rollout hard stop; all raw and missing denominators retained")
                check();score_started=time.monotonic()
                metrics=evaluate_saved_predictions(a,generated,raw_steps,calibration,method="trained",cohort=name)
                atomic_json(folder/f"cohort-{name}.metrics.json",metrics)
                stats["endpoint_rows_scored"]+=len(metrics["rows"]);stats["CPU_scoring_seconds"]+=time.monotonic()-score_started;check()
        finally:
            sampling_active=False;restore_rng(rng,generator);model.train(old_mode);peak()
            stats["peak_cuda_reserved"]=torch.cuda.max_memory_reserved();stats["peak_cuda_allocated"]=torch.cuda.max_memory_allocated()
            stats["evaluation_and_scoring_seconds"]=time.monotonic()-started
        if prepared.state_dict()!=order_before:raise RuntimeError("Native evaluation changed PreparedInputs order")
        if (stats["sampler_batch_attempts"],stats["sampler_batch_completed"],stats["raw_frames_saved"],stats["endpoint_rows_scored"])!=(36,36,144,72):raise RuntimeError("Incomplete frozen evaluation matrix")
        if stats["denoiser_forward_attempts"]!=stats["denoiser_forward_completed"] or stats["denoiser_forward_completed"]!=36*setting:raise RuntimeError("Actual sampler forward counts differ")
        event(events,"evaluation_complete",lineage_updates=update,setting=setting,observed_denoiser_forwards=stats["denoiser_forward_completed"],model_seconds=work_budget.elapsed)

    try:
        check();dataset=R2C0Dataset(config["data_manifest"],split="train",cache_bytes=1048576)
        cohorts={name:load_eval_cohort(config["data_manifest"],config["evaluation_indices"],name) for name in COHORTS};initialize_outputs()
        import torch
        if not torch.cuda.is_available():raise RuntimeError("C3 requires existing CUDA device, no fallback")
        torch.set_num_threads(2);random.seed(5478);np.random.seed(5478);torch.manual_seed(5478);torch.cuda.manual_seed_all(5478);torch.cuda.reset_peak_memory_stats()
        result["runtime"]=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda)
        source=load_world_model_source(config["source_root"]);inner=deepcopy(world_model_spec()["model"])
        denoiser={k:inner.pop(k) for k in ("sigma_data","sigma_offset_noise","noise_previous_obs")}
        with torch.random.fork_rng(devices=[]):model=source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**inner),**denoiser)).float()
        model.to("cuda").requires_grad_(True);model.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.));model.train()
        optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=3000,eta_min=1e-5)
        scaler=torch.amp.GradScaler("cuda",enabled=True,init_scale=128.);generator=make_generator(5478,next(model.parameters()).device)
        payload=load_checkpoint_payload(config["resume_checkpoint"]);result["resume_checkpoint_loads"]+=1
        if sha(config["parent_resume_config"])!=R1_CONFIG_SHA or payload["config"]!=read(config["parent_resume_config"]) or payload["identity"]["config_sha256"]!=R1_CONFIG_SHA:
            raise ValueError("Parent full payload config identity differs")
        order=make_training_order(dataset,payload["order"])
        prepared=PreparedInputs(dataset,order,dataset_identity=dataset.identity(),sampling_identity=order.identity(),evaluation_identity={n:a["identity"] for n,a in cohorts.items()})
        progress=restore_payload(payload,model,optimizer,scaler,scheduler,generator,prepared,restore_rng_fn=restore_rng)
        restored=dict(model=_tree_digest(model.state_dict(),torch)==_tree_digest(payload["model"],torch),
            optimizer=_tree_digest(optimizer.state_dict(),torch)==_tree_digest(payload["optimizer"],torch),
            scaler=_tree_digest(scaler.state_dict(),torch)==_tree_digest(payload["scaler"],torch),
            scheduler=_tree_digest(scheduler.state_dict(),torch)==_tree_digest(payload["scheduler"],torch),
            rng=_tree_digest(capture_rng(generator),torch)==_tree_digest(payload["rng"],torch),
            prepared_legacy_semantics=legacy_prepared_snapshot(prepared)==payload["order"],progress=progress==payload["progress"])
        if not all(restored.values()):raise RuntimeError("Complete300-update restore verification failed")
        atomic_json(run/"resume-verification.json",dict(passed=True,components=restored,
            old_prepared_state_sha256=payload["order"]["state_sha256"],new_prepared_state_sha256=prepared.state_dict()["state_sha256"],
            order_continuation_identity=order.continuation_identity(),order_restoration_audit=order.restoration_audit,
            parent_checkpoint_sha256=RESUME_SHA,progress_before_continuation=progress))
        del payload;payload=None
        if scheduler.last_epoch!=300:raise RuntimeError("Restored scheduler must retain last_epoch300")
        params=list(model.named_parameters());initial_params=tensor_hashes(model);summary=initial_tensor_summary(model,torch)
        if len(params)!=296 or summary["parameter_count"]!=8729827 or not summary["all_parameters_trainable"] or not all(r["finite"] for r in summary["state_tensors"]):raise RuntimeError("Full model tensor set differs")
        atomic_json(run/"initial-tensors.json",summary);atomic_json(run/"initial-parameter-hashes.json",initial_params)
        progress["prior_stage_stop_reason"]=progress.pop("stop_reason",None)
        progress["r2c3"]=dict(attempts=0,completed_updates=0,attempted_draws=0,actual_draws=0,resumed_from_updates=300,planned_target_updates=6000,
            AMP_skips=0,consecutive_overflows=0,max_consecutive_overflows_observed=0)
        def forward_start(module,args):
            if sampling_active:counters["attempts"]+=1
        def forward_done(module,args,output):
            if sampling_active:counters["completed"]+=1
        def optimizer_applied(optimizer,args,kwargs):counters["optimizer_applications"]+=1
        handles=[model.inner_model.register_forward_pre_hook(forward_start),model.inner_model.register_forward_hook(forward_done),
            optimizer.register_step_post_hook(optimizer_applied)]
        while progress["updates"]<6000:
            check();work_budget.reserve_update();progress["attempts"]+=1;progress["r2c3"]["attempts"]+=1
            boundary_rng=capture_rng(generator);boundary_data=prepared.state_dict();clean=False;optimizer_step_started=False;progress["complete_optimizer_boundary"]=False
            indices=prepared.begin_update();draw_indices.extend(indices);progress["r2c3"]["attempted_draws"]+=16;progress["r2c3"]["actual_draws"]+=16;checkpoint_dirty=True
            progress["samples"]=prepared.delivered_draws;progress["actual_draws_total"]=prepared.delivered_draws
            event(events,"attempt_reserved_before_draw",attempt=progress["r2c3"]["attempts"],lineage_attempt=progress["attempts"],indices=indices,
                window_ids=[dataset.window_rows[i]["window_id"] for i in indices],model_seconds=work_budget.elapsed)
            began=time.monotonic();optimizer.zero_grad(set_to_none=True);loss_total=0.;used_lr=optimizer.param_groups[0]["lr"]
            for micro_index in range(4):
                check();chunk=prepared.next_microbatch();batch={k:torch.as_tensor(v,device=next(model.parameters()).device) for k,v in chunk["model_inputs"].items()}
                result["training_forward_calls"]+=1
                with torch.autocast("cuda",dtype=torch.float16,enabled=True):loss,_=supervised_denoising_loss(model,batch,generator=generator)
                result["training_forward_completed_calls"]+=1
                if loss.dtype!=torch.float32 or not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite/nonFP32 original residual loss")
                scaler.scale(loss*.25).backward();loss_total+=float(loss.detach())*.25
                result["training_backward_completed_microbatches"]+=1
                event(events,"microbatch_backward_complete",attempt=progress["r2c3"]["attempts"],micro_index=micro_index,indices=chunk["indices"],loss=float(loss.detach()),model_seconds=work_budget.elapsed)
            check();scaler.unscale_(optimizer)
            if any(p.grad is None for _,p in params):raise FloatingPointError("Missing full-network gradients")
            finite_gradients=all(bool(torch.isfinite(p.grad).all()) for _,p in params)
            if not all(bool(torch.isfinite(p).all()) for _,p in params):raise FloatingPointError("Nonfinite model parameters")
            # An ordinary scaled-gradient overflow belongs to GradScaler. Do
            # not throw from clip before scaler can perform its real skip.
            before_skip_model=before_skip_optimizer=None
            if finite_gradients:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            else:
                norm=None;before_skip_model=_tree_digest(model.state_dict(),torch);before_skip_optimizer=_tree_digest(optimizer.state_dict(),torch)
            if finite_gradients and "first_gradient_norms" not in progress["r2c3"]:progress["r2c3"]["first_gradient_norms"]={n:float(p.grad.float().norm()) for n,p in params}
            optimizer_step_started=True
            outcome=scaled_optimizer_step(scaler,optimizer,application_count=lambda:counters["optimizer_applications"])
            old_scale=outcome["scale_before"];new_scale=outcome["scale_after"];applied=outcome["applied"]
            if applied==0:
                if not all(bool(torch.isfinite(p).all()) for _,p in params):raise FloatingPointError("Parameters nonfinite during AMP skip")
                if before_skip_model is not None and (before_skip_model!=_tree_digest(model.state_dict(),torch) or before_skip_optimizer!=_tree_digest(optimizer.state_dict(),torch)):
                    raise FloatingPointError("Real AMP skip unexpectedly changed model/optimizer state")
                prepared.order.extend_after_amp_skip()
                work_budget.record_amp_skip(attempt=progress["r2c3"]["attempts"],scale_before=old_scale,scale_after=new_scale)
                progress["r2c3"]["AMP_skips"]+=1;progress["r2c3"]["consecutive_overflows"]+=1
                progress["r2c3"]["max_consecutive_overflows_observed"]=max(progress["r2c3"]["max_consecutive_overflows_observed"],progress["r2c3"]["consecutive_overflows"])
                progress["actual_draws_total"]=prepared.delivered_draws
                event(events,"AMP_skipped_update",attempt=progress["r2c3"]["attempts"],scale_before=old_scale,scale_after=new_scale,
                    completed_updates=progress["updates"],optimizer_applied=False,scheduler_advanced=False,
                    consecutive_overflows=progress["r2c3"]["consecutive_overflows"],actual_draws_total=prepared.delivered_draws,
                    continuation_identity=prepared.order.continuation_identity(),model_seconds=work_budget.elapsed)
                optimizer.zero_grad(set_to_none=True);clean=True;optimizer_step_started=False;progress["complete_optimizer_boundary"]=True
                if progress["r2c3"]["consecutive_overflows"]>=config["max_consecutive_overflows"]:
                    raise FloatingPointError("16 consecutive real AMP overflows: numerical instability")
                check();continue
            if any(not bool(torch.isfinite(p).all()) for _,p in params):raise FloatingPointError("Nonfinite parameter after optimizer step")
            work_budget.complete_update();new_update=progress["updates"]+1
            progress["lr_continuation_state"]=advance_lr_after_success(scheduler,optimizer,new_update)
            progress["updates"]=new_update;progress["completed_update_draws"]=new_update*16;progress["r2c3"]["completed_updates"]+=1
            progress["r2c3"]["consecutive_overflows"]=0;progress["actual_draws_total"]=prepared.delivered_draws
            progress["complete_optimizer_boundary"]=True;clean=True;optimizer_step_started=False
            event(events,"update_complete",update=new_update,new_complete_updates=progress["r2c3"]["completed_updates"],attempt=progress["r2c3"]["attempts"],
                loss=loss_total,gradient_norm=float(norm),optimizer_lr_used=used_lr,lr_after_scheduler_or_hold=optimizer.param_groups[0]["lr"],
                scheduler_last_epoch=scheduler.last_epoch,lr_continuation_state=progress["lr_continuation_state"],scaler_before=old_scale,scaler_after=scaler.get_scale(),
                optimizer_applied=True,seconds=time.monotonic()-began,model_seconds=work_budget.elapsed)
            if new_update%100==0:print(json.dumps(dict(update=new_update,new_complete=progress["r2c3"]["completed_updates"],loss=loss_total,lr=used_lr,model_seconds=work_budget.elapsed)),flush=True)
            if new_update%config["checkpoint_interval_updates"]==0:save_complete("periodic_complete_boundary")
            if new_update in TARGETS:
                if last_saved_update!=new_update:save_complete("training_observation")
                if new_update in (1000,3000):save_model_only()
                if new_update==6000:
                    # One actual final reload of the full payload; no duplicate
                    # checkpoint file and no loss/order reset.
                    loaded=load_checkpoint_payload(run/"last.pt")
                    for name,live in (("model",model.state_dict()),("optimizer",optimizer.state_dict()),("scaler",scaler.state_dict()),("scheduler",scheduler.state_dict()),("rng",capture_rng(generator)),("order",prepared.state_dict()),("progress",progress)):
                        if _tree_digest(live,torch)!=_tree_digest(loaded[name],torch):raise RuntimeError("Final payload serialization differs: "+name)
                    model.load_state_dict(loaded["model"],strict=True);optimizer.load_state_dict(loaded["optimizer"]);scaler.load_state_dict(loaded["scaler"]);scheduler.load_state_dict(loaded["scheduler"])
                    prepared.load_state_dict(loaded["order"]);restore_rng(loaded["rng"],generator)
                    for name,live in (("model",model.state_dict()),("optimizer",optimizer.state_dict()),("scaler",scaler.state_dict()),("scheduler",scheduler.state_dict()),("rng",capture_rng(generator)),("order",prepared.state_dict()),("progress",progress)):
                        if _tree_digest(live,torch)!=_tree_digest(loaded[name],torch):raise RuntimeError("Actual final reload differs: "+name)
                    del loaded;result["final_checkpoint_reload_verified"]=True
                result["observations"][str(new_update)]["fixed_loss"]=fixed_loss(new_update)
                for setting in (3,16):evaluate(new_update,setting)
                atomic_json(run/"training-result-progress.json",result)
                observation_path=milestones/f"observation-{new_update}.json"
                if observation_path.exists():raise FileExistsError("Closed observation is immutable")
                point_inventory=evaluation_inventory(evalroot/f"update{new_update}")
                observation=dict(kind="R2C3-closed-training-observation.v1",closed=True,lineage_update=new_update,
                    progress=deepcopy(progress),fixed_loss=result["observations"][str(new_update)]["fixed_loss"],
                    settings={str(s):deepcopy(result["observations"][str(new_update)][str(s)]) for s in (3,16)},
                    evaluation_root=str(evalroot/f"update{new_update}"),evaluation_inventory=point_inventory,
                    stable_model_only=[deepcopy(m) for m in result["milestones"] if m["updates"]==new_update],
                    full_payload_at_observation=deepcopy(result["checkpoint"]),
                    full_path_is_rolling_reference_and_sha_is_historical=True,
                    model_seconds_statistics_at_point_close=work_budget.elapsed,
                    parent_checkpoint_sha256=RESUME_SHA,source_identity_sha256=identity["source_identity_sha256"],
                    whole_training_stage_closed=False,continuing_without_waiting=new_update<6000)
                atomic_json(observation_path,observation)
                print(json.dumps(dict(milestone_complete=new_update,observation=str(observation_path),sha256=sha(observation_path),model_seconds=work_budget.elapsed)),flush=True)
        progress["stop_reason"]="planned_three_training_observations_complete_not_convergence_claim"
        final_params=tensor_hashes(model)
        result.update(status="complete",final_parameter_sha256=final_params,
            parameter_tensors_changed_from_300=sum(initial_params[n]!=final_params[n] for n in initial_params))
        if progress["r2c3"]["completed_updates"]!=5700 or progress["updates"]!=6000:raise RuntimeError("Incorrect completed continuation scope")
    except BaseException as exc:
        result.update(status="stopped_on_error",failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),
            during_optimizer_step=optimizer_step_started,clean_boundary=clean))
        event(events,"fatal_stop",**result["failure"],model_seconds=work_budget.elapsed)
        if prepared is not None and progress is not None and not clean and not optimizer_step_started and boundary_rng is not None:
            try:
                restore_rng(boundary_rng,generator);prepared.load_state_dict(boundary_data);optimizer.zero_grad(set_to_none=True)
                progress["samples"]=prepared.delivered_draws;progress["actual_draws_total"]=prepared.delivered_draws
                clean=True;progress["complete_optimizer_boundary"]=True;result["failed_attempt_draw_rng_rewound_but_cost_retained"]=True
            except BaseException as rewind:result["rewind_error"]=repr(rewind)
        if model is not None and progress is not None and clean and (checkpoint_dirty or last_saved_update!=progress["updates"]):
            try:save_complete("last_complete_after_error")
            except BaseException as saving:result["checkpoint_error"]=repr(saving)
    finally:
        result["progress"]=deepcopy(progress);result["evaluation_inventory"]=evaluation_inventory(evalroot)
        result["denoiser_forward_attempts"]=counters["attempts"];result["denoiser_forward_completed"]=counters["completed"]
        result["optimizer_step_applications_observed"]=counters["optimizer_applications"]
        atomic_npz(run/"actual-new-draw-indices.npz",indices=np.asarray(draw_indices,dtype=np.int32))
        if dataset is not None:dataset.close()
        peak();result["peak_cuda_reserved"]=overall_peak["reserved"];result["peak_cuda_allocated"]=overall_peak["allocated"];result["cleanup_errors"]=[]
        for handle in handles:
            try:handle.remove()
            except BaseException as exc:result["cleanup_errors"].append(repr(exc))
        sample=sampler=measured_sample=model=optimizer=scaler=scheduler=generator=source=batch=loss=norm=payload=None;params=[]
        torch_module=sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_initialized():
            gc.collect()
            for operation in (torch_module.cuda.synchronize,torch_module.cuda.empty_cache):
                try:operation()
                except BaseException as exc:result["cleanup_errors"].append(repr(exc))
        if result["cleanup_errors"] and result["failure"] is None:result.update(status="stopped_on_error",failure=dict(type="CleanupError",reason=";".join(result["cleanup_errors"])))
        result["budget_before_finish"]=work_budget.report();atomic_json(run/"training-result.json",result)
    return result
