"""Fixed-weight native sampler-step diagnostic; no optimizer or training path.

Importing this module does not import Torch or construct a model. One caller
owns the unchanged record-only ledger stage, storage checks and cleanup clock.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback
import uuid

from .native_rgb_r2c1r1_training import (
    atomic_json, atomic_npz, event, evaluation_inventory, initial_tensor_summary,
    normalized_predictor, sha, torch_native_sample)
from .native_rgb_r2c1r1_accounting_v2 import RecordOnlyTimeLedger, RecordOnlyTimeStage
from .native_rgb_r2c1_budget import R2C1StorageBudget
from .native_rgb_training import atomic_json as ledger_atomic

COHORTS=("train_visual", "val_natural", "val_motion")
CHECKPOINT_SHA="3c397035857fe98b896ea5083b52f32c129f1787c1be6884d4c740a9e58e5be0"
SAMPLER=dict(num_steps_denoising=3,sigma_min=.002,sigma_max=5,rho=7,order=1,s_churn=0,s_cond=0)


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_closed_prior(config):
    """Continue v2 accounting as-is; never rerun R1's historical migration."""
    path=Path(config["ledger_path"]);prior=Path(config["prior_ledger_copy"])
    if sha(path)!=config["prior_ledger_sha256"] or sha(prior)!=config["prior_ledger_sha256"]:
        raise ValueError("R2C2 closed prior/copy SHA differs")
    value=read(path)
    if (value!=read(prior) or value.get("kind")!="native_gpc_rgb_cost.v2-record-only-time"
            or value.get("max_seconds") is not None or value.get("active_stage") is not None
            or value.get("charged_updates")!=550 or value.get("completed_updates")!=550
            or abs(value.get("charged_seconds",-1)-1382.2180000000008)>1e-8
            or config.get("prior_attempts")!=550
            or abs(config.get("prior_model_seconds",-1)-value["charged_seconds"])>1e-8):
        raise ValueError("R2C2 must preserve closed v2 history550/1382.218 and no time cap")
    return value


class ExistingInferenceLedger(RecordOnlyTimeLedger):
    """Use public v2 heartbeat/end bookkeeping without its one-time migration."""
    def __init__(self,path,config,*,clock=None):
        self.path=Path(path).resolve();self.lock=self.path.with_name(self.path.name+".lock")
        self.clock=time.monotonic if clock is None else clock
        self.max_updates=3000;self.max_seconds=None;self.config=config
        verify_closed_prior(config)

    def start(self,phase):
        with self._locked():
            value=verify_closed_prior(self.config);token=uuid.uuid4().hex
            value["active_stage"]=dict(token=token,phase=str(phase),pid=__import__("os").getpid(),
                started_unix=time.time(),seconds_before=value["charged_seconds"],observed_seconds=0.,
                reserved_updates=0,completed_updates=0,time_policy="record_only_v2",
                planned_attempts=0,inference_settings=[8,16],training_authorized=False)
            ledger_atomic(self.path,value)
        return RecordOnlyTimeStage(self,token)

    def change(self,token,elapsed,*,reserve=False,complete=False,end=None):
        if reserve or complete:raise PermissionError("R2C2 is inference only; no update reservation/completion")
        return super().change(token,elapsed,end=end)


def storage_for_config(config):
    stage=Path(config["output"]).resolve();roots=[Path(p).resolve() for p in config["storage_roots"]]
    if stage not in roots or config.get("stage_artifact_cap_bytes")!=128*1024**2:
        raise ValueError("Explicit R2C2 stage root and128MiB artifact limit required")
    storage=R2C1StorageBudget(roots,stage_root=stage,source_files=config.get("new_stage_source_files",()),
        max_bytes=3*1024**3,atomic_reserve_bytes=256*1024**2,minimum_free_bytes=512*1024**2,interval_seconds=60.)
    storage.stage_limit_bytes=config["stage_artifact_cap_bytes"]
    return storage


def evaluate_setting(config,*,setting,sample,cohorts,calibration,check,result,events,call_stats):
    """Save every raw step, then use unchanged budgeted scoring; injectable CPU boundary."""
    import numpy as np
    from .native_rgb_r2c0_evaluation import rollout_raw
    from scripts.native_rgb_r2c1_report import evaluate_saved_predictions
    folder=Path(config["output"])/"run/evaluation"/f"steps{setting}"
    stats=result["settings"][str(setting)];began=time.monotonic()
    for name,prepared in cohorts.items():
        check();rawfiles=[];raw_steps=[[None]*4 for _ in prepared["window_ids"]]
        positions={key:i for i,key in enumerate(prepared["window_ids"])}
        batches={tuple(prepared["window_ids"][i:i+4]):i//4 for i in range(0,len(prepared["window_ids"]),4)}
        def before_call(**context):
            check();stats["sampler_batch_attempts"]+=1
            event(events,"native_call_start",setting=setting,**context,model_seconds=call_stats["elapsed"]())
        def sink(raw,**context):
            raw=np.asarray(raw);batch=batches[tuple(context["window_ids"])]
            relative=f"steps/{name}-batch{batch:02d}-step{context['step']}.npz";path=folder/relative
            atomic_npz(path,raw=raw,window_ids=np.asarray(context["window_ids"]))
            valid_shape=raw.shape==(len(context["window_ids"]),3,96,96)
            for i,key in enumerate(context["window_ids"]):
                raw_steps[positions[key]][context["step"]]=raw[i].copy() if valid_shape else raw.copy()
            rawfiles.append(dict(file=relative,sha256=sha(path),bytes=path.stat().st_size,
                batch_index=batch,step=context["step"],window_ids=context["window_ids"],seed=context["seed"],
                raw_shape=list(raw.shape),raw_dtype=str(raw.dtype),
                input_history_sha256=hashlib.sha256(context["histories"].tobytes()).hexdigest(),
                normalized_actions_sha256=hashlib.sha256(context["normalized_actions"].tobytes()).hexdigest()))
            stats["sampler_batch_completed"]+=1
            if valid_shape:stats["raw_frames_saved"]+=len(raw)
            if raw.dtype!=np.float32:
                raise TypeError("Native sample must remain original float32; wrong-dtype raw preserved")
            check()
        normalized=normalized_predictor(sample,before_call=before_call,raw_sink=sink)
        def predictor(history,actions,**context):
            try:
                return normalized(history,actions,**context)
            except BaseException as exc:
                # rollout_raw intentionally captures callback exceptions. Save
                # the complete original traceback before that boundary, also
                # for input/before_call/storage/raw_sink failures.
                event(events,"predictor_exception",setting=setting,**context,type=type(exc).__name__,
                    reason=str(exc),traceback=traceback.format_exc(),model_seconds=call_stats["elapsed"]())
                raise
        generated=rollout_raw(predictor,prepared["histories"],prepared["absolute_actions"],
            window_ids=prepared["window_ids"],seed=5478,cohort=name,batch_size=4)
        path=folder/("cohort-"+name)
        atomic_npz(path.with_suffix(".npz"),window_ids=np.asarray(generated["window_ids"]),h1=generated["h1"],h4=generated["h4"],
            status_h1=np.asarray(generated["status_h1"],dtype="U128"),status_h4=np.asarray(generated["status_h4"],dtype="U128"))
        atomic_json(path.with_suffix(".json"),dict(cohort_identity=prepared["identity"],window_ids=generated["window_ids"],
            step_records=generated["step_records"],hard_stop=generated["hard_stop"],
            evaluation_complete=not generated["hard_stop"],attempted=True,raw_step_files=rawfiles,
            method=f"steps{setting}",checkpoint_sha256=CHECKPOINT_SHA,finite_OOR_retained_unmodified=True,
            sampler=dict(config["sampler"],num_steps_denoising=setting)))
        event(events,"native_cohort",setting=setting,cohort=name,hard_stop=generated["hard_stop"],
            calls=generated["callback_calls"],model_seconds=call_stats["elapsed"]())
        if generated["hard_stop"]:raise RuntimeError("Native rollout hard stop; preserved raw/missing denominators, no retry")
        check();score_started=time.monotonic()
        metrics=evaluate_saved_predictions(prepared,generated,raw_steps,calibration,method="trained",cohort=name)
        atomic_json(path.with_name(path.name+".metrics.json"),metrics)
        stats["endpoint_rows_scored"]+=len(metrics["rows"]);stats["CPU_scoring_seconds"]+=time.monotonic()-score_started
        check()
    stats["evaluation_and_scoring_seconds"]=time.monotonic()-began


def infer(config,*,stage,storage,execute=False):
    if execute is not True or config.get("execution_authorized") is not True:
        raise PermissionError("Explicit fixed-weight inference authorization required")
    import numpy as np
    from scripts.native_rgb_r2c0_evaluation import load_eval_cohort
    from scripts.native_rgb_r2c1_report import metrics_placeholder
    from .gpc_world_model import load_world_model_source,world_model_spec
    from .native_rgb_training import load_checkpoint_payload,tensor_hashes
    root=Path(config["output"]);run=root/"run";run.mkdir(exist_ok=False);events=run/"events.jsonl"
    evalroot=run/"evaluation";evalroot.mkdir();model=source=sample=payload=None;handles=[]
    result=dict(status="initializing",failure=None,settings={},evaluation_inventory=[],
        optimizer_attempts=0,completed_updates=0,training_forward_calls=0,new_checkpoints_saved=0,
        model_checkpoint_loads=0,new_physics_controls=0,environment_constructions=0,
        automatic_retry=False,time_policy="record_only_v2",enforced_time_limit=None,
        denoiser_forward_counter_scope="actual inner_model forward pre/post hooks; sample calls denoise directly")
    counters=dict(attempts=0,completed=0);current_setting=None
    def check():
        stage.heartbeat();storage.check()
        if (root/"STOP").exists() or (run/"STOP").exists():raise RuntimeError("STOP requested")
        if model is not None:
            import torch
            if torch.cuda.max_memory_reserved()>config["max_cuda_bytes"]:raise RuntimeError("6GiB CUDA reserved ceiling")
            if model.training:raise RuntimeError("Inference model left eval mode")
    try:
        check();cohorts={name:load_eval_cohort(config["data_manifest"],config["evaluation_indices"],name) for name in COHORTS}
        calibration=read(config["calibration"])
        for setting in config["new_settings"]:
            folder=evalroot/f"steps{setting}";folder.mkdir()
            result["settings"][str(setting)]=dict(sampler_batch_attempts=0,sampler_batch_completed=0,
                raw_frames_saved=0,endpoint_rows_scored=0,denoiser_forward_attempts=0,
                denoiser_forward_completed=0,sampler_seconds=0.,CPU_scoring_seconds=0.,calls=[])
            for name,prepared in cohorts.items():
                n=len(prepared["window_ids"]);path=folder/("cohort-"+name)
                atomic_npz(path.with_suffix(".npz"),window_ids=np.asarray(prepared["window_ids"]),
                    h1=np.full((n,3,96,96),np.nan,np.float32),h4=np.full((n,3,96,96),np.nan,np.float32),
                    status_h1=np.full(n,"not_attempted",dtype="U128"),status_h4=np.full(n,"not_attempted",dtype="U128"))
                atomic_json(path.with_suffix(".json"),dict(cohort_identity=prepared["identity"],window_ids=prepared["window_ids"],
                    step_records=[],hard_stop=False,evaluation_complete=False,attempted=False,raw_step_files=[],method=f"steps{setting}",checkpoint_sha256=CHECKPOINT_SHA))
                atomic_json(path.with_name(path.name+".metrics.json"),metrics_placeholder(prepared,method="trained",cohort=name))
        import torch
        if not torch.cuda.is_available():raise RuntimeError("R2C2 requires existing CUDA device; no fallback")
        torch.set_num_threads(2);random.seed(5478);np.random.seed(5478);torch.manual_seed(5478);torch.cuda.manual_seed_all(5478)
        torch.cuda.reset_peak_memory_stats()
        result["runtime"]=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
            inference_autocast=False,input_dtype="float32",model_dtype="float32",native_denoiser_quantization_unchanged=True)
        check();source=load_world_model_source(config["source_root"]);spec=world_model_spec();inner=deepcopy(spec["model"])
        denoiser={key:inner.pop(key) for key in ("sigma_data","sigma_offset_noise","noise_previous_obs")}
        with torch.random.fork_rng(devices=[]):
            model=source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**inner),**denoiser)).float()
        payload=load_checkpoint_payload(config["checkpoint"]);result["model_checkpoint_loads"]+=1
        if payload["progress"]["updates"]!=300 or not payload["progress"]["complete_optimizer_boundary"]:
            raise ValueError("Only the complete300-update trusted checkpoint is authorized")
        model.load_state_dict(payload["model"],strict=True);del payload;payload=None
        model.to("cuda").requires_grad_(False).eval()
        initial=initial_tensor_summary(model,torch);initial_params=tensor_hashes(model)
        if initial["parameter_count"]!=8729827 or len(initial["state_tensors"])!=298 or len(initial_params)!=296 or not all(r["finite"] for r in initial["state_tensors"]):
            raise ValueError("Loaded full native model tensor identity/finite state differs")
        r1=read(Path(config["reuse_root"])/"run/training-result.json")
        if initial_params!=r1["final_parameter_sha256"]:raise ValueError("R2C2 loaded parameters differ from R1 final audit")
        atomic_json(run/"initial-tensors.json",initial);atomic_json(run/"initial-parameter-hashes.json",initial_params)
        result.update(parameter_count=initial["parameter_count"],parameter_tensors=len(initial_params),state_tensors=len(initial["state_tensors"]))
        result["construction_peak_cuda_reserved"]=torch.cuda.max_memory_reserved()
        result["construction_peak_cuda_allocated"]=torch.cuda.max_memory_allocated()
        def forward_start(module,args):counters["attempts"]+=1
        def forward_complete(module,args,output):counters["completed"]+=1
        handles=[model.inner_model.register_forward_pre_hook(forward_start),model.inner_model.register_forward_hook(forward_complete)]
        for setting in config["new_settings"]:
            current_setting=setting;check();stats=result["settings"][str(setting)]
            torch.cuda.reset_peak_memory_stats();sampler=torch_native_sample(model,source,dict(config["sampler"],num_steps_denoising=setting))
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
            evaluate_setting(config,setting=setting,sample=measured_sample,cohorts=cohorts,calibration=calibration,
                check=check,result=result,events=events,call_stats={"elapsed":stage.elapsed})
            stats["peak_cuda_reserved"]=torch.cuda.max_memory_reserved();stats["peak_cuda_allocated"]=torch.cuda.max_memory_allocated()
            if (stats["sampler_batch_attempts"],stats["sampler_batch_completed"],stats["raw_frames_saved"],stats["endpoint_rows_scored"])!=(36,36,144,72):
                raise RuntimeError("Completed setting work differs from frozen36-window scope")
            if stats["denoiser_forward_attempts"]!=stats["denoiser_forward_completed"] or stats["denoiser_forward_completed"]!=36*setting:
                raise RuntimeError("Observed inner-model forward counts differ from native first-order schedule")
            event(events,"setting_complete",setting=setting,sampler_batch_completed=stats["sampler_batch_completed"],
                denoiser_forward_completed=stats["denoiser_forward_completed"],model_seconds=stage.elapsed())
            atomic_json(run/"inference-result-progress.json",result)
            print(json.dumps(dict(setting=setting,complete=True,observed_denoiser_forwards=stats["denoiser_forward_completed"],model_seconds=stage.elapsed())),flush=True)
        check();final=initial_tensor_summary(model,torch);final_params=tensor_hashes(model)
        atomic_json(run/"final-tensors.json",final);atomic_json(run/"final-parameter-hashes.json",final_params)
        initial_states={r["name"]:(r["shape"],r["dtype"],r["sha256"]) for r in initial["state_tensors"]}
        final_states={r["name"]:(r["shape"],r["dtype"],r["sha256"]) for r in final["state_tensors"]}
        if initial_params!=final_params or initial_states!=final_states:raise RuntimeError("Fixed-weight/state inference modified model tensors")
        result.update(parameters_unchanged=True,all_state_tensors_unchanged=True,model_eval_throughout=True,
            checkpoint_sha256_after=sha(config["checkpoint"]))
        if result["checkpoint_sha256_after"]!=CHECKPOINT_SHA:raise RuntimeError("Trusted checkpoint file changed")
        result["status"]="complete"
    except BaseException as exc:
        result.update(status="stopped_on_error",failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),setting=current_setting))
        event(events,"fatal_stop",**result["failure"],model_seconds=stage.elapsed())
    finally:
        result["denoiser_forward_attempts"]=counters["attempts"];result["denoiser_forward_completed"]=counters["completed"]
        result["evaluation_inventory"]=evaluation_inventory(evalroot);result["cleanup_errors"]=[]
        for handle in handles:
            try:handle.remove()
            except BaseException as exc:result["cleanup_errors"].append(repr(exc))
        model=source=sample=payload=None
        if "torch" in locals() and torch.cuda.is_initialized():
            result["peak_cuda_reserved"]=max([torch.cuda.max_memory_reserved(),result.get("construction_peak_cuda_reserved",0)]+
                [s.get("peak_cuda_reserved",0) for s in result["settings"].values()])
            result["peak_cuda_allocated"]=max([torch.cuda.max_memory_allocated(),result.get("construction_peak_cuda_allocated",0)]+
                [s.get("peak_cuda_allocated",0) for s in result["settings"].values()])
            # Drop closures retaining the sampler/model before CUDA cleanup.
            sampler=measured_sample=None;gc.collect()
            for operation in (torch.cuda.synchronize,torch.cuda.empty_cache):
                try:operation()
                except BaseException as exc:result["cleanup_errors"].append(repr(exc))
        if result["cleanup_errors"] and result["failure"] is None:
            result.update(status="stopped_on_error",failure=dict(type="CleanupError",reason=";".join(result["cleanup_errors"])))
        atomic_json(run/"inference-result.json",result)
    return result
