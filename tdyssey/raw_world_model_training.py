"""Finite raw T/L supervised denoising; no import-time Torch, arrays or training.

Only official train episodes are indexed/read. One four-history/four-control
window predicts its fifth frame. A caller executing train must own the external
process/resource guard; the command-line entry supplies it.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import gc
import hashlib
import io
import json
import math
from pathlib import Path

from .dataset import load_episode, read_manifest
from .gpc_world_model import world_model_spec
from .official_world_model import RAW_CONTRACT

SEED = 5478
ENVIRONMENT_FAMILY = "official-pusht-capitals-v1"
OBJECTIVE = dict(name="phase_two_one_step_residual_mse",history_frames=4,target_frames=1,
    sigma_data=.5,sigma_offset_noise=.1,noise_previous_obs=True,
    sigma=dict(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.),foreground_weights=False,replay=False)
OPTIMIZER = dict(lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
OBJECTIVE_AST_SHA256 = "9f7b06a20f70db0df8501644ca4400cc7ca08b012474959bc161d2ec18b3e156"


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def _sha(value):return hashlib.sha256(value).hexdigest()


def plan_training(dataset_roots, steps=None, seed=SEED):
    """Metadata-only, deterministic, balanced unique order; never reads held records."""
    if type(seed) is not int or seed!=SEED:raise ValueError("Raw SFT uses seed5478")
    roots=[Path(value).expanduser().resolve() for value in dataset_roots]
    if not roots or len(set(roots))!=len(roots):raise ValueError("Supply distinct finished official dataset roots")
    sources=[];episodes=[];seen=set();scene_splits={};image_splits={};missing_scenes=0
    for root in roots:
        payload=(root/"manifest.json").read_bytes();manifest=read_manifest(root)
        if json.loads(payload)!=manifest:raise ValueError("Manifest changed while planning")
        shape=manifest["config"].get("shape")
        if (shape not in ("T","L") or manifest["provenance"].get("environment_family")!=ENVIRONMENT_FAMILY
                or manifest["rates"]!=dict(simulation_hz=100,control_hz=10)):
            raise ValueError("Need official T/L 100Hz/10Hz datasets")
        membership={identifier:split for split,ids in manifest["splits"].items() for identifier in ids}
        source_index=len(sources)
        sources.append(dict(root=str(root),manifest_sha256=_sha(payload),shape=shape))
        for entry in manifest["episodes"]:
            identifier=entry["trajectory_id"];split=membership[identifier]
            if identifier in seen:raise ValueError("Repeated trajectory across dataset roots")
            seen.add(identifier)
            for item in entry["files"]:
                if (type(item.get("bytes")) is not int or item["bytes"]<=0 or not isinstance(item.get("sha256"),str)
                        or len(item["sha256"])!=64 or any(c not in "0123456789abcdef" for c in item["sha256"])):
                    raise ValueError("Episode payload needs positive size and fixed SHA256")
                if Path(item["name"]).suffix==".npz":
                    previous=image_splits.get(item["sha256"])
                    if previous is not None and previous!=split:raise ValueError("Image payload crosses whole-episode splits")
                    image_splits[item["sha256"]]=split
            scene=entry.get("collection_metadata",{}).get("initial_state_sha256")
            if scene is None:missing_scenes+=1
            elif scene in scene_splits and scene_splits[scene]!=split:raise ValueError("Initial scene crosses splits")
            else:scene_splits[scene]=split
            if split=="train" and entry["control_steps"]>=4:
                episodes.append(dict(source=source_index,shape=shape,trajectory_id=identifier,entry=deepcopy(entry)))
    pools={}
    for shape in ("T","L"):
        rows=sorted((row for row in episodes if row["shape"]==shape),key=lambda row:row["trajectory_id"])
        queues=[deque(range(3,row["entry"]["control_steps"])) for row in rows];pool=[]
        while any(queues):
            for row,queue in zip(rows,queues):
                if queue:pool.append(dict(episode=episodes.index(row),trajectory_id=row["trajectory_id"],shape=shape,anchor=queue.popleft()))
        pools[shape]=pool
    available={shape:len(pool) for shape,pool in pools.items()};maximum=2*min(available.values())
    steps=maximum if steps is None else steps
    if type(steps) is not int or steps<2 or steps%2 or steps>maximum:
        raise ValueError("Requested steps must be positive/even and fit both unique T/L pools")
    entries=[pools[("T","L")[index%2]][index//2] for index in range(steps)]
    result=dict(kind="raw_tl_sft_plan.v1",sources=sources,episodes=episodes,entries=entries,steps=steps,seed=seed,
        available_windows=available,unused_windows={shape:available[shape]-steps//2 for shape in ("T","L")},
        selected_shapes=["T","L"],whole_episode_splits=True,selected_split="train",epochs=1,resampling=False,
        scene_identity="metadata incomplete; independence unknown" if missing_scenes else "manifest initial-state identities",
        input_contract=deepcopy(RAW_CONTRACT),action_training="float32 absolute/511*2-1",objective=deepcopy(OBJECTIVE))
    result["list_sha256"]=_sha(encoded(entries));result["plan_sha256"]=_sha(encoded(result))
    return result


class RawWindowDataset:
    """One episode in memory; source hashes, timing and aligned raw targets are checked."""
    def __init__(self,plan):
        value=deepcopy(plan);digest=value.pop("plan_sha256")
        if digest!=_sha(encoded(value)) or value.get("selected_split")!="train":raise ValueError("Frozen plan changed")
        self._sources=tuple(value["sources"]);self._episodes=tuple(value["episodes"]);self._entries=tuple(value["entries"])
        self._cache_key=self._cache=None
    def __len__(self):return len(self._entries)
    def __getitem__(self,index):
        import numpy as np
        if type(index) is not int or not 0<=index<len(self):raise IndexError(index)
        selected=self._entries[index];row=self._episodes[selected["episode"]];source=self._sources[row["source"]]
        if self._cache_key!=selected["episode"]:
            root=Path(source["root"])
            if _sha((root/"manifest.json").read_bytes())!=source["manifest_sha256"]:raise ValueError("Dataset manifest changed")
            trajectory,record,arrays=load_episode(root,row["entry"])
            if (trajectory.trajectory_id!=row["trajectory_id"] or trajectory.control_dt!=.1
                    or any(step.before.shape!=row["shape"] or step.after.shape!=row["shape"] for step in trajectory.transitions)):
                raise ValueError("Official source shape/timing/identity differs")
            self._cache_key,self._cache=selected["episode"],(trajectory,arrays)
        trajectory,arrays=self._cache;anchor=selected["anchor"]
        image=arrays["images"][anchor-3:anchor+2].astype(np.float32)/np.float32(255)
        action=np.asarray([step.action for step in trajectory.transitions[anchor-3:anchor+1]],dtype=np.float32)
        if image.shape!=(5,3,96,96) or action.shape!=(4,2) or not np.isfinite(action).all() or (action<0).any() or (action>511).any():
            raise ValueError("Need five raw frames and four aligned absolute actions in[0,511]")
        return dict(image=image,action=action/np.float32(511)*np.float32(2)-np.float32(1),
                    trajectory_id=row["trajectory_id"],shape=row["shape"],anchor=anchor)


@dataclass(frozen=True)
class NoisePlan:
    sigma_cond: object
    history_offset: object
    history_epsilon: object
    sigma: object
    target_offset: object
    target_epsilon: object
    generator_before_sha256: str = "explicit"
    generator_after_sha256: str = "explicit"


def make_generator(seed, device="cpu"):
    import torch
    if type(seed) is not int or not 0<=seed<2**63: raise ValueError("Seed must be an unsigned 63-bit integer")
    return torch.Generator(device=device).manual_seed(seed)


def _state_hash(generator):
    return hashlib.sha256(bytes(generator.get_state().cpu().tolist())).hexdigest()


def _batch(batch, device=None):
    import torch
    if not isinstance(batch,dict) or not all(key in batch for key in ("image","action")):
        raise ValueError("Provide image and already-normalized action tensors")
    image,action=batch["image"],batch["action"]
    if (not isinstance(image,torch.Tensor) or not isinstance(action,torch.Tensor)
            or image.dtype!=torch.float32 or action.dtype!=torch.float32
            or image.ndim!=5 or tuple(image.shape[1:])!=(5,3,96,96)
            or tuple(action.shape)!=(image.shape[0],4,2) or image.shape[0]<1
            or image.requires_grad or action.requires_grad
            or not bool(torch.isfinite(image).all()) or not bool(torch.isfinite(action).all())
            or bool((image<0).any()) or bool((image>1).any())
            or bool((action< -1).any()) or bool((action>1).any())):
        raise ValueError("Expected finite constant float32 image[B,5,3,96,96] [0,1] and normalized action[B,4,2] [-1,1]")
    return {"image":image.to(device) if device is not None else image,
            "action":action.to(device) if device is not None else action}


def make_noise_plan(batch, generator):
    """Native draw order, on a dedicated generator, independent of global RNG."""
    import torch
    batch=_batch(batch,generator.device); image=batch["image"];b=image.shape[0]
    before=_state_hash(generator)
    def normal(shape): return torch.randn(shape,device=image.device,dtype=torch.float32,generator=generator)
    def sigma(): return (normal((b,))*1.2-1.2).exp().clamp(.002,20.)
    sigma_cond=sigma()
    history_offset=.1*normal((b,12,1,1));history_epsilon=normal((b,12,96,96))
    target_sigma=sigma()
    target_offset=.1*normal((b,3,1,1));target_epsilon=normal((b,3,96,96))
    return NoisePlan(sigma_cond,history_offset,history_epsilon,target_sigma,target_offset,target_epsilon,
                     before,_state_hash(generator))


def _validate_noise(plan, image):
    import torch
    if not isinstance(plan,NoisePlan): raise ValueError("Expected an explicit NoisePlan")
    b=image.shape[0]
    for name,shape in (("sigma_cond",(b,)),("history_offset",(b,12,1,1)),("history_epsilon",(b,12,96,96)),
                       ("sigma",(b,)),("target_offset",(b,3,1,1)),("target_epsilon",(b,3,96,96))):
        tensor=getattr(plan,name)
        if (not isinstance(tensor,torch.Tensor) or tuple(tensor.shape)!=shape or tensor.device!=image.device
                or tensor.dtype!=torch.float32 or tensor.requires_grad or not bool(torch.isfinite(tensor).all())):
            raise ValueError("NoisePlan tensor shape/device/dtype/finite mismatch: "+name)
    if any(bool((getattr(plan,name)<.002).any()) or bool((getattr(plan,name)>20).any()) for name in ("sigma_cond","sigma")):
        raise ValueError("NoisePlan sigma is outside pinned training bounds")


def supervised_denoising_loss(denoiser, batch, *, generator=None, noise_plan=None):
    """Exact one-step residual loss; native compute_model_output stays differentiable."""
    import torch
    import torch.nn.functional as F
    if (generator is None)==(noise_plan is None): raise ValueError("Provide exactly generator or matching NoisePlan")
    cfg=denoiser.cfg
    if (cfg.sigma_data!=.5 or cfg.sigma_offset_noise!=.1 or cfg.noise_previous_obs is not True
            or cfg.inner_model.num_steps_conditioning!=4 or getattr(denoiser,"is_upsampler",False)):
        raise ValueError("Denoiser conditioning differs from the pinned objective")
    device=next(denoiser.parameters()).device;batch=_batch(batch,device);image=batch["image"]
    plan=noise_plan if noise_plan is not None else make_noise_plan(batch,generator)
    _validate_noise(plan,image)
    b=image.shape[0];history=image[:,:4].reshape(b,12,96,96);clean=image[:,4]
    noisy_history=history+plan.history_offset+plan.history_epsilon*plan.sigma_cond[:,None,None,None]
    noisy_target=clean+plan.target_offset+plan.target_epsilon*plan.sigma[:,None,None,None]
    conditioners=denoiser.compute_conditioners(plan.sigma,plan.sigma_cond)
    output=denoiser.compute_model_output(noisy_target,noisy_history,batch["action"],conditioners)
    if output.shape!=clean.shape or not bool(torch.isfinite(output).all()): raise ValueError("Nonfinite/wrong-shape denoiser output")
    target=(clean-conditioners.c_skip*noisy_target)/conditioners.c_out
    loss=F.mse_loss(output,target)
    if not bool(torch.isfinite(loss)): raise ValueError("Nonfinite supervised denoising loss")
    return loss,{"loss_denoising":float(loss.detach()),"batch_size":b,"target_frames":1,
                 "model_output_calls":1,"inference_sampler_calls":0,"actions_renormalized":False,
                 "noise_generator_before_sha256":plan.generator_before_sha256,
                 "noise_generator_after_sha256":plan.generator_after_sha256}


def _write_json(path,value,maximum=4*1024**2):
    payload=encoded(value)
    if len(payload)>maximum:raise ValueError("Training metadata slot exceeded")
    with Path(path).open("xb") as stream:
        if stream.write(payload)!=len(payload):raise OSError("Short immutable metadata write")


def export_model(model,output,updates,identity,torch):
    """Two files only: model state_dict and raw provenance; no optimizer/RNG."""
    from .official_world_model import model_profile
    output=Path(output);weight=output/f"raw-step-{updates:06d}.pt";sidecar=weight.with_suffix(".json")
    if weight.exists() or sidecar.exists():raise FileExistsError("Checkpoint export slot is spent")
    state={name:value.detach().cpu() for name,value in model.state_dict().items()}
    try:
        if (not state or any(not name.startswith("inner_model.") or not bool(torch.isfinite(value).all()) for name,value in state.items())
                or sum(value.numel()*value.element_size() for value in state.values())>64*1024**2):
            raise ValueError("Expected finite bounded model-only state_dict")
        buffer=io.BytesIO();torch.save(state,buffer);payload=buffer.getvalue()
        if len(payload)>64*1024**2:raise ValueError("Checkpoint exceeds64MiB")
        with weight.open("xb") as stream:
            if stream.write(payload)!=len(payload):raise OSError("Short immutable checkpoint write")
    finally:state.clear()
    digest=_sha(payload)
    metadata=dict(kind="raw_tl_sft_model_only.v1",updates=updates,
        train_shapes=["T","L"],input_transform="identity",input_contract=RAW_CONTRACT["version"],
        raw_contract=deepcopy(RAW_CONTRACT),t1=False,optimizer_state=False,rng_state=False,
        checkpoint=dict(file=weight.name,bytes=len(payload),sha256=digest),**identity)
    _write_json(sidecar,metadata,65536)
    return dict(updates=updates,checkpoint=str(weight),sha256=digest,sidecar=str(sidecar),
        profile=model_profile(weight,digest,sidecar=sidecar,expected_sidecar_sha256=_sha(sidecar.read_bytes())))


def export_boundaries(steps,export_steps=()):
    if any(type(step) is not int or not 1<=step<=steps for step in export_steps):
        raise ValueError("Export boundaries must be integer updates inside this run")
    exports=sorted(set(export_steps)|{steps})
    if len(exports)>8 or any(type(step) is not int or not 1<=step<=steps for step in exports):
        raise ValueError("Choose at most8 distinct in-run model-only exports")
    return exports


def run_updates(model,optimizer,dataset,device,generator,counts,check,export,export_steps,torch,
                *,loss_function=supervised_denoising_loss):
    """Finite batch-one loop; arithmetic fixtures can inject a non-Torch facade."""
    selected=[parameter for parameter in model.parameters() if parameter.requires_grad]
    if not selected:raise ValueError("No intended trainable parameters")
    metrics=[]
    for index in range(len(dataset)):
        check("training",deepcopy(counts));sample=dataset[index]
        inputs={key:torch.from_numpy(sample[key][None]).to(device) for key in ("image","action")}
        counts["reserved_updates"]+=1;check("training",deepcopy(counts))
        optimizer.zero_grad(set_to_none=True)
        loss,details=loss_function(model,inputs,generator=generator);loss.backward()
        grads=[parameter.grad for parameter in selected if parameter.grad is not None]
        if not grads or any(not bool(torch.isfinite(gradient).all()) for gradient in grads):
            raise FloatingPointError("Missing/nonfinite intended gradients")
        del grads
        norm=torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in selected):raise FloatingPointError("Nonfinite updated weights")
        counts["completed_updates"]+=1;check("training",deepcopy(counts))
        step=index+1
        metrics.append(dict(step=step,shape=sample["shape"],trajectory_id=sample["trajectory_id"],anchor=sample["anchor"],
                            loss=details["loss_denoising"],gradient_norm=float(norm)))
        if step in export_steps:
            check("export",deepcopy(counts));export(step)
            counts["exported_steps"].append(step);check("export",deepcopy(counts))
    return metrics


def train(plan,source_root,profile,output,*,execute=False,check=None,export_steps=()):
    """One externally guarded training run; no resume, validation or selection."""
    if execute is not True:raise PermissionError("Training requires explicit execution")
    owner=getattr(check,"__self__",None)
    if not callable(check) or not callable(getattr(owner,"attach_torch",None)):
        raise PermissionError("Training requires the ownership guard callback before Torch")
    dataset=RawWindowDataset(plan);steps=len(dataset)
    exports=export_boundaries(steps,export_steps)
    output=Path(output).expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    _write_json(output/"training-plan.json",plan)
    counts=dict(model_instances=0,model_loads=0,reserved_updates=0,completed_updates=0,exported_steps=[])
    check("setup",deepcopy(counts))
    model=optimizer=generator=None;torch=None;metrics=[];checkpoints=[];failure=None
    try:
        import torch
        owner.attach_torch(torch)
        from .official_world_model import load_denoiser
        counts.update(model_instances=1,model_loads=1);check("model_loading",deepcopy(counts))
        model,device,_source=load_denoiser(profile,source_root,training=True,execute=True)
        optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER);generator=make_generator(SEED,device)
        check("training",deepcopy(counts))
        identity=dict(init_sha256=model.asset_identity["checkpoint"]["sha256"],
            data_identity=deepcopy(plan["sources"]),list_identity=dict(sha256=plan["list_sha256"],examples=steps),
            source_identity=dict(upstream_commit=world_model_spec()["source_commit"],
                training_module_sha256=_sha(Path(__file__).read_bytes())),objective=deepcopy(OBJECTIVE),seed=SEED)
        def export(step):checkpoints.append(export_model(model,output,step,identity,torch))
        metrics=run_updates(model,optimizer,dataset,device,generator,counts,check,export,exports,torch)
        optimizer.zero_grad(set_to_none=True)
    except BaseException as error:
        failure=error
        try:_write_json(output/"training-error.json",dict(error=type(error).__name__+": "+str(error),counts=counts,metrics=metrics))
        except BaseException as secondary:error.add_note("Secondary diagnostic write: "+str(secondary))
        raise
    finally:
        model=optimizer=generator=dataset=None;gc.collect()
        if torch is not None:
            try:torch.cuda.empty_cache()
            except BaseException as cleanup:
                if failure is None:raise
                failure.add_note("Secondary CUDA cleanup failure: "+str(cleanup))
    counts["model_instances"]=0;check("export",deepcopy(counts))
    report=dict(status="complete",counts=counts,metrics=metrics,checkpoints=checkpoints,
                plan_sha256=plan["plan_sha256"],no_optimizer_restore=True,no_resampling=True)
    _write_json(output/"training-report.json",report);check("complete",deepcopy(counts))
    return report
