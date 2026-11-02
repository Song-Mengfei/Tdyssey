"""Pinned GPC one-step supervised objective; all Torch/model work is lazy.

Preparation can import/audit this module with stdlib alone. The loss is the
t=5 subcase of phase-two Denoiser.forward, using its native differentiable
compute_model_output; the no-grad inference sampler is never a training path.
"""
from dataclasses import dataclass
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_SPEC = {
    "name": "pinned_phase_two_one_step_residual_mse", "history_frames": 4,
    "target_frames": 1, "image_shape": [3, 96, 96], "image_range": [0., 1.],
    "actions": "already normalized a/511*2-1; unchanged by loss", "action_shape": [4, 2],
    "sigma_log_loc": -1.2, "sigma_log_scale": 1.2, "sigma_min": .002, "sigma_max": 20.,
    "sigma_data": .5, "sigma_offset_noise": .1, "noise_previous_obs": True,
    "loss": "mean((model_output-(clean-c_skip*noisy)/c_out)^2)",
    "weighting": "implicit 1/c_out^2 reconstruction weighting; no extra weight",
    "offset": "independent per flattened channel, added without multiplying by sigma",
    "feedback": "t=5 has no later target; native quantized tail is unused by this loss",
}
OPTIMIZER_SPEC = {"name": "AdamW", "lr": .0001, "betas": [.9, .999], "eps": 1e-8,
                  "weight_decay": .01, "amsgrad": False}
REAL_TRAINABLE_PREFIXES = ("inner_model.",)
OBJECTIVE_SHA256 = hashlib.sha256(json.dumps(OBJECTIVE_SPEC,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): value.update(block)
    return value.hexdigest()


def audit_upstream_objective(gpc_root, manifest_path=None):
    """Read source bytes/AST only. This never imports Torch or upstream code."""
    root=Path(gpc_root).resolve(); manifest_path=Path(manifest_path or ROOT/"manifests/gpc_world_model.json")
    spec=json.loads(manifest_path.read_bytes())
    revision=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if revision!=spec["source_commit"]: raise ValueError("Pinned upstream revision differs")
    paths=["world_model_train_phase_two/train.py","world_model_train_phase_two/utils.py"]
    paths += ["world_model_train_phase_two/"+suffix for suffix in
              ("diffusion/denoiser.py","diffusion/inner_model.py","diffusion/blocks.py","diffusion/diffusion_sampler.py")]
    evidence={}
    for name in paths:
        expected=spec["source_sha256"].get(name)
        if expected is None:
            expected=spec["source_sha256"][name.replace("world_model_train_phase_two/","gpc_rank_evaluation/")]
        if _sha(root/name)!=expected: raise ValueError("Pinned objective source differs: "+name)
        if "/diffusion/" in name:
            peer=name.replace("world_model_train_phase_two/","gpc_rank_evaluation/")
            if _sha(root/peer)!=expected:raise ValueError("Evaluation/training diffusion copies differ: "+peer)
        text=(root/name).read_text(encoding="utf-8")
        tree=ast.parse(text)
        evidence[name]={"sha256":expected,"bytes":(root/name).stat().st_size,
            "definitions":[{"name":node.name,"line":node.lineno,"end_line":node.end_lineno}
                           for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.ClassDef))]}
        if "/diffusion/" in name:
            evidence[name]["byte_identical_rank_peer"]={"path":peer,"sha256":expected,
                                                       "bytes":(root/peer).stat().st_size}
    model=spec["model"]
    for key in ("sigma_data","sigma_offset_noise","noise_previous_obs"):
        if model[key]!=OBJECTIVE_SPEC[key]: raise ValueError("Manifest denoising setting differs")
    return {"kind":"gpc_supervised_objective_source_audit.v1","source_commit":revision,
        "manifest_sha256":_sha(manifest_path),"files":evidence,"objective":OBJECTIVE_SPEC,
        "objective_sha256":OBJECTIVE_SHA256,"optimizer":OPTIMIZER_SPEC,"trainable_prefixes":list(REAL_TRAINABLE_PREFIXES),
        "native_source_evidence":{
            "sigma_distribution":"world_model_train_phase_two/train.py:107-114 and diffusion/denoiser.py:57-64",
            "offset_and_pixel_noise":"world_model_train_phase_two/diffusion/denoiser.py:66-69",
            "conditioning_and_implicit_weight":"world_model_train_phase_two/diffusion/denoiser.py:71-83,134-138",
            "history_target_action_slices":"world_model_train_phase_two/diffusion/denoiser.py:118-124",
            "image_and_action_range":"world_model_train_phase_two/utils.py:97-102,195-209",
            "no_grad_quantized_tail_unused_at_t5":"world_model_train_phase_two/diffusion/denoiser.py:85-90,140-143",
            "optimizer":"world_model_train_phase_two/train.py:168-170"},
        "phase_two_and_evaluation_diffusion_bytes_match":True,
        "one_step_is_native_t5_subcase_not_original_t16_training_reproduction":True,
        "pilot_grad_clip":{"value":1.,"upstream_has_clip":False,"applied_here":False,"owner":"orchestrator"},
        "runtime_operations":{"Torch_imports":0,"real_model_constructions":0,"weight_loads":0,
                              "forwards":0,"training_updates":0,"sampler_calls":0,"physics":0}}


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


def validation_loss(denoiser,batch,*,noise_plan):
    """A reusable validation NoisePlan consumes neither training nor global RNG."""
    import torch
    was_training=denoiser.training
    try:
        denoiser.eval()
        with torch.no_grad(): loss,metrics=supervised_denoising_loss(denoiser,batch,noise_plan=noise_plan)
        return float(loss),metrics
    finally: denoiser.train(was_training)


def select_trainable_parameters(denoiser, prefixes=REAL_TRAINABLE_PREFIXES):
    """Explicit named selection; buffers/architecture are never replaced."""
    prefixes=tuple(prefixes)
    if not prefixes or any(not isinstance(p,str) or not p for p in prefixes) or len(set(prefixes))!=len(prefixes):
        raise ValueError("Provide unique nonempty trainable name prefixes")
    parameters=list(denoiser.named_parameters())
    if any(not any(name.startswith(prefix) for name,_ in parameters) for prefix in prefixes):
        raise ValueError("A trainable prefix matched no parameter")
    selected=[];frozen=[]
    for name,parameter in parameters:
        trainable=any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(trainable);parameter.grad=None
        (selected if trainable else frozen).append({"name":name,"shape":list(parameter.shape),"numel":parameter.numel()})
    if not selected: raise ValueError("Empty optimizer parameter selection")
    return {"prefixes":list(prefixes),"trainable":selected,"frozen":frozen,
            "trainable_numel":sum(p["numel"] for p in selected),"frozen_numel":sum(p["numel"] for p in frozen),
            "selection_sha256":hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()}


def make_optimizer(denoiser, config=None):
    """The pinned AdamW settings. Gradient clipping belongs to the pilot runner."""
    settings={**OPTIMIZER_SPEC,**(config or {})}
    if isinstance(settings.get("betas"),tuple):settings["betas"]=list(settings["betas"])
    if settings!=OPTIMIZER_SPEC:raise ValueError("Optimizer differs from fixed adaptation settings")
    parameters=[p for p in denoiser.parameters() if p.requires_grad]
    if not parameters:raise ValueError("Select trainable parameters before constructing the optimizer")
    import torch
    return torch.optim.AdamW(parameters,lr=.0001,betas=(.9,.999),eps=1e-8,weight_decay=.01,amsgrad=False)


def make_real_denoiser(gpcroot, initckpt, device, config, state_dict=None):
    """Future execute-only factory; no sampler, policy, checkpoint write or update.

    This is not execution authority by itself. The caller must reserve work and
    validate its explicit user authorization before passing execute=True.
    Source/checkpoint checks are repeated here before any model construction.
    """
    if not isinstance(config,dict) or config.get("execute") is not True:
        raise PermissionError("Real GPC construction requires a separately authorized explicit execute")
    if config.get("evaluation_only") is True and state_dict is None:
        raise ValueError("Eval-only construction requires independently verified model-only state")
    if config.get("objective",OBJECTIVE_SPEC)!=OBJECTIVE_SPEC:
        raise ValueError("Objective config differs from the audited one-step objective")
    prefixes=tuple(config.get("trainable_prefixes",REAL_TRAINABLE_PREFIXES))
    if prefixes!=REAL_TRAINABLE_PREFIXES:raise ValueError("Real adaptation trains exactly all inner_model parameters")
    optimizer_settings={**OPTIMIZER_SPEC,**config.get("optimizer",{})}
    if isinstance(optimizer_settings.get("betas"),tuple):optimizer_settings["betas"]=list(optimizer_settings["betas"])
    if optimizer_settings!=OPTIMIZER_SPEC:raise ValueError("Real optimizer settings differ")
    audit=audit_upstream_objective(gpcroot)
    spec=json.loads((ROOT/"manifests/gpc_world_model.json").read_bytes())
    if state_dict is None:
        checkpoint=Path(initckpt)
        if checkpoint.stat().st_size!=spec["checkpoint"]["bytes"] or _sha(checkpoint)!=spec["checkpoint"]["sha256"]:
            raise ValueError("Initialization checkpoint differs from the exact pinned phase-two asset")
    from .gpc_world_model import load_world_model_source
    import torch
    source=load_world_model_source(gpcroot,spec)
    model_cfg=dict(spec["model"])
    denoiser_cfg={name:model_cfg.pop(name) for name in ("sigma_data","sigma_offset_noise","noise_previous_obs")}
    # Constructor randomness is overwritten by the pinned state, and must not
    # consume the caller's independent training/validation/sampling RNG streams.
    with torch.random.fork_rng(devices=[]):
        inner=source.InnerModelConfig(**model_cfg)
        denoiser=source.Denoiser(source.DenoiserConfig(inner_model=inner,**denoiser_cfg)).float()
    # A supplied adapted state is loaded only after the caller's independent
    # selected-checkpoint identity check. No optimizer/count/ledger is restored.
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True) if state_dict is None else state_dict
    if not isinstance(payload,dict) or not payload or not all(isinstance(v,torch.Tensor) and bool(torch.isfinite(v).all()) for v in payload.values()):
        raise ValueError("Non-tensor/nonfinite initialization state")
    if set(payload)!=set(denoiser.state_dict()) or any(payload[name].shape!=value.shape or payload[name].dtype!=value.dtype for name,value in denoiser.state_dict().items()):
        raise ValueError("Checkpoint keys/shapes/dtypes differ from pinned architecture")
    denoiser.load_state_dict(payload,strict=True);del payload
    evaluation_only=config.get("evaluation_only") is True
    if evaluation_only:
        denoiser.to(torch.device(device)).requires_grad_(False).eval()
    else:
        denoiser.to(torch.device(device)).train()
        denoiser.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.))
    if any(not name.startswith("inner_model.") for name,_ in denoiser.named_parameters()):
        raise ValueError("Unexpected upstream top-level parameter outside inner_model")
    selection=None if evaluation_only else select_trainable_parameters(denoiser,prefixes)
    denoiser.supervised_adaptation_provenance={"objective_audit":audit,"parameter_selection":selection,
        "checkpoint_sha256":spec["checkpoint"]["sha256"] if state_dict is None else None,
        "adapted_checkpoint_validation_required_of_caller":state_dict is not None,
        "checkpoint_identity_validated_here":state_dict is None,
        "checkpoint_identity":config.get("validated_checkpoint_identity"),
        "sampler_constructed":False,"policy_loaded":False}
    denoiser.supervised_adaptation_source=source
    return denoiser


def make_sampler(denoiser, gpcroot=None, source=None):
    """Future evaluation only: native three-Euler sampler; no call is made here."""
    if source is None:source=getattr(denoiser,"supervised_adaptation_source",None)
    if source is None:
        if gpcroot is None:raise ValueError("Provide the verified upstream source or its pinned root")
        audit_upstream_objective(gpcroot)
        from .gpc_world_model import load_world_model_source
        source=load_world_model_source(gpcroot)
    spec=json.loads((ROOT/"manifests/gpc_world_model.json").read_bytes())
    denoiser.eval()
    return source.DiffusionSampler(denoiser,source.DiffusionSamplerConfig(**spec["sampler"]))
