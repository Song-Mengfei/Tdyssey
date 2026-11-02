"""Small actor deltas and reproducible update state; no physical restart claim.

The verified compact base supplies every frozen parameter, encoder, normalizer
and buffer. Only the selected actor parameters are copied to CPU here. Optimizer
moments and the critic are also saved, so total bytes exceed the actor delta.
Loading reconstructs one actor from that same base; it does not restore a physics
engine, contact cache, observation history or an in-progress simulator episode.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import re

import torch

from .official import OfficialDiffusionPolicy, official_spec
from .policy import hash_file
from .policy_rl import DenoisingPolicy, SampledChain, SamplerConfig, _canonical_scheduler_signature


SCHEMA_VERSION = 1


def _tensor_tree(value, device=None):
    """Allow only weights_only-compatible data, optionally copying tensors."""
    if isinstance(value, torch.Tensor):
        return value if device is None else value.detach().to(device=device, copy=True)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (tuple, list):
        return type(value)(_tensor_tree(item, device) for item in value)
    if isinstance(value, dict) and all(type(key) in (str, int) for key in value):
        return {key: _tensor_tree(item, device) for key, item in value.items()}
    raise ValueError(f"Checkpoint data must contain primitives and tensors, got {type(value).__name__}")


def _base_identity(value):
    if (not isinstance(value, dict) or not isinstance(value.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
            or type(value.get("bytes")) is not int or value["bytes"] <= 0):
        raise ValueError("base_identity needs a verified lowercase sha256 and positive byte count")
    return _tensor_tree(value, "cpu")


def _official_identity():
    spec = official_spec()
    return {name: spec[name] for name in ("source_commit", "source_sha256", "policy_config")}


def _parameter_groups(optimizer, named_parameters):
    names = {id(parameter): name for name, parameter in named_parameters.items()}
    groups = []
    for group in optimizer.param_groups:
        if any(id(parameter) not in names for parameter in group["params"]):
            raise ValueError("Optimizer contains a parameter outside its declared model selection")
        groups.append([names[id(parameter)] for parameter in group["params"]])
    flattened = [name for group in groups for name in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(named_parameters):
        raise ValueError("Optimizer must include each declared parameter exactly once")
    return groups


def _selected_parameters(actor):
    parameters = dict(actor.network.named_parameters())
    selection = actor.selection
    if (len(selection.names) != len(set(selection.names)) or not selection.names
            or len(selection.names) != len(selection.counts)
            or any(name not in parameters for name in selection.names)
            or tuple(parameters[name].numel() for name in selection.names) != selection.counts
            or {name for name, parameter in parameters.items() if parameter.requires_grad} != set(selection.names)):
        raise ValueError("Actor trainable parameters differ from its recorded selection")
    return {name: parameters[name] for name in selection.names}


def _capture_rng(generator):
    import numpy as np

    if not isinstance(generator, torch.Generator):
        raise ValueError("Provide the actor's explicit local torch.Generator")
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    numpy_state = np.random.get_state()
    return _tensor_tree({
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda,
        "local_generator": {"device": str(generator.device), "state": generator.get_state()},
        "python": random.getstate(),
        "numpy": {"algorithm": numpy_state[0],
                  "keys": torch.tensor(numpy_state[1].astype("int64")),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
    }, "cpu")


def _local_generator(rng, device, restore_rng):
    source_device = torch.device(rng["local_generator"]["device"])
    target_device = torch.device(device)
    if source_device.type != target_device.type:
        if restore_rng:
            raise ValueError("CPU/CUDA generator algorithms differ; RNG replay requires the saved device type")
        return None
    generator = torch.Generator(device=target_device)
    generator.set_state(rng["local_generator"]["state"])
    return generator


def _restore_rng(rng):
    import numpy as np

    cuda_states = rng["torch_cuda"]
    if cuda_states and (not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count()):
        raise ValueError("Restoring global CUDA RNG requires the saved number of available CUDA devices")
    numpy_state = rng["numpy"]
    if (numpy_state["algorithm"] != "MT19937" or numpy_state["keys"].shape != (624,)
            or not 0 <= numpy_state["position"] <= 624 or numpy_state["has_gauss"] not in (0, 1)):
        raise ValueError("Invalid saved NumPy MT19937 RNG state")
    # Construct NumPy's original uint32 array only after safe tensor-only loading.
    numpy_tuple = (numpy_state["algorithm"], numpy_state["keys"].numpy().astype("uint32"),
                   numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"])
    torch.set_rng_state(rng["torch_cpu"])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    random.setstate(rng["python"])
    np.random.set_state(numpy_tuple)


def save_update_checkpoint(path, actor, critic, actor_optimizer, critic_optimizer, *,
                           base_identity, generator, metadata):
    """Save a new tensor-only delta artifact; refuse to overwrite any output.

    base_identity must describe the compact artifact already verified when this
    actor was loaded. This caller-supplied provenance is not a new EMA-equality
    claim. Optimizer parameter order is recorded for checked later restoration.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    selected = _selected_parameters(actor)
    critic_parameters = dict(critic.named_parameters())
    if (set(critic.config) != {"input_dim", "hidden_dim"}
            or any(type(value) is not int or value <= 0 for value in critic.config.values())):
        raise ValueError("Critic config must specify positive input_dim and hidden_dim")
    payload = {
        "schema_version": SCHEMA_VERSION, "kind": "tdyssey.policy_update_delta",
        "base_identity": _base_identity(base_identity), "official": _official_identity(),
        "selection": asdict(actor.selection), "sampler_config": asdict(actor.config),
        "scheduler_signature": actor.scheduler_signature,
        "actor_parameters": selected,
        "critic": {"config": dict(critic.config), "state_dict": critic.state_dict()},
        "optimizers": {"actor": actor_optimizer.state_dict(), "critic": critic_optimizer.state_dict()},
        "optimizer_types": {name: f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
                            for name, optimizer in (("actor", actor_optimizer), ("critic", critic_optimizer))},
        "optimizer_parameter_names": {"actor": _parameter_groups(actor_optimizer, selected),
                                      "critic": _parameter_groups(critic_optimizer, critic_parameters)},
        "rng": _capture_rng(generator), "metadata": metadata,
    }
    payload = _tensor_tree(payload, "cpu")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a second writer after exists().
    with path.open("xb") as output:
        try:
            torch.save(payload, output)
        except BaseException:
            output.close()
            path.unlink()
            raise
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": hash_file(path)}


@dataclass
class LoadedCheckpoint:
    official: object
    actor: DenoisingPolicy
    critic: torch.nn.Module
    payload: dict
    generator: torch.Generator | None


def _construct_critic(config, device):
    from .policy_training import VisualCritic

    if (set(config) != {"input_dim", "hidden_dim"}
            or any(type(value) is not int or value <= 0 for value in config.values())):
        raise ValueError("Invalid saved critic configuration")
    return VisualCritic(**config).to(device)


def _validate_state_dict(saved, expected, label):
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise ValueError(f"{label} parameter keys differ from the reconstructed model")
    for name, tensor in saved.items():
        reference = expected[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != reference.shape
                or tensor.dtype != reference.dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid {label} parameter shape/dtype/value: {name}")


def load_update_checkpoint(path, *, source_root, base_checkpoint, device="cuda", restore_rng=False):
    """Rebuild from the verified base, then strictly apply only selected weights.

    Raw complete optimizer states are in payload['optimizers']; use
    restore_optimizers with newly constructed optimizers to resume their state.
    With restore_rng=False global RNGs are left at their post-construction state.
    Local RNG is reconstructed when the requested device type matches; otherwise
    generator=None and the original state remains available in the payload.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _tensor_tree(payload)
    expected_keys = {"schema_version", "kind", "base_identity", "official", "selection", "sampler_config",
                     "scheduler_signature", "actor_parameters", "critic", "optimizers", "optimizer_types",
                     "optimizer_parameter_names", "rng", "metadata"}
    if (not isinstance(payload, dict) or set(payload) != expected_keys
            or payload["schema_version"] != SCHEMA_VERSION or payload["kind"] != "tdyssey.policy_update_delta"):
        raise ValueError("Unsupported update checkpoint schema")
    identity = _base_identity(payload["base_identity"])
    if payload["official"] != _official_identity():
        raise ValueError("Saved official source/config differs from this pinned implementation")
    if Path(base_checkpoint).stat().st_size != identity["bytes"]:
        raise ValueError("Base checkpoint byte count differs from saved identity")
    official = OfficialDiffusionPolicy(source_root, base_checkpoint, device=device,
                                       checkpoint_sha256=identity["sha256"])
    actor = DenoisingPolicy(official, subset=payload["selection"]["subset"],
                            config=SamplerConfig(**payload["sampler_config"]))
    if (asdict(actor.selection) != payload["selection"]
            or _canonical_scheduler_signature(actor.scheduler_signature)
            != _canonical_scheduler_signature(payload["scheduler_signature"])):
        raise ValueError("Selected parameter names/counts or scheduler differ from saved configuration")
    selected = _selected_parameters(actor)
    _validate_state_dict(payload["actor_parameters"], selected, "actor")
    with torch.no_grad():
        for name, parameter in selected.items():
            parameter.copy_(payload["actor_parameters"][name])
    critic = _construct_critic(payload["critic"]["config"], device)
    _validate_state_dict(payload["critic"]["state_dict"], critic.state_dict(), "critic")
    critic.load_state_dict(payload["critic"]["state_dict"], strict=True)
    generator = _local_generator(payload["rng"], device, restore_rng)
    if restore_rng:
        _restore_rng(payload["rng"])
    return LoadedCheckpoint(official, actor, critic, payload, generator)


def restore_optimizers(loaded, actor_optimizer, critic_optimizer):
    """Restore complete states only into matching optimizer types/group order."""
    parameter_sets = {"actor": _selected_parameters(loaded.actor),
                      "critic": dict(loaded.critic.named_parameters())}
    for name, optimizer in (("actor", actor_optimizer), ("critic", critic_optimizer)):
        expected_type = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        if (expected_type != loaded.payload["optimizer_types"][name]
                or _parameter_groups(optimizer, parameter_sets[name])
                != loaded.payload["optimizer_parameter_names"][name]):
            raise ValueError(f"{name} optimizer type or parameter group order differs from checkpoint")
    for name, optimizer in (("actor", actor_optimizer), ("critic", critic_optimizer)):
        optimizer.load_state_dict(loaded.payload["optimizers"][name])


def _validate_sample_payload(value):
    keys = {"schema_version", "kind", "observations", "chain", "timesteps", "old_means", "old_stds",
            "old_log_prob", "targets", "actions", "sampler_config", "scheduler_signature", "mode", "actor_version"}
    if (not isinstance(value, dict) or set(value) != keys or value["schema_version"] != SCHEMA_VERSION
            or value["kind"] != "tdyssey.sampled_chain"):
        raise ValueError("Unsupported sampled-chain schema")
    if (value["mode"] not in {"training", "baseline"} or not isinstance(value["actor_version"], str)
            or not isinstance(value["scheduler_signature"], str) or not value["scheduler_signature"]):
        raise ValueError("Invalid sampled-chain mode/provenance")
    SamplerConfig(**value["sampler_config"])
    chain = value["chain"]
    if not isinstance(chain, torch.Tensor) or chain.ndim != 4 or chain.shape[0] < 1 or chain.shape[1:] != (101, 16, 2):
        raise ValueError("Expected sampled chain [B,101,16,2]")
    batch = chain.shape[0]
    observations = value["observations"]
    if not isinstance(observations, dict) or set(observations) != {"image", "agent_pos"}:
        raise ValueError("Expected the exact official image/agent_pos conditioning")
    shapes = {"chain": (batch, 101, 16, 2), "timesteps": (100,),
              "old_means": (batch, 100, 16, 2), "old_stds": (100,),
              "targets": (batch, 16, 2), "actions": (batch, 8, 2)}
    for name, shape in shapes.items():
        tensor = value[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid sampled-chain tensor: {name}")
    for name, shape in (("image", (batch, 2, 3, 96, 96)), ("agent_pos", (batch, 2, 2))):
        tensor = observations[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid conditioning tensor: {name}")
    if (value["timesteps"].tolist() != list(range(99, -1, -1))
            or observations["image"].min() < 0 or observations["image"].max() > 1
            or not torch.equal(value["actions"], value["targets"][:, 1:9])):
        raise ValueError("Sampled timesteps, RGB range or executed prefix differs from official convention")
    stds = value["old_stds"]
    if value["mode"] == "training":
        log_prob = value["old_log_prob"]
        if (not isinstance(log_prob, torch.Tensor) or log_prob.shape != (batch, 100)
                or not torch.isfinite(log_prob).all() or not (stds > 0).all()):
            raise ValueError("Training samples require finite log densities and positive standard deviations")
    elif value["old_log_prob"] is not None or stds[-1] != 0 or not (stds[:-1] > 0).all():
        raise ValueError("Baseline samples have a deterministic final transition and no joint log densities")


def sample_to_payload(sample):
    """Copy one detached chain/conditioning record to a CPU tensor-only dict."""
    fields = {name: getattr(sample, name) for name in SampledChain.__dataclass_fields__}
    fields["sampler_config"] = asdict(sample.sampler_config)
    value = _tensor_tree({"schema_version": SCHEMA_VERSION, "kind": "tdyssey.sampled_chain", **fields}, "cpu")
    _validate_sample_payload(value)
    return value


def sample_from_payload(payload, device="cpu"):
    """Rebuild fixed rollout tensors; no simulator or policy forward is run."""
    _tensor_tree(payload)
    _validate_sample_payload(payload)
    fields = {name: _tensor_tree(payload[name], device) for name in SampledChain.__dataclass_fields__
              if name != "sampler_config"}
    return SampledChain(**fields, sampler_config=SamplerConfig(**payload["sampler_config"]))
