"""CPU weights_only inspection of the two small C deltas; no base model load."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import torch

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
config = json.loads((ROOT / "configs/policy_c.example.json").read_text())
spec = json.loads((ROOT / "manifests/official_dp.json").read_text())
config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def primitive(value):
    if isinstance(value, torch.Tensor):
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(memoryview(raw)).hexdigest()}
    if isinstance(value, dict):
        return {str(k): primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [primitive(v) for v in value]
    if type(value) is float:
        assert math.isfinite(value)
    return value


def verify_tree(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
        if value.is_floating_point() or value.is_complex():
            assert bool(torch.isfinite(value).all()), "Nonfinite checkpoint tensor"
        return 1, value.numel() * value.element_size()
    children = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else ()
    sums = [verify_tree(child) for child in children]
    return sum(x[0] for x in sums), sum(x[1] for x in sums)


output = {"at_utc": datetime.now(timezone.utc).isoformat(), "checkpoints": [],
          "scope": "CPU weights_only read of small incremental files only; no base reconstruction, forward, sampler, optimizer or ledger owner"}
for path in sorted((ROOT / "artifacts/milestone-c-20261024").glob("checkpoint-*.pt")):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["kind"] == "tdyssey.policy_update_delta" and payload["schema_version"] == 1
    metadata = payload["metadata"]
    assert metadata["kind"] == "tdyssey-C" and metadata["experiment_id"] == "c-tl-2048-20261024"
    assert metadata["config"] == config and metadata["config_hash"] == config_hash
    assert payload["base_identity"] == config["base"]
    assert payload["official"] == {name: spec[name] for name in ("source_commit", "source_sha256", "policy_config")}
    actor, critic = payload["actor_parameters"], payload["critic"]["state_dict"]
    assert len(actor) == 6 and sum(t.numel() for t in actor.values()) == 1313282
    assert len(critic) == 4 and sum(t.numel() for t in critic.values()) == 17153
    assert set(actor) == set(payload["selection"]["names"]) and payload["selection"]["subset"] == "output"
    assert all(name.startswith("model.final_conv.") for name in actor)
    step_counts = {}
    for owner, parameters in (("actor", actor), ("critic", critic)):
        saved = payload["optimizers"][owner]
        assert len(saved["state"]) == len(parameters)
        steps = {int(state["step"].item()) for state in saved["state"].values()}
        expected = metadata["updater"][owner + "_steps"]
        assert steps == {expected}
        step_counts[owner] = expected
        expected_lr = config["optimization"][owner + "_lr"]
        assert all(group["lr"] == expected_lr and group["weight_decay"] == 0 for group in saved["param_groups"])
        names = [name for group in payload["optimizer_parameter_names"][owner] for name in group]
        assert set(names) == set(parameters) and len(names) == len(parameters)
    tensors, tensor_bytes = verify_tree(payload)
    output["checkpoints"].append({"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size,
        "metadata": primitive(metadata), "base_identity": payload["base_identity"],
        "optimizer_steps": step_counts, "all_tensors_finite": True, "all_tensors_on_cpu": True,
        "tensor_count": tensors, "tensor_bytes": tensor_bytes,
        "selected_actor_parameters": 1313282, "critic_parameters": 17153})
    del payload, actor, critic, metadata, saved, parameters
assert len(output["checkpoints"]) == 2
output["cuda_initialized"] = torch.cuda.is_initialized()
assert output["cuda_initialized"] is False
destination = HERE / "checkpoint-metadata-verified.json"
with destination.open("x", encoding="utf-8") as target:
    json.dump(output, target, sort_keys=True, indent=2, allow_nan=False)
    target.write("\n")
    target.flush()
    os.fsync(target.fileno())
print(json.dumps({"output": str(destination), "cuda_initialized": False,
    "checkpoints": [{"path": item["path"], "steps": item["optimizer_steps"],
                     "resumable_at_save": item["metadata"]["resumable"], "all_tensors_finite": True}
                    for item in output["checkpoints"]]}, indent=2))
