"""Real, ledger-charged native GPC checks; no work runs on import."""
from copy import deepcopy
import gc
from pathlib import Path
import time


def _same(first, second):
    import numpy as np
    import torch
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and first.dtype == second.dtype and torch.equal(first, second)
    if isinstance(first, np.ndarray): return isinstance(second, np.ndarray) and np.array_equal(first, second)
    if isinstance(first, dict): return first.keys() == second.keys() and all(_same(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)): return type(first) is type(second) and len(first) == len(second) and all(_same(a,b) for a,b in zip(first,second))
    return first == second


def resource_probe(config, output, data_root, physics_root, gpc_root, init_checkpoint):
    """Batch1/2/4 real native residual forward/backward, zero optimizer updates."""
    import numpy as np
    import torch
    from tdyssey.native_rgb_data import NativeRGBDataset
    from tdyssey.native_rgb_training import CostLedger, construct_model, atomic_json, make_generator, supervised_denoising_loss
    output = Path(output)
    dataset = NativeRGBDataset(data_root, "train", cache_bytes=16*1024**2, source_root=physics_root)
    ledger = CostLedger(config["ledger_path"], config["max_updates"], config["max_train_seconds"])
    model = source = moments = None
    rows = []
    with ledger.start("native_full_GPC_resource_probe_zero_updates") as stage:
        try:
            torch.set_num_threads(2)
            model, source = construct_model(gpc_root, init_checkpoint, trainable=True)
            # Account for Adam's two future moment tensors without executing Adam.
            moments = [torch.zeros_like(p) for p in model.parameters() for _ in range(2)]
            generator = make_generator(config["seed"] + 800000, "cuda")
            candidates = (1, 2, 4) if config.get("local_pilot", True) else (config["microbatch"],)
            for batch in candidates:
                stage.heartbeat()
                if ledger.snapshot()["charged_seconds"] >= config["max_train_seconds"]: break
                torch.cuda.reset_peak_memory_stats()
                row = dict(microbatch=batch, optimizer_updates=0, backward_calls=0)
                try:
                    examples = [dataset[i % len(dataset)] for i in range(batch)]
                    inputs = {key: torch.as_tensor(np.stack([v[key] for v in examples]), device="cuda") for key in ("image", "action")}
                    elapsed = 0.
                    for repeat in range(3):
                        model.zero_grad(set_to_none=True)
                        torch.cuda.synchronize(); started = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.float16, enabled=config["amp"]):
                            loss, _ = supervised_denoising_loss(model, inputs, generator=generator)
                        loss.backward(); torch.cuda.synchronize()
                        duration = time.perf_counter() - started
                        if repeat: elapsed += duration
                        row["backward_calls"] += 1
                        stage.heartbeat()
                        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                            raise FloatingPointError("Native full-network gradients are missing or nonfinite")
                    row.update(status="passed", loss=float(loss), samples_per_second=2*batch/elapsed,
                        cuda_allocated=torch.cuda.max_memory_allocated(), cuda_reserved=torch.cuda.max_memory_reserved(),
                        all_parameter_gradients_present=True)
                    if config.get("max_cuda_bytes") and row["cuda_reserved"] > config["max_cuda_bytes"]:
                        row["status"] = "over_memory_ceiling"
                except torch.cuda.OutOfMemoryError as error:
                    row.update(status="oom", error=str(error)[:300])
                finally:
                    model.zero_grad(set_to_none=True)
                    inputs = None
                    gc.collect(); torch.cuda.empty_cache()
                rows.append(row)
                if row["status"] != "passed": break
            valid = [r for r in rows if r["status"] == "passed"]
            if not valid: raise RuntimeError("No native GPC microbatch passed real forward/backward")
            chosen = max(valid, key=lambda r:r["samples_per_second"])["microbatch"]
            result = dict(kind="native-GPC-resource-probe.v1", rows=rows, chosen_microbatch=chosen,
                native_parameters=sum(p.numel() for p in model.parameters()), optimizer_updates=0,
                adam_moments_reserved=True, objective="original_phase_two_residual_MSE", input_size=96)
            atomic_json(output / "resource-probe.json", result)
            return result
        finally:
            model = source = moments = dataset = None
            gc.collect(); torch.cuda.empty_cache()


def run_engineering(config, output, data_root, physics_root, gpc_root, init_checkpoint):
    """Four charged updates: two base, one reference and one restored repeat."""
    from tdyssey.native_rgb_training import train, atomic_json, load_checkpoint_payload
    output = Path(output); engineering = output / "engineering"
    if (engineering / "report.json").exists(): raise FileExistsError("Engineering output already exists")
    probe = resource_probe(config, output, data_root, physics_root, gpc_root, init_checkpoint)
    effective = deepcopy(config); effective["microbatch"] = probe["chosen_microbatch"]
    atomic_json(output / "pilot-effective.json", effective)
    kwargs = dict(physics_source_root=physics_root, execute=True, phase="engineering")
    base = train(effective, engineering / "base", data_root, gpc_root, init_checkpoint, updates=2, **kwargs)
    if base["status"] != "complete": raise RuntimeError("Native base engineering updates did not complete")
    checkpoint = engineering / "base/last.pt"
    reference = train(effective, engineering / "reference", data_root, gpc_root, init_checkpoint,
                      resume=checkpoint, updates=3, **kwargs)
    restored = train(effective, engineering / "restored", data_root, gpc_root, init_checkpoint,
                     resume=checkpoint, updates=3, **kwargs)
    first = load_checkpoint_payload(engineering / "reference/last.pt")
    second = load_checkpoint_payload(engineering / "restored/last.pt")
    equality = {key: _same(first[key], second[key]) for key in ("model","optimizer","scaler","scheduler","rng","order","identity","config")}
    update_check = base["progress"]["parameter_update_report"]
    result = dict(kind="native-GPC-engineering.v1", passed=all(equality.values()) and update_check["all_parameter_tensors_changed"],
        charged_optimizer_updates=4, exact_next_update=equality,
        all_parameters=update_check, native_parameters=base["native_parameter_count"],
        base_updates=base["progress"]["updates"], reference_updates=reference["progress"]["updates"],
        restored_updates=restored["progress"]["updates"], resource_probe=probe,
        effective_microbatch=effective["microbatch"], test_set_inspected=False)
    atomic_json(engineering / "report.json", result)
    if not result["passed"]: raise RuntimeError("Native full-parameter or exact-resume engineering gate failed")
    return result
