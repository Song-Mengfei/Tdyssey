"""One supervised candidate; planning never imports Torch or queries resources."""
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"configs/gpc_supervised_adaptation.json"
BASE_CONFIG_SHA256 = "1c5683baa16daee11ca419772532cdda95595275cca794b977261188d99abe2c"
AUTHORIZATION_01 = "20261026-user-autonomous-gpc-supervised-pilot-01"
AUTHORIZATION_02 = "20261026-user-autonomous-gpc-supervised-pilot-02"
REPORT_SHA = "c7b1e3257468874940ad074d3ffc00c3d3d88f9f6de16531aba6d2c3cc6e3795"
T1_SHA = "800751b7a3e32df4fee5660d1ea22c7ee0fb45b51a7b41632d420189238a7838"


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def repository_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT/path
    for parent in (path, *path.parents):
        if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
            raise ValueError("Artifact aliases are forbidden")
    path = path.resolve()
    if not path.is_relative_to(ROOT/"artifacts"):
        raise ValueError("Adaptation artifacts must stay inside this checkout's artifacts")
    return path


def execution_identity(config_path,cfg,binding=None,*,strict_path=True):
    """Only the two explicitly authorized pilot identities; no new-name factory.

    Planning may read a temporary copy. Production always binds the corresponding
    fixed config path, candidate, authorization and output directory together.
    """
    identities=(
        ("gpc-tl-supervised-one-step-01",AUTHORIZATION_01,
         "artifacts/gpc-supervised-adaptation-pilot-20261025","configs/gpc_supervised_adaptation.json"),
        ("gpc-tl-supervised-one-step-02",AUTHORIZATION_02,
         "artifacts/gpc-supervised-adaptation-pilot02-20261026","configs/gpc_supervised_adaptation_pilot02.json"))
    match=next((row for row in identities if isinstance(cfg,dict) and cfg.get("candidate_id")==row[0]),None)
    if match is None or cfg.get("output_dir")!=match[2]:raise ValueError("Candidate/output is not an explicit pilot identity")
    candidate,authorization,output,relative=match;expected=(ROOT/relative).resolve()
    if strict_path:
        if config_path is None:raise PermissionError("Execution requires its fixed pilot configuration path")
        supplied=Path(config_path)
        if not supplied.is_absolute():supplied=ROOT/supplied
        if any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in (supplied,*supplied.parents)):
            raise PermissionError("Pilot config aliases are forbidden")
        if supplied.resolve()!=expected:raise PermissionError("Pilot configuration path/identity differs")
    if binding is not None:
        if not isinstance(binding,dict) or binding.get("authorization_id")!=authorization:
            raise PermissionError("Pilot candidate and authorization differ")
        for name,value in (("candidate_id",candidate),("output_dir",output)):
            if name in binding and binding[name]!=value:raise PermissionError("Pilot execution binding differs: "+name)
        if "config_path" in binding:
            bound_path=Path(binding["config_path"])
            if not bound_path.is_absolute():bound_path=ROOT/bound_path
            if bound_path.resolve()!=expected:raise PermissionError("Pilot bound configuration path differs")
    return dict(candidate_id=candidate,authorization_id=authorization,output_dir=output,config_path=str(expected))


def read_config(path=CONFIG):
    from .gpc_supervised_adaptation import OBJECTIVE_SPEC, OPTIMIZER_SPEC
    from .gpc_adaptation_reporting import RETENTION_RULES
    path = Path(path).resolve()
    cfg = json.loads(path.read_bytes())
    original=CONFIG.read_bytes()
    if hashlib.sha256(original).hexdigest()!=BASE_CONFIG_SHA256:raise ValueError("Original pilot01 config bytes changed")
    baseline=json.loads(original);science=lambda value:{key:item for key,item in value.items() if key not in ("candidate_id","output_dir")}
    if science(cfg)!=science(baseline):raise ValueError("Pilot science differs from the immutable pilot01 configuration")
    execution_identity(path,cfg,strict_path=False)
    from .gpc_training_schedule import ORDER
    schedule=cfg.get("training_schedule")
    if "training_schedule" in cfg and (not isinstance(schedule,dict) or set(schedule)!={"path","sha256"}
            or not isinstance(schedule["path"],str) or not schedule["path"] or not isinstance(schedule["sha256"],str)
            or len(schedule["sha256"])!=64 or any(c not in "0123456789abcdef" for c in schedule["sha256"])):
        raise ValueError("Explicit frozen schedule path/SHA256 required")
    if (cfg.get("kind") != "tdyssey.gpc_supervised_adaptation.v1" or cfg.get("t1_sha256") != T1_SHA
            or cfg.get("objective") != OBJECTIVE_SPEC or cfg.get("optimizer") != OPTIMIZER_SPEC
            or cfg.get("trainable_prefixes") != ["inner_model."]
            or cfg.get("train") != dict(batch_size=1, max_updates=500, max_examples=500,
                phase_seconds=900, checkpoint_every=50, selected_update=500, seed=5478,
                grad_clip_norm=1., shape_ratio=[1, 1], order=ORDER if schedule else "alternate_T_L_no_shuffle")
            or cfg.get("evaluation") != dict(loss_examples_per_role=16, loss_shape_ratio=[1, 1],
                loss_seed=15478, seeds=[5478, 5479, 5480], modes={"AR": [1, 2, 3, 4], "TF": [2, 4]},
                regression_branches=63, independent_branches=48, sampler_euler_steps=3)
            or cfg.get("limits") != dict(wall_seconds=3600, model_loads=3, active_gpu_models=1,
                eval_batches=32, eval_examples=32, forecasts=3996, euler_steps=11988,
                output_bytes=1073741824, metadata_bytes=16777216, telemetry_bytes=67108864,
                checkpoint_bytes=167772160, cpu_threads=2,
                cuda_allocated_bytes=536870912, cuda_reserved_bytes=1073741824)):
        raise ValueError("Configuration differs from the fixed adaptation protocol")
    if cfg.get("resources") != dict(startup_ram_gib=4, startup_commit_gib=6, startup_gpu_gib=4,
            startup_disk_gib=2, runtime_ram_gib=1.5, runtime_commit_gib=3, runtime_gpu_gib=1.5,
            process_working_set_gib=2, process_private_gib=3, native_job_gib=3):
        raise ValueError("Native single-model resource settings differ")
    if cfg.get("screens") != dict(shape_valid_fraction=.9, pusher_valid_fraction=.95, position_p90_pixels=5,
            angle_p90_degrees=5, pusher_p90_pixels=5, coverage_p90=.05, noncontact_drift_pixels=2,
            rotation_direction_fraction=.8, pair_separation=.05, ranking_accuracy=.75,
            selected_regret=.03, tie_tolerance=1e-6, success_strictly_above=.95):
        raise ValueError("Fixed engineering screens differ")
    if cfg.get("retention") != RETENTION_RULES:
        raise ValueError("Predeclared paired retention rules differ")
    return cfg, sha_file(path)


def read_benchmark(path, data):
    """Metadata only, including held scene identities, never held payloads."""
    path = Path(path).resolve()
    value = json.loads(path.read_bytes())
    if (value.get("kind") != "tdyssey.gpc_adaptation_benchmark.v1"
            or value.get("regression", {}).get("report_sha256") != REPORT_SHA):
        raise ValueError("Benchmark kind/regression identity differs")
    regression = repository_path(value["regression"]["path"])
    if sha_file(regression/"report.json") != REPORT_SHA:
        raise ValueError("Original regression report changed")
    declarations = json.loads(Path(data["path"]).read_bytes())
    excluded = set(declarations["reviewed_scene_sha256"]) | {
        row["initial_state_sha256"] for row in data["episodes"]}
    rows, seen, scenes, payload_scenes = [], set(), {}, {}
    declared_payloads = {Path(row["root"])/item["name"] for row in data["episodes"] for item in row["entry"]["files"]}
    def pose_valid(pose):
        return (isinstance(pose,dict) and isinstance(pose.get("position"),list) and len(pose["position"])==2
            and all(type(n) in (int,float) and math.isfinite(n) for n in [*pose["position"],pose.get("angle")]))
    def truth_valid(truth):
        return (isinstance(truth,dict) and pose_valid(truth.get("pose")) and isinstance(truth.get("pusher"),list)
            and len(truth["pusher"])==2 and all(type(n) in (int,float) and math.isfinite(n) for n in truth["pusher"])
            and type(truth.get("coverage")) in (int,float) and 0<=truth["coverage"]<=1)
    for case in value.get("cases", []):
        anchor, candidate, shape = case.get("anchor"), case.get("candidate"), case.get("shape")
        anchors = [f"I{shape}{n:02d}" for n in range(6)] if shape in ("T", "L") else []
        key, scene = (anchor, candidate), case.get("initial_state_sha256", "")
        if (anchor not in anchors or candidate not in ("hold", "toward", "side", "away") or key in seen
                or case.get("role") != "independent" or case.get("prior_uses") != []
                or len(scene) != 64 or any(c not in "0123456789abcdef" for c in scene)
                or scene in excluded or not isinstance(case.get("truth"), list) or len(case["truth"]) != 4
                or not all(truth_valid(row) for row in case["truth"]) or not truth_valid(case.get("latest"))
                or not pose_valid(case.get("goal")) or type(case.get("contact_by_h4")) is not bool):
            raise ValueError("Independent benchmark identity/labels/scene isolation differs")
        if anchor in scenes and scenes[anchor] != scene:
            raise ValueError("Candidates must share their anchor's initial scene")
        if scene in scenes.values() and scenes.get(anchor) != scene:
            raise ValueError("Distinct independent anchors must have distinct scenes")
        payload = case["payload"]
        source = repository_path(path.parent/payload["path"])
        if source in declared_payloads:
            raise ValueError("Benchmark cannot open any adaptation-source episode payload, including held test")
        if source.suffix != ".npz" or source.is_symlink() or not source.is_file():
            raise ValueError("Missing ordinary independent benchmark NPZ")
        if type(payload.get("bytes")) is not int or source.stat().st_size != payload["bytes"]:
            raise ValueError("Independent benchmark byte identity differs")
        if len(payload.get("sha256", "")) != 64 or any(c not in "0123456789abcdef" for c in payload["sha256"]):
            raise ValueError("Independent benchmark hash is missing")
        if payload["sha256"] in payload_scenes and payload_scenes[payload["sha256"]] != scene:
            raise ValueError("Duplicate image payload cannot represent independent scenes")
        payload_scenes[payload["sha256"]]=scene
        seen.add(key); scenes[anchor] = scene
        rows.append(dict(case, resolved_payload=str(source)))
    wanted = {(f"I{s}{n:02d}", c) for s in ("T", "L") for n in range(6)
              for c in ("hold", "toward", "side", "away")}
    ready = seen == wanted
    return dict(ready=ready, gaps=[] if ready else ["Need 48 new independent branches (12 anchors)"],
        path=str(path), sha256=sha_file(path), regression_path=str(regression), cases=rows)


def plan(config_path=CONFIG, *, claimed=False):
    from .gpc_adaptation_data import read_adaptation_manifest
    cfg, identity = read_config(config_path)
    try:
        data = read_adaptation_manifest(repository_path(cfg["data_manifest"]))
    except FileNotFoundError:
        data = dict(path=str(repository_path(cfg["data_manifest"])), sha256=None, episodes=[],
            readiness=dict(ready=False, gaps=["Missing official adaptation data manifest"]))
    try:
        benchmark = read_benchmark(repository_path(cfg["benchmark_manifest"]), data)
    except FileNotFoundError:
        benchmark = dict(ready=False, gaps=["Missing benchmark manifest or bound regression report"], cases=[])
    gaps = data["readiness"]["gaps"]+benchmark["gaps"]
    training_schedule=None
    if cfg.get("training_schedule"):
        from .gpc_training_schedule import load_schedule
        declared=cfg["training_schedule"];schedule_path=repository_path(declared["path"])
        try:
            value=load_schedule(schedule_path,declared["sha256"],data=data)
            training_schedule=dict(path=str(schedule_path),sha256=declared["sha256"],order_sha256=value["order_sha256"],
                rule_sha256=value["rule_sha256"])
        except FileNotFoundError:gaps.append("Missing frozen train-only schedule")
    for shape, row in data["readiness"].get("shapes", {}).items():
        if row["independent_validation_windows"] < cfg["evaluation"]["loss_examples_per_role"]//2:
            gaps.append(shape+": fewer than eight distinct independent loss-validation windows")
    if not claimed and repository_path(cfg["output_dir"]).exists():
        gaps.append("Fixed candidate directory already exists; no retry/alternate output")
    return dict(kind="tdyssey.gpc_adaptation_plan.v1", config=cfg, config_sha256=identity,
        config_path=str(Path(config_path).absolute()),
        ready=not gaps, gaps=gaps, data=data, benchmark=benchmark,training_schedule=training_schedule, mode="read_only",
        real_operations=0, native_resource_queries=0)


def verify_execution(prepared, *, authorization_id, reviewed_commit, authorized_config_sha256):
    if not prepared["ready"]:
        raise ValueError("Real startup blocked: "+"; ".join(prepared["gaps"]))
    if not authorization_id or not authorization_id.strip() or len(authorization_id)>256 or authorized_config_sha256 != prepared["config_sha256"]:
        raise PermissionError("Explicit execution authority and exact configuration hash are required")
    identity=execution_identity(prepared.get("config_path"),prepared["config"],
        dict(authorization_id=authorization_id))
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()
    if dirty or head != reviewed_commit or len(reviewed_commit) != 40:
        raise PermissionError("A clean exact reviewed source commit is required")
    result=dict(config_sha256=prepared["config_sha256"], source_commit=head,
        data_sha256=prepared["data"]["sha256"], benchmark_sha256=prepared["benchmark"]["sha256"],
        **identity)
    if prepared["config"].get("training_schedule"):
        from .gpc_training_schedule import binding_fields,load_schedule
        declaration=prepared["config"]["training_schedule"]
        value=load_schedule(repository_path(declaration["path"]),declaration["sha256"],data=prepared["data"])
        result.update(binding_fields(value,declaration["sha256"]))
    return result


class OutputWriter:
    """Single-owner bounded chunks; two checkpoint slots reserve atomic overlap."""
    def __init__(self, directory, limits):
        self.directory, self.limits = Path(directory), limits
        self.metadata = 0
        # The finite protocol generates <16MiB of attempt/highwater records;
        # prepay that single-owner ledger in addition to checkpoint overlap.
        self.bytes = 2*limits["checkpoint_bytes"]+limits["metadata_bytes"]

    def write(self, name, payload, *, metadata=True):
        path = self.directory/name
        if not path.resolve().is_relative_to(self.directory.resolve()) or path.exists():
            raise ValueError("Output must be new and inside this candidate")
        size = len(payload)
        if (self.bytes+size > self.limits["output_bytes"]
                or (metadata and self.metadata+size > self.limits["metadata_bytes"])):
            raise OSError("Fixed adaptation output budget exhausted")
        self.bytes += size
        if metadata:
            self.metadata += size
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            written = stream.write(payload)
            if written != size:
                raise OSError("Short immutable output write; no retry")
        return str(path.relative_to(self.directory))

    def json(self, name, value):
        return self.write(name, encoded(value))

    def raw(self, name, array):
        import numpy as np
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return self.write(name, buffer.getvalue(), metadata=False)


def batch(dataset, indices, device):
    import numpy as np
    import torch
    rows = [dataset[i] for i in indices]
    return {key: torch.from_numpy(np.stack([row[key] for row in rows])).to(device) for key in ("image", "action")}


def balanced_indices(dataset, start, count):
    buckets = [dataset.shape_indices(shape) for shape in ("T", "L")]
    if not all(buckets):
        raise ValueError("Both T and L are required by the fixed 1:1 order")
    return [buckets[k % 2][(k//2) % len(buckets[k % 2])] for k in range(start, start+count)]


def training_check(budget, seconds, monotonic=time.monotonic):
    budget.check()
    if (monotonic() >= budget.attempt["started_monotonic"]+seconds
            or budget.clock() >= budget.attempt["started_unix"]+seconds):
        raise TimeoutError("Training phase deadline: no earlier checkpoint may be selected")


def train_candidate(model, optimizer, dataset, *, cfg, budget, store, generator,
                    progress=lambda *args: None, metrics=lambda *args: None, monotonic=time.monotonic,schedule=None):
    """Reserve update #1 before its first forward; no free feasibility update."""
    import torch
    from .gpc_supervised_adaptation import supervised_denoising_loss
    train = cfg["train"]
    device = next(model.parameters()).device
    selected = [p for p in model.parameters() if p.requires_grad]
    try:
        from .gpc_training_schedule import ORDER
        if (train["order"]==ORDER)!=(schedule is not None) or (cfg.get("training_schedule") is not None)!=(schedule is not None):
            raise ValueError("Configured frozen schedule cannot fall back to corpus-prefix sampling")
        if schedule is not None and any(budget.attempt["binding"].get(k)!=v for k,v in schedule.binding.items()):
            raise ValueError("Training account schedule identity differs")
        model.train()
        while budget.snapshot()["completed_updates"] < train["max_updates"]:
            training_check(budget, train["phase_seconds"], monotonic)
            state = budget.snapshot()
            size = min(train["batch_size"], train["max_examples"]-state["spent_train_examples"])
            if size <= 0:
                break
            cursor=state["spent_train_examples"]
            indices=schedule.indices(cursor,size) if schedule is not None else balanced_indices(dataset,cursor,size)
            inputs = batch(dataset, indices, device)
            operation = budget.reserve_update(size,metadata=dict(training_schedule=schedule.checkpoint_state(cursor),
                next_cursor=cursor+size,dataset_indices=indices)) if schedule is not None else budget.reserve_update(size)
            optimizer.zero_grad(set_to_none=True)
            loss, values = supervised_denoising_loss(model, inputs, generator=generator)
            loss.backward()
            grads = [p.grad for p in selected if p.grad is not None]
            if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError("Missing or nonfinite intended parameter gradients")
            norm = torch.nn.utils.clip_grad_norm_(selected, train["grad_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in selected):
                raise FloatingPointError("Nonfinite updated parameter")
            progress("training", state["completed_updates"]+1)
            training_check(budget, train["phase_seconds"], monotonic)
            budget.complete_update(operation)
            step = budget.snapshot()["completed_updates"]
            metrics(step, dict(values, gradient_norm=float(norm), indices=indices))
            if step % train["checkpoint_every"] == 0 or step == train["max_updates"]:
                cursor=budget.snapshot()["spent_train_examples"];extra=dict(next_example=cursor)
                if schedule is not None:extra["training_schedule"]=schedule.checkpoint_state(cursor)
                store.save(model, optimizer, extra_state=extra)
        state = budget.snapshot()
        if state["completed_updates"] != train["selected_update"] or state["checkpoint"] is None:
            raise RuntimeError("Fixed final update not reached; candidate incomplete, no evaluation")
        training_check(budget, train["phase_seconds"], monotonic)
        budget.select_checkpoint(state["checkpoint"]["sha256"])
        training_check(budget, train["phase_seconds"], monotonic)
        result=dict(completed_updates=state["completed_updates"], selected_checkpoint=state["checkpoint"])
        if schedule is not None:result["training_schedule"]=schedule.checkpoint_state(state["spent_train_examples"])
        return result
    except BaseException as error:
        budget.fail("training_failed: "+type(error).__name__+": "+str(error))
        raise


def evaluate_loss(model, dataset, *, cfg, budget, role, progress=lambda *args: None):
    from .gpc_supervised_adaptation import make_generator, make_noise_plan, validation_loss
    device = next(model.parameters()).device
    count = cfg["evaluation"]["loss_examples_per_role"]
    if any(len(dataset.shape_indices(shape)) < count//2 for shape in ("T", "L")):
        raise ValueError("Independent loss evaluation cannot repeat missing validation examples")
    rows = []
    for ordinal, index in enumerate(balanced_indices(dataset, 0, count)):
        operation = budget.reserve_eval(1, metadata=dict(role=role, example=ordinal))
        inputs = batch(dataset, [index], device)
        generator = make_generator(cfg["evaluation"]["loss_seed"]+ordinal, device)
        noise = make_noise_plan(inputs, generator)
        value, details = validation_loss(model, inputs, noise_plan=noise)
        progress("evaluation", budget.snapshot()["completed_updates"])
        budget.complete_eval(operation)
        rows.append(dict(index=index, loss=value, noise=details["noise_generator_after_sha256"]))
    return rows


def rollout_case(case, seed, *, role, sampler, rng, budget, matched, sink=None, measure=None,
                 progress=lambda *args: None):
    """T1 on real RGB only; saved horizon noise matches both weight roles and TF."""
    import numpy as np
    from .wm_compat_transform import transform_t1
    original = rng.capture()
    key = (case["anchor"], case["candidate"], seed)
    pair = matched.setdefault(key, dict(entries={}, exits={}))
    context = np.stack([transform_t1(frame)[0] for frame in case["history"]])
    rows = []
    try:
        rng.seed(seed)
        for mode, horizons in (("AR", (1, 2, 3, 4)), ("TF", (2, 4))):
            for horizon in horizons:
                if role == "frozen" and mode == "AR":
                    pair["entries"][horizon] = rng.capture()
                else:
                    rng.restore(pair["entries"][horizon])
                entry = rng.capture()
                if mode == "TF":
                    real = np.concatenate((case["history"], case["actual"][:horizon-1]))[-4:]
                    given = np.stack([transform_t1(frame)[0] for frame in real])
                else:
                    given = context.copy()
                actions = np.concatenate((case["past_actions"], case["future_actions"][:horizon]))[-4:]
                actions = actions.astype(np.float32)/np.float32(511)*np.float32(2)-np.float32(1)
                operation = budget.reserve_forecast(metadata=dict(role=role, case=list(key), mode=mode, horizon=horizon))
                raw = sampler.sample(given, actions)
                if not isinstance(raw, np.ndarray) or raw.dtype != np.float32 or raw.shape != (3, 96, 96) or not np.isfinite(raw).all():
                    raise ValueError("Sampler must return finite raw float32 CHW")
                exit_state = rng.capture()
                if role == "frozen" and mode == "AR":
                    pair["exits"][horizon] = exit_state
                if entry != pair["entries"][horizon] or exit_state != pair["exits"][horizon]:
                    raise ValueError("Matched inference noise consumption differs")
                name = f"predictions/{role}-{case['anchor']}-{case['candidate']}-{seed}-{mode}-H{horizon}.npy"
                path = sink.raw(name, raw) if sink else None
                row = dict(role=role, status="completed", anchor=case["anchor"], candidate=case["candidate"], shape=case["shape"],
                    stratum=case["role"], seed=seed, mode=mode, horizon=horizon, raw_path=path,
                    raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(), raw_min=float(raw.min()), raw_max=float(raw.max()),
                    input_sha256=hashlib.sha256(given.tobytes()).hexdigest(), action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                    rng_entry=hashlib.sha256(entry).hexdigest(), rng_exit=hashlib.sha256(exit_state).hexdigest(),
                    rgb_mae=float(np.abs(np.clip(raw, 0, 1)-case["actual"][horizon-1]).mean()),
                    measurement=measure(raw, case["shape"], case["goal"]) if measure else None,
                    truth=case["truth"][horizon-1], latest=case.get("latest"), contact_by_h4=case.get("contact_by_h4"))
                rows.append(row)
                if sink:
                    sink.json(name.replace("predictions/", "measurements/").replace(".npy", ".json"), row)
                progress("evaluation", budget.snapshot()["completed_updates"])
                budget.complete_forecast(operation)
                if mode == "AR":
                    context = np.concatenate((context[1:], raw[None]))
        return rows
    except BaseException as error:
        budget.fail("evaluation_failed: "+type(error).__name__+": "+str(error))
        raise
    finally:
        rng.restore(original)


def load_cases(prepared):
    import numpy as np
    from .wm_compat_evaluation import load_saved_inputs
    cases = [dict(vars(branch), role="regression") for branch in
             load_saved_inputs(prepared["benchmark"]["regression_path"])["branches"]]
    for declaration in prepared["benchmark"]["cases"]:
        path = Path(declaration["resolved_payload"])
        if sha_file(path) != declaration["payload"]["sha256"]:
            raise ValueError("Independent case payload hash differs")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in ("history", "actual", "past_actions", "future_actions")}
        for name in ("history", "actual"):
            value = arrays[name]
            if value.shape != (4, 3, 96, 96) or value.dtype != np.float32 or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise ValueError("Independent real RGB must be float32[4,3,96,96] [0,1]")
        for name, length in (("past_actions", 3), ("future_actions", 4)):
            value = arrays[name]
            if value.shape != (length, 2) or not np.isfinite(value).all() or value.min() < 0 or value.max() > 511:
                raise ValueError("Independent absolute controls differ")
        cases.append(dict(declaration, **arrays))
    if len(cases) != 111:
        raise ValueError("Fixed 63 regression plus 48 independent cases are required")
    shared={}
    for case in cases:
        key=(case["role"],case["anchor"])
        value=(case["history"].tobytes(),case["past_actions"].tobytes(),encoded(case["latest"]))
        if key in shared and shared[key] != value:
            raise ValueError("Matched candidates must share exact real history, prior actions and latest state")
        shared[key]=value
    return cases


class NativeSampler:
    """Evaluation-only original sampler, without a second denoiser/optimizer."""
    def __init__(self, model, check):
        from .gpc_supervised_adaptation import make_sampler
        self.model, self.device, self.check = model, next(model.parameters()).device, check
        self.sampler = make_sampler(model)

    def capture(self):
        import torch
        return torch.cuda.get_rng_state(self.device).cpu().numpy().tobytes()

    def restore(self, value):
        import numpy as np
        import torch
        torch.cuda.set_rng_state(torch.from_numpy(np.frombuffer(value, dtype=np.uint8).copy()), self.device)

    def seed(self, value):
        import torch
        with torch.cuda.device(self.device):
            torch.cuda.manual_seed(value)

    def sample(self, context, actions):
        import torch
        with torch.inference_mode():
            output, _ = self.sampler.sample(torch.tensor(context, device=self.device)[None], torch.tensor(actions, device=self.device)[None])
            torch.cuda.synchronize(self.device)
            self.check()
            return output[0].cpu().numpy()


def run_candidate(prepared, binding, *, factory, close_model, make_sampler, writer,
                  progress=lambda *args: None, memory_check=lambda: None, clock=time.time, monotonic=time.monotonic):
    """Train, release, evaluate frozen, release, evaluate selected adapted weights."""
    from .gpc_adaptation_data import GPCAdaptationDataset, fit_train_stats
    from .gpc_adaptation_runtime import AdaptationBudget, CheckpointStore
    from .gpc_supervised_adaptation import make_generator, make_optimizer, select_trainable_parameters
    cfg = prepared["config"]
    directory = writer.directory
    seconds = binding.get("deadline_unix", clock()+cfg["limits"]["wall_seconds"])-clock()
    if seconds <= 0:
        raise TimeoutError("Launch deadline expired before candidate creation")
    budget = AdaptationBudget.create(directory/"ledger", binding, max_updates=cfg["train"]["max_updates"],
        max_eval_batches=cfg["limits"]["eval_batches"], max_batch_size=cfg["train"]["batch_size"],
        max_train_examples=cfg["train"]["max_examples"], max_eval_examples=cfg["limits"]["eval_examples"],
        max_forecasts=cfg["limits"]["forecasts"], max_euler_steps=cfg["limits"]["euler_steps"],
        max_model_loads=cfg["limits"]["model_loads"], deadline_seconds=seconds, clock=clock, monotonic=monotonic)
    run_cfg = deepcopy(cfg)
    run_cfg["train"]["phase_seconds"] -= max(0, clock()-binding.get("started_unix", budget.attempt["started_unix"]))
    model = optimizer = sampler = None
    loss_rows, matched, forecast_rows = {}, {}, []
    def check_progress(phase, step):
        budget.check(); memory_check(); progress(phase, step, budget.snapshot())
    try:
        dataset = GPCAdaptationDataset(prepared["data"]["path"], "train", require_ready=True)
        valid = GPCAdaptationDataset(prepared["data"]["path"], "independent_validation", require_ready=True)
        schedule=None
        if cfg.get("training_schedule"):
            from .gpc_training_schedule import ScheduleConsumer,load_schedule
            declaration=cfg["training_schedule"]
            value=load_schedule(repository_path(declaration["path"]),declaration["sha256"],data=dataset.manifest)
            schedule=ScheduleConsumer(value,declaration["sha256"],dataset)
            if any(binding.get(k)!=v for k,v in schedule.binding.items()):raise ValueError("Worker schedule binding changed")
            writer.json("training-schedule-binding.json",dict(schedule.identity,cursor=0,source_manifest_sha256=value["source_manifest_sha256"]))
        writer.json("training-statistics.json", fit_train_stats(dataset))
        cases = load_cases(prepared)
        training_check(budget, run_cfg["train"]["phase_seconds"], monotonic)
        operation = budget.reserve_model_load("train_init")
        check_progress("model_loading", 0)
        model = factory("train_init")
        memory_check(); budget.complete_model_load(operation)
        writer.json("trainable-parameters.json", select_trainable_parameters(model))
        optimizer = make_optimizer(model, cfg["optimizer"])
        generator = make_generator(cfg["train"]["seed"], next(model.parameters()).device)
        store = CheckpointStore(directory/"adapted-latest.pt", budget, binding,
            max_bytes=cfg["limits"]["checkpoint_bytes"], generators={"train_noise": generator})
        result = train_candidate(model, optimizer, dataset, cfg=run_cfg, budget=budget, store=store, generator=generator,
            progress=check_progress, metrics=lambda step, values: writer.json(f"updates/{step:04d}.json", values), monotonic=monotonic,schedule=schedule)
        check_progress("evaluation", result["completed_updates"])
        model = optimizer = None
        close_model()
        count = 0
        from .wm_compat_engine import compact_measurement
        from .wm_image_measurement_v2 import decode
        for role in ("frozen", "adapted"):
            check_progress("evaluation", result["completed_updates"])
            operation = budget.reserve_model_load(role)
            check_progress("model_loading", result["completed_updates"])
            model = factory(role)
            if role == "adapted":
                store.load_selected_for_evaluation(model)
            model.requires_grad_(False).eval()
            memory_check(); budget.complete_model_load(operation)
            loss_rows[role] = evaluate_loss(model, valid, cfg=cfg, budget=budget, role=role, progress=check_progress)
            sampler = make_sampler(model)
            for case in cases:
                for seed in cfg["evaluation"]["seeds"]:
                    rows = rollout_case(case, seed, role=role, sampler=sampler, rng=sampler,
                        budget=budget, matched=matched, sink=writer,
                        measure=lambda raw, shape, goal: compact_measurement(decode(raw, shape, goal)), progress=check_progress)
                    count += len(rows)
                    forecast_rows.extend(rows)
            sampler = model = None
            close_model()
        if [row["noise"] for row in loss_rows["frozen"]] != [row["noise"] for row in loss_rows["adapted"]]:
            raise ValueError("Frozen/adapted validation noise differs")
        state = budget.check()
        if state["completed_forecasts"] != cfg["limits"]["forecasts"] or state["completed_eval_batches"] != cfg["limits"]["eval_batches"]:
            raise ValueError("Incomplete fixed evaluation")
        from .gpc_adaptation_reporting import summarize
        writer.json("screens.json", summarize(forecast_rows, tuple(cfg["evaluation"]["seeds"]), planned_cases=cases))
        report = dict(kind="tdyssey.gpc_adaptation_result.v1", status="complete", training=result,
            loss_rows=loss_rows, forecast_records=count, budget=state,
            physical_operations=0, policy_operations=0, new_collection=0,
            acceptance="Completion alone is not acceptance; apply predeclared per-stratum gates to immutable measurements")
        writer.json("completion.json", report)
        check_progress("complete", result["completed_updates"])
        return report
    except BaseException as error:
        budget.fail(type(error).__name__+": "+str(error))
        raise
    finally:
        sampler = model = optimizer = None
        close_model()
