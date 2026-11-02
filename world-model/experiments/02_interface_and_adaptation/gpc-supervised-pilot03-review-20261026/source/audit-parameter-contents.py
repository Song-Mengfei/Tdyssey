"""Post-exit pilot03 storage-content audit; stdlib only, never run by import.

Whitelisted pickle globals become metadata tokens, never Torch classes.
Unsupported serialization returns unavailable; there is no runtime fallback.
"""
import argparse
from array import array
from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import pickle
import pickletools
import stat
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT/"artifacts/gpc-supervised-pilot03-review-20261026"
RUN = ROOT/"artifacts/gpc-supervised-adaptation-pilot03-20261026"
PINNED_SHA = "4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892"
PINNED_BYTES = 35043916
CHECKPOINT_CAP = 167772160
METADATA_CAP = 16*1024**2


class Unavailable(Exception):
    pass


def require(condition, reason):
    if not condition:
        raise Unavailable(reason)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def ordinary(path):
    path = Path(path).absolute()
    for parent in (path, *path.parents):
        require(not parent.is_symlink() and not getattr(parent, "is_junction", lambda: False)(),
                "Path alias refused: "+str(parent))
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), "Ordinary saved file required: "+str(path))
    return path, info


def read_saved(path, maximum, expected_sha=None, expected_bytes=None):
    path, before = ordinary(path)
    require(before.st_size <= maximum, "Input exceeds its fixed byte bound")
    if expected_bytes is not None:
        require(before.st_size == expected_bytes, "Declared saved byte length differs")
    with path.open("rb") as stream:
        value = stream.read(maximum+1)
    after = path.stat()
    require(len(value) == before.st_size and len(value) <= maximum
            and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "Saved input changed while being read")
    if expected_sha is not None:
        require(digest(value) == expected_sha, "Saved SHA256 differs from its immutable identity")
    return value


def read_json(path, maximum=METADATA_CAP):
    return json.loads(read_saved(path, maximum))


def post_exit_identity(run):
    """Do this before opening either weight file; no PID/native query is made."""
    require(Path(run).absolute() == RUN, "This audit is restricted to fixed pilot03")
    report = read_json(run/"guardian/stop-record.json", 65536)
    outcome = report.get("status")
    require(outcome in ("complete", "stopped"), "Guardian has no terminal outcome")
    require(report.get("job_active_processes_after_cleanup") == 0
            and report.get("cleanup_errors") == []
            and report.get("single_model_lock_retained") is False
            and type(report.get("worker_exit_code")) is int
            and type(report.get("launcher_exit_code")) is int,
            "Post-exit Job/launcher/worker cleanup is not proven")
    status = report.get("last_status", {})
    if outcome == "complete":
        require(report.get("guard_reason") is None and report["worker_exit_code"] == 0
                and report["launcher_exit_code"] == 0 and status.get("phase") == "complete"
                and status.get("training_complete") is True and status.get("model_instances") == 0,
                "Final complete status/model release missing")
    handoff = read_json(run/"guardian/handoff.json", 65536)
    binding = report.get("binding")
    require(binding == handoff.get("binding") and isinstance(binding, dict), "Guardian/handoff binding differs")
    require(binding.get("candidate_id") == "gpc-tl-supervised-one-step-03"
            and binding.get("authorization_id") == "20261026-user-autonomous-gpc-supervised-pilot-03"
            and binding.get("output_dir") == "artifacts/gpc-supervised-adaptation-pilot03-20261026"
            and binding.get("checkpoint_sha256") == PINNED_SHA
            and binding.get("checkpoint_bytes") == PINNED_BYTES, "Fixed pilot03/pinned identity differs")
    attempt = read_json(run/"ledger/attempt.json")
    require(attempt.get("binding") == binding and attempt.get("binding_sha256") == digest(encoded(binding)),
            "Ledger attempt binding differs")
    journal_bytes = read_saved(run/"ledger/highwater.jsonl", METADATA_CAP)
    previous, pending, latest, selected, first_reason = "0"*64, {}, None, [], None
    counts = Counter()
    rows = journal_bytes.splitlines(keepends=True)
    for sequence, raw in enumerate(rows, 1):
        row = json.loads(raw)
        require(raw == encoded(row) and row.get("sequence") == sequence
                and row.get("previous_sha256") == previous, "Ledger canonical hash chain differs")
        previous = digest(raw)
        kind, data = row.get("kind"), row.get("data", {})
        counts[kind] += 1
        if kind.startswith("reserve_"):
            require(not pending, "Ledger has overlapping reservations")
            pending[sequence] = kind
        elif kind.startswith("complete_"):
            operation = data.get("operation_id")
            expected = "reserve_"+kind[len("complete_"):]
            require(pending.pop(operation, None) == expected, "Ledger completion/reservation differs")
        elif kind == "checkpoint_committed":
            require(not pending and not selected, "Checkpoint boundary has a pending reservation or prior selection")
            latest = data
        elif kind == "checkpoint_selected":
            require(not pending and latest and data.get("sha256") == latest.get("sha256"),
                    "Selected checkpoint is not the sole current boundary")
            selected.append(data["sha256"])
        elif kind == "failed":
            require(outcome == "stopped" and sequence == len(rows) and len(selected) == 1,
                    "Failure was not terminal and after selected training")
            first_reason = data.get("reason")
        else:
            require(kind == "created" and sequence == 1, "Failed or unsupported ledger operation: "+str(kind))
    require(counts["reserve_update"] == counts["complete_update"] == 500
            and len(selected) == 1 and latest and selected[0] == latest["sha256"],
            "Exactly 500 durable updates and one selected checkpoint required")
    require(not pending or (outcome == "stopped" and all(kind in
            ("reserve_forecast", "reserve_eval", "reserve_model_load") for kind in pending.values())),
            "An incomplete training reservation cannot support selected training")
    require(Path(latest.get("path", "")).absolute() == run/"adapted-latest.pt",
            "Selected checkpoint path differs")
    require(latest["boundary"].get("spent_reserved_updates") == 500
            and latest["boundary"].get("spent_train_examples") == 500,
            "Selected checkpoint is not the 500-example boundary")
    if outcome == "complete":
        completion = read_json(run/"completion.json")
        require(completion.get("status") == "complete", "Completion record is not complete")
        final_budget = completion.get("budget", {})
        require(not pending and final_budget.get("journal_sequence") == len(rows)
                and final_budget.get("journal_tip_sha256") == previous,
                "Completion/ledger tip differs")
    else:
        failure = read_json(run/"ledger/first-failure.json")
        require(first_reason is not None and failure.get("reason") == first_reason
                and failure.get("attempt") == attempt, "Stopped run has no matching durable first failure")
        final_budget = failure.get("state", {})
        require(final_budget.get("status") == "stopped", "First-failure state is not stopped")
        durable_pending = final_budget.get("pending")
        require((not pending and durable_pending is None) or (len(pending) == 1
                and isinstance(durable_pending, dict) and durable_pending.get("operation_id") in pending
                and "reserve_"+str(durable_pending.get("kind")) == pending[durable_pending["operation_id"]]),
                "Stopped pending reservation/ledger differs")
    require(final_budget.get("completed_updates") == 500 and final_budget.get("spent_reserved_updates") == 500
            and final_budget.get("spent_train_examples") == 500
            and final_budget.get("selected_checkpoint_sha256") == selected[0]
            and final_budget.get("selection_count") == 1
            and final_budget.get("checkpoint") == latest, "Final durable state/ledger selection differs")
    return dict(report=report, binding=binding, attempt=attempt, selected=latest,
                initial=Path(handoff["checkpoint"]), journal_sha256=digest(journal_bytes),
                journal_tip_sha256=previous, journal_records=len(rows),
                outcome=dict(guardian_status=outcome, guard_reason=report.get("guard_reason"),
                    first_reason=first_reason, evaluation_complete=outcome == "complete",
                    completed_updates=500, completed_eval_batches=final_budget.get("completed_eval_batches"),
                    reserved_forecasts=final_budget.get("spent_reserved_forecasts"),
                    completed_forecasts=final_budget.get("completed_forecasts"),
                    pending=final_budget.get("pending")))


@dataclass(frozen=True)
class StorageType:
    name: str
    code: str
    itemsize: int


@dataclass(frozen=True)
class StorageRef:
    kind: StorageType
    key: str
    location: str
    count: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    offset: int
    shape: tuple
    stride: tuple


STORAGES = {name: StorageType(name, code, size) for name, code, size in (
    ("FloatStorage", "f", 4), ("DoubleStorage", "d", 8), ("ByteStorage", "B", 1),
    ("CharStorage", "b", 1), ("ShortStorage", "h", 2), ("IntStorage", "i", 4),
    ("LongStorage", "q", 8), ("BoolStorage", "B", 1), ("UntypedStorage", "B", 1))}
DTYPES = {name: STORAGES[storage] for name, storage in (
    ("float32", "FloatStorage"), ("float64", "DoubleStorage"), ("uint8", "ByteStorage"),
    ("int8", "CharStorage"), ("int16", "ShortStorage"), ("int32", "IntStorage"),
    ("int64", "LongStorage"), ("bool", "BoolStorage"))}


def rebuild_tensor(storage, offset, shape, stride, requires_grad=False, backward_hooks=None, metadata=None):
    require(isinstance(storage, StorageRef) and type(offset) is int and offset >= 0,
            "Unsupported tensor storage/offset")
    require(isinstance(shape, (tuple, list)) and isinstance(stride, (tuple, list))
            and len(shape) == len(stride) <= 16
            and all(type(n) is int and n >= 0 for n in (*shape, *stride)), "Unsupported tensor shape/stride")
    require(requires_grad in (True, False) and (backward_hooks is None or not backward_hooks)
            and (metadata is None or metadata == {}), "Unsupported tensor rebuild metadata/flags")
    return TensorRef(storage, offset, tuple(shape), tuple(stride))


def rebuild_tensor_v3(storage, offset, shape, stride, requires_grad, backward_hooks, dtype, metadata=None):
    require(isinstance(storage, StorageRef) and storage.kind.name == "UntypedStorage"
            and dtype in DTYPES.values() and storage.count % dtype.itemsize == 0,
            "Unsupported v3 untyped storage/dtype")
    typed = StorageRef(dtype, storage.key, storage.location, storage.count//dtype.itemsize)
    return rebuild_tensor(typed, offset, shape, stride, requires_grad, backward_hooks, metadata)


def rebuild_parameter(tensor, requires_grad, backward_hooks):
    require(isinstance(tensor, TensorRef) and requires_grad in (True, False) and not backward_hooks,
            "Unsupported parameter wrapper")
    return tensor


class RestrictedMetadata(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch" and name in STORAGES:
            return STORAGES[name]
        if module == "torch.storage" and name == "UntypedStorage":
            return STORAGES[name]
        if module == "torch" and name in DTYPES:
            return DTYPES[name]
        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
            return rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_tensor_v3":
            return rebuild_tensor_v3
        if module == "torch._utils" and name == "_rebuild_parameter":
            return rebuild_parameter
        raise Unavailable("Unsupported pickle global (never imported): "+module+"."+name)

    def persistent_load(self, value):
        require(isinstance(value, tuple) and len(value) == 5 and value[0] == "storage",
                "Unsupported persistent storage reference")
        _, kind, key, location, count = value
        require(isinstance(kind, StorageType) and kind in STORAGES.values()
                and isinstance(key, str) and key.isascii() and key.isdecimal()
                and isinstance(location, str) and type(count) is int and 0 <= count <= CHECKPOINT_CAP,
                "Malformed storage token")
        return StorageRef(kind, key, location, count)


class Archive:
    def __init__(self, payload):
        self.zip = zipfile.ZipFile(io.BytesIO(payload))
        infos = self.zip.infolist()
        names = [item.filename for item in infos]
        require(len(set(names)) == len(names) and len(names) <= 10000, "Duplicate/excess ZIP members")
        require(all(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
                    and "\\" not in name for name in names), "Unsafe ZIP member name")
        require(all(item.compress_type == zipfile.ZIP_STORED for item in infos)
                and sum(item.file_size for item in infos) <= CHECKPOINT_CAP, "Unsupported ZIP compression/size")
        pickles = [name for name in names if name.endswith("/data.pkl")]
        require(len(pickles) == 1, "Exactly one PyTorch ZIP data.pkl required")
        self.prefix = pickles[0][:-len("data.pkl")]
        pickle_info = self.zip.getinfo(pickles[0])
        require(pickle_info.file_size <= METADATA_CAP, "Pickle metadata exceeds bound")
        raw = self.zip.read(pickles[0])
        for count, (operation, argument, position) in enumerate(pickletools.genops(raw), 1):
            require(count <= 2000000 and operation.name not in
                    {"EXT1", "EXT2", "EXT4", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "NEXT_BUFFER", "READONLY_BUFFER"},
                    "Unsupported pickle construction opcode")
        stream = io.BytesIO(raw)
        self.value = RestrictedMetadata(stream).load()
        require(stream.tell() == len(raw), "Trailing pickle data refused")
        byteorder = self.prefix+"byteorder"
        self.byteorder = self.zip.read(byteorder).decode("ascii").strip() if byteorder in names else "little"
        self.byteorder_source = "explicit" if byteorder in names else "legacy_PyTorch_default_little"
        require(self.byteorder in ("little", "big"), "Unsupported byte order")

    def values(self, tensor, expected=None):
        require(isinstance(tensor, TensorRef), "Tensor metadata token required")
        storage = tensor.storage
        if expected is not None:
            require(storage.kind.name == expected, "Unsupported tensor dtype: "+storage.kind.name)
        raw = self.zip.read(self.prefix+"data/"+storage.key)
        require(len(raw) == storage.count*storage.kind.itemsize, "Storage byte length differs")
        values = array(storage.kind.code)
        require(values.itemsize == storage.kind.itemsize, "Platform scalar size differs")
        values.frombytes(raw)
        if self.byteorder != sys.byteorder and values.itemsize > 1:
            values.byteswap()
        count = math.prod(tensor.shape)
        require(count <= CHECKPOINT_CAP//storage.kind.itemsize, "Tensor element count exceeds bound")
        maximum = tensor.offset+sum((size-1)*stride for size, stride in zip(tensor.shape, tensor.stride))
        require((count == 0 and tensor.offset <= len(values)) or (count > 0 and maximum < len(values)),
                "Tensor view exceeds storage")
        contiguous, expected_stride = True, 1
        for size, stride in zip(reversed(tensor.shape), reversed(tensor.stride)):
            if size > 1 and stride != expected_stride:
                contiguous = False
            expected_stride *= size
        if contiguous:
            yield from itertools.islice(values, tensor.offset, tensor.offset+count)
        else:
            for index in itertools.product(*(range(size) for size in tensor.shape)):
                yield values[tensor.offset+sum(i*stride for i, stride in zip(index, tensor.stride))]

    def rng_identity(self, tensor):
        require(isinstance(tensor, TensorRef), "Saved RNG tensor is not metadata")
        raw = bytes(self.values(tensor, "ByteStorage"))
        return dict(dtype="uint8", shape=list(tensor.shape), elements=len(raw), sha256=digest(raw),
                    saved_location=tensor.storage.location)


def check_checkpoint_metadata(checkpoint, identity):
    require(isinstance(checkpoint, dict) and checkpoint.get("kind") == "gpc_adaptation_checkpoint.v1",
            "Unsupported checkpoint payload")
    require(checkpoint.get("logical_steps") == 500 and checkpoint.get("config_identity") == identity["binding"]
            and checkpoint.get("attempt") == identity["attempt"]
            and checkpoint.get("budget") == identity["selected"]["boundary"]
            and checkpoint["budget"].get("budget_id") == identity["attempt"]["budget_id"]
            and checkpoint["budget"].get("binding_sha256") == identity["attempt"]["binding_sha256"],
            "Checkpoint/ledger metadata differs")
    binding = identity["binding"]
    schedule = dict(sha256=binding["training_schedule_sha256"], order_sha256=binding["training_schedule_order_sha256"],
                    rule_sha256=binding["training_schedule_rule_sha256"], cursor=500)
    require(checkpoint.get("extra_state") == dict(next_example=500, training_schedule=schedule)
            and checkpoint["budget"].get("training_schedule") == schedule,
            "Selected schedule identity/cursor differs")
    return dict(logical_steps=500, config_identity=checkpoint["config_identity"],
                budget=checkpoint["budget"], extra_state=checkpoint["extra_state"])


def compare_parameters(initial, adapted, trainable):
    old, new = initial.value, adapted.value["model"]
    require(isinstance(old, dict) and isinstance(new, dict) and set(old) == set(new), "Model state names differ")
    selected = trainable["trainable"]
    require(len({row["name"] for row in selected}) == len(selected)
            and sum(row["numel"] for row in selected) == trainable["trainable_numel"], "Trainable declaration differs")
    rows = []
    for declared in selected:
        name = declared["name"]
        require(name in old and isinstance(old[name], TensorRef) and isinstance(new[name], TensorRef),
                "Missing selected tensor: "+name)
        require(old[name].shape == new[name].shape == tuple(declared["shape"])
                and math.prod(old[name].shape) == declared["numel"], "Selected tensor shape differs: "+name)
        changed, finite, squared, maximum = 0, True, 0., 0.
        old_values, new_values = initial.values(old[name], "FloatStorage"), adapted.values(new[name], "FloatStorage")
        elements = 0
        for before, after in zip(old_values, new_values):
            elements += 1
            finite = finite and math.isfinite(before) and math.isfinite(after)
            if before != after:
                changed += 1
            if math.isfinite(before) and math.isfinite(after):
                delta = after-before
                squared += delta*delta
                maximum = max(maximum, abs(delta))
        require(elements == declared["numel"], "Selected tensor value count differs")
        rows.append(dict(name=name, elements=elements, changed_elements=changed, finite=finite,
                         l2=math.sqrt(squared) if finite else None, max_abs=maximum if finite else None))
    finite = all(row["finite"] for row in rows)
    return dict(compared_tensors=len(rows), compared_elements=sum(row["elements"] for row in rows),
        changed_tensors=sum(row["changed_elements"] > 0 for row in rows),
        changed_elements=sum(row["changed_elements"] for row in rows), finite=finite,
        l2=math.sqrt(math.fsum(row["l2"]**2 for row in rows)) if finite else None,
        max_abs=max((row["max_abs"] for row in rows), default=None) if finite else None,
        comparison="float32 numerical values in logical tensor order; +/-zero count as equal", tensors=rows)


def optimizer_and_rng(archive, trainable):
    checkpoint = archive.value
    signature, optimizer = checkpoint["optimizer_signature"], checkpoint["optimizer"]
    groups, state = optimizer["param_groups"], optimizer["state"]
    require(isinstance(signature, dict) and signature.get("kind") == "torch.optim.adamw.AdamW"
            and len(groups) == len(signature["parameter_groups"]), "Unsupported optimizer signature")
    rows, names = [], []
    for group, declared_names in zip(groups, signature["parameter_groups"]):
        require(len(group["params"]) == len(declared_names), "Optimizer names/parameter IDs differ")
        for parameter, name in zip(group["params"], declared_names):
            names.append(name)
            saved = state.get(parameter)
            if saved is None:
                rows.append(dict(name=name, participating=False, step=None))
                continue
            step = saved.get("step")
            values = list(archive.values(step, "FloatStorage")) if isinstance(step, TensorRef) else [step]
            require(len(values) == 1 and type(values[0]) in (int, float) and math.isfinite(values[0]),
                    "Unsupported/nonfinite optimizer step")
            rows.append(dict(name=name, participating=True, step=values[0]))
    require(len(names) == len(set(names)) and set(names) == {row["name"] for row in trainable["trainable"]},
            "Optimizer trainable-name ownership differs")
    saved_rng = checkpoint["rng"]
    rng = dict(python_state_sha256=digest(encoded(saved_rng["python"])),
        numpy_state_sha256=digest(encoded(saved_rng["numpy"])),
        torch_cpu=archive.rng_identity(saved_rng["torch_cpu"]),
        torch_cuda=[archive.rng_identity(value) for value in saved_rng["torch_cuda"]], generators={})
    for name, value in saved_rng["generators"].items():
        rng["generators"][name] = dict(device=value["device"], state=archive.rng_identity(value["state"]))
    participating = [row["step"] for row in rows if row["participating"]]
    return dict(signature=signature, participating_tensors=len(participating), missing_state_tensors=len(rows)-len(participating),
        step_counts={str(key): count for key, count in Counter(participating).items()},
        all_participating_steps_equal_500=bool(participating) and all(step == 500 for step in participating),
        parameters=rows), rng


def audit(run):
    identity = post_exit_identity(run)
    # Only after every post-exit/ledger precondition has passed are weights read.
    initial_bytes = read_saved(identity["initial"], PINNED_BYTES, PINNED_SHA, PINNED_BYTES)
    adapted_bytes = read_saved(run/"adapted-latest.pt", CHECKPOINT_CAP, identity["selected"]["sha256"])
    initial, adapted = Archive(initial_bytes), Archive(adapted_bytes)
    metadata = check_checkpoint_metadata(adapted.value, identity)
    trainable = read_json(run/"trainable-parameters.json")
    comparison = compare_parameters(initial, adapted, trainable)
    optimizer, rng = optimizer_and_rng(adapted, trainable)
    return dict(status="available", kind="gpc_pilot03_parameter_content_audit.v1",
        formal_run_outcome=identity["outcome"],
        input_identity=dict(initial_sha256=PINNED_SHA, initial_bytes=len(initial_bytes),
            selected_sha256=identity["selected"]["sha256"], selected_bytes=len(adapted_bytes),
            journal_sha256=identity["journal_sha256"], journal_tip_sha256=identity["journal_tip_sha256"],
            journal_records=identity["journal_records"]), checkpoint=metadata, parameters=comparison,
        optimizer=optimizer, rng=rng, serialization=dict(initial_byteorder_source=initial.byteorder_source,
            adapted_byteorder_source=adapted.byteorder_source),
        limitations=["Content changes do not establish validation or physical improvement.",
            "No tensor/model/RNG was instantiated or restored; unsupported formats have no fallback."])


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=REVIEW/"parameter-contents.json")
    args = parser.parse_args()
    output = args.output.absolute()
    require(output.parent == REVIEW and not output.exists(), "Audit output must be new in the review namespace")
    try:
        report = audit(args.run_dir.absolute())
    except Exception as error:
        report = dict(status="unavailable", kind="gpc_pilot03_parameter_content_audit.v1",
                      reason=type(error).__name__+": "+str(error), no_runtime_fallback=True)
    payload = encoded(report)
    require(len(payload) <= 1024**2, "Audit report exceeds 1MiB")
    with output.open("xb") as stream:
        stream.write(payload)
    print(json.dumps(dict(status=report["status"], output=str(output), bytes=len(payload), sha256=digest(payload))))
    return 0 if report["status"] == "available" else 2


if __name__ == "__main__":
    raise SystemExit(main())
