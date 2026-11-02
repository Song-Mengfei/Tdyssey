"""Server-only stopping signals around the unchanged native RGB trainer.

This module observes H1 validation denoising losses and the same seed's durable
cost ledger. It never loads a model, changes checkpoint selection or parameters,
reads unseen outputs, removes STOP files, or terminates processes.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import uuid

POLL_SECONDS = 1.
CHECKPOINT_RESERVE_SECONDS = 50.


def _settings(config):
    patience = config.get("early_stopping_patience", 8)
    delta = config.get("early_stopping_min_delta", 1e-6)
    if type(patience) is not int or patience < 1:
        raise ValueError("early_stopping_patience must be a positive integer")
    if isinstance(delta, bool) or not math.isfinite(float(delta)) or float(delta) < 0:
        raise ValueError("early_stopping_min_delta must be finite and nonnegative")
    maximum = config.get("max_train_seconds")
    if maximum is not None and (isinstance(maximum, bool) or not math.isfinite(float(maximum)) or float(maximum) <= 0):
        raise ValueError("max_train_seconds must be finite and positive")
    return patience, float(delta), None if maximum is None else float(maximum)


def inspect_decision(config, validation_rows, ledger=None):
    """Pure observation of unique validation checkpoints and the time budget.

    `best_denoising_loss` is the true observed minimum. The separate patience
    reference changes only after an improvement larger than min_delta. Repeated
    copies of a checkpoint count once; the last recorded value for its update is
    used. Malformed observations disable patience, while a valid ledger can still
    request the fixed checkpoint time reserve.
    """
    if config.get("server") is not True:
        return dict(enabled=False, should_stop=False, reason=None)
    patience, delta, maximum = _settings(config)
    warnings, unique, valid_rows = [], {}, 0
    if not isinstance(validation_rows, list):
        warnings.append("validation_not_a_list")
        validation_rows = []
    for row in validation_rows:
        if (not isinstance(row, dict) or type(row.get("update")) is not int or row["update"] < 0
                or isinstance(row.get("denoising_loss"), bool)
                or not isinstance(row.get("denoising_loss"), (int, float))
                or not math.isfinite(row["denoising_loss"]) or row["denoising_loss"] < 0):
            warnings.append("invalid_validation_checkpoint")
            continue
        unique[row["update"]] = float(row["denoising_loss"])
        valid_rows += 1
    observations = [dict(update=update, denoising_loss=loss) for update, loss in sorted(unique.items())]
    best = reference = best_update = None
    no_improvement = 0
    for row in observations:
        loss = row["denoising_loss"]
        if best is None or loss < best:
            best, best_update = loss, row["update"]
        if reference is None or loss < reference-delta:
            reference, no_improvement = loss, 0
        else:
            no_improvement += 1
    charged = None
    ledger_valid = False
    if ledger is not None:
        if isinstance(ledger, dict):
            observed = ledger.get("charged_seconds")
            bound = ledger.get("max_seconds")
            if (not isinstance(observed, bool) and isinstance(observed, (int, float))
                    and math.isfinite(observed) and observed >= 0):
                charged = float(observed)
                if maximum is None and not isinstance(bound, bool) and isinstance(bound, (int, float)) and math.isfinite(bound) and bound > 0:
                    maximum = float(bound)
                ledger_valid = maximum is not None
        if not ledger_valid:
            warnings.append("invalid_cost_ledger")
    time_guard = ledger_valid and charged >= max(0., maximum-CHECKPOINT_RESERVE_SECONDS)
    patience_guard = not warnings and bool(observations) and no_improvement >= patience
    reason = "checkpoint_time_reserve" if time_guard else "validation_patience" if patience_guard else None
    return dict(enabled=True, should_stop=reason is not None, reason=reason,
        best_denoising_loss=best, best_update=best_update, patience_reference_loss=reference,
        consecutive_no_improvement=no_improvement, unique_validation_checkpoints=len(observations),
        observations=observations, duplicate_checkpoint_rows=valid_rows-len(unique),
        invalid_checkpoint_rows=len(validation_rows)-valid_rows,
        charged_seconds=charged, max_train_seconds=maximum, checkpoint_reserve_seconds=CHECKPOINT_RESERVE_SECONDS,
        criterion=dict(validation_field="denoising_loss", validation_split="seen_val", patience=patience,
                       min_delta=delta, improvement="loss < patience_reference_loss - min_delta"),
        warnings=sorted(set(warnings)), unseen_outputs_read=False,
        native_loss_selector_and_parameters_unchanged=True)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")


def _publish_once(path, value):
    """Publish complete bytes atomically without replacing any existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if os.name == "nt":
                # Windows rename is atomic and refuses an existing destination;
                # unlike POSIX rename, it cannot overwrite another STOP.
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path):
    """Read one inode's bytes; a concurrent atomic replacement is harmless."""
    try:
        payload = Path(path).read_bytes()
    except FileNotFoundError:
        return None, None
    return json.loads(payload.decode("utf-8-sig")), hashlib.sha256(payload).hexdigest()


@dataclass
class MonitorState:
    enabled: bool
    observation: dict | None = None
    evidence: dict | None = None
    last_error: str | None = None
    thread: threading.Thread | None = None
    done: threading.Event = field(default_factory=threading.Event)


@contextmanager
def monitor_training(config, training_dir, *, artifact_root=None):
    """Observe one server seed, signal STOP, and join the monitor within5seconds.

    Local runs yield a disabled state without reading or creating any files.
    Complete last.pt writing remains the original trainer's responsibility.
    """
    state = MonitorState(enabled=config.get("server") is True)
    if not state.enabled:
        try:
            yield state
        finally:
            state.done.set()
        return
    config = dict(config)  # Freeze the scalar monitoring contract for this context.
    _settings(config)
    training = Path(training_dir).resolve()
    artifact = Path(artifact_root or training.parent).resolve()
    if not training.is_relative_to(artifact):
        raise ValueError("Training directory must belong to this seed's artifact root")
    ledger_path = artifact/"native-rgb-cost-ledger.json"
    if config.get("ledger_path") is not None and Path(config["ledger_path"]).resolve() != ledger_path:
        raise ValueError("Run control cannot observe another seed's cost ledger")
    validation_path = training/"validation.json"
    evidence_path, stop_path = artifact/"early-stopping.json", artifact/"STOP"
    source = Path(__file__).resolve()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    shutdown = threading.Event()

    def observe():
        try:
            if evidence_path.exists():
                state.last_error = "Existing immutable early-stopping evidence was preserved"
                return
            while not shutdown.is_set():
                rows, validation_sha, ledger, ledger_sha, errors = [], None, None, None, {}
                try:
                    loaded, validation_sha = _read_json(validation_path)
                    rows = [] if loaded is None else loaded
                except (OSError, ValueError) as error:
                    errors["validation"] = type(error).__name__+": "+str(error)
                try:
                    ledger, ledger_sha = _read_json(ledger_path)
                except (OSError, ValueError) as error:
                    errors["ledger"] = type(error).__name__+": "+str(error)
                decision = inspect_decision(config, rows, ledger)
                state.observation = decision
                if errors:
                    state.last_error = json.dumps(errors, sort_keys=True)
                if decision["should_stop"]:
                    evidence = dict(kind="native-gpc-server-run-control.v1", source=str(source),
                        source_sha256=source_sha, decision=decision, read_errors=errors,
                        validation=dict(path=str(validation_path), sha256=validation_sha),
                        ledger=dict(path=str(ledger_path), sha256=ledger_sha),
                        action="write_this_seed_STOP_only", stop_path=str(stop_path),
                        process_termination=False, checkpoint_budget_extended=False)
                    created = _publish_once(stop_path, evidence)
                    evidence["stop_created"] = created
                    evidence["preexisting_stop_preserved"] = not created
                    _publish_once(evidence_path, evidence)
                    state.evidence = evidence
                    return
                shutdown.wait(POLL_SECONDS)
        except BaseException as error:
            state.last_error = type(error).__name__+": "+str(error)
        finally:
            state.done.set()

    state.thread = threading.Thread(target=observe, name="native-rgb-server-run-control", daemon=True)
    state.thread.start()
    primary = None
    try:
        yield state
    except BaseException as error:
        primary = error
        raise
    finally:
        shutdown.set()
        state.thread.join(timeout=5)
        if state.thread.is_alive():
            if primary is None:
                raise RuntimeError("Native RGB run-control monitor did not stop within5seconds")
            primary.add_note("Native RGB run-control monitor did not stop within5seconds")
