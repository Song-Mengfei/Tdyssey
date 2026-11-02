"""Manual prediction/physics comparison. No training, dataset, or experiment resume."""

from collections import deque
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from .envs.official import make_official_shape_env_class, official_shape
from .evaluation import goal_metrics
from .geometry import Pose
from .state import BodyState, Snapshot
from .wm_compat_transform import T1_SHA256, transform_t1

ROOT = Path(__file__).resolve().parents[1]
LEGACY_DOMAIN = "legacy-t1"
OFFICIAL_DOMAIN = "official-original-color"


def display_observation(frame, image_domain=LEGACY_DOMAIN):
    if image_domain == OFFICIAL_DOMAIN:
        return frame.copy()
    if image_domain != LEGACY_DOMAIN:
        raise ValueError("Unknown manual image domain")
    return transform_t1(frame)[0]


def _case_view_keys(metadata):
    domain = metadata.get("image_domain", LEGACY_DOMAIN)
    if domain == OFFICIAL_DOMAIN:
        if metadata.get("kind") != "manual_world_model_case.v2" or metadata.get("t1_sha256") is not None:
            raise ValueError("Official case must declare identity RGB, without T1")
        return "history_model", "actual_model", "model"
    if domain != LEGACY_DOMAIN or metadata.get("kind") != "manual_world_model_case.v1" or metadata.get("t1_sha256") != T1_SHA256:
        raise ValueError("Legacy case T1 identity differs")
    return "history_t1", "actual_t1", "t1"


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _snapshot(value):
    """Require the original complete physical/controller Snapshot schema."""
    if hasattr(value, "to_json"):
        value = json.loads(value.to_json())
    return json.loads(Snapshot.from_dict(value).to_json())


def _request_key(request):
    fields = {key: request[key] for key in
              ("shape", "horizon", "noise_seed", "initial_state", "history_states", "history_steps")}
    fields["target"] = np.asarray(request["target"], dtype=np.float64).tolist()
    if "image_domain" in request:
        fields["image_domain"] = request["image_domain"]
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True, allow_nan=False).encode())
    for key in ("images", "past_actions"):
        array = np.asarray(request[key])
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_request(request):
    domain = request.get("image_domain", LEGACY_DOMAIN)
    if domain not in (LEGACY_DOMAIN, OFFICIAL_DOMAIN) or (domain == OFFICIAL_DOMAIN and request["horizon"] != 1):
        raise ValueError("Official original-color baseline supports H1 only")
    if request["role"] not in ("frozen", "adapted") or request["shape"] not in ("T", "L"):
        raise ValueError("Unknown manual request shape/checkpoint")
    if type(request["horizon"]) is not int or request["horizon"] not in (1, 4) or type(request["noise_seed"]) is not int or request["noise_seed"] < 0:
        raise ValueError("Invalid fixed manual horizon/noise seed")
    for key, shape in (("images", (4, 3, 96, 96)), ("past_actions", (3, 2))):
        array = np.asarray(request[key])
        maximum = 1 if key == "images" else 511
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all() or (array < 0).any() or (array > maximum).any():
            raise ValueError("Malformed manual request " + key)
    target = np.asarray(request["target"])
    if target.shape != (2,) or not np.isfinite(target).all() or (target < 0).any() or (target > 511).any():
        raise ValueError("Malformed manual request target")
    initial = _snapshot(request["initial_state"])
    states = [_snapshot(state) for state in request["history_states"]]
    steps = request["history_steps"]
    if len(states) != 4 or len(steps) != 3 or initial != states[-1] or initial["shape"] != request["shape"]:
        raise ValueError("Manual request needs four full aligned history states")
    if request.get("goal") != Snapshot.from_dict(initial).goal:
        raise ValueError("Manual request goal differs from captured state")
    for index, step in enumerate(steps):
        if step["before_state"] != states[index] or step["after_state"] != states[index+1]:
            raise ValueError("Manual history state/control alignment differs")
        if not np.array_equal(np.asarray(step["action"], dtype=np.float32), request["past_actions"][index]):
            raise ValueError("Manual past action history differs")
        if states[index+1]["control_step"] != states[index]["control_step"]+1:
            raise ValueError("Manual history control steps are discontinuous")
    if request.get("request_key") != _request_key(request):
        raise ValueError("Manual request identity differs")


def request_from_case(bundle, role=None, *, _validate=True):
    """Recreate fixed model inputs only; never restores/steps an environment."""
    metadata, arrays = bundle["metadata"], bundle["arrays"]
    request = deepcopy(metadata["request"])
    request["role"] = metadata.get("latest_accepted_role", request["role"]) if role is None else role
    request["images"] = arrays["history_raw"].copy()
    request["past_actions"] = arrays["past_actions"].copy()
    request["target"] = np.asarray(request["target"], dtype=np.float64)
    request["goal"] = Snapshot.from_dict(request["initial_state"]).goal
    request["offline_fixed_request"] = True
    if _validate:
        _validate_request(request)
    return request


def _validate_case(bundle):
    metadata, arrays = bundle["metadata"], bundle["arrays"]
    if metadata.get("kind") not in ("manual_world_model_case.v1", "manual_world_model_case.v2") or metadata.get("training_data") is not False:
        raise ValueError("Not a manual exploration case")
    history_key, actual_key, _ = _case_view_keys(metadata)
    if not metadata["predictions"]:
        raise ValueError("Manual case lacks an accepted prediction")
    request = request_from_case(bundle, _validate=False)
    _validate_request(request)
    if request.get("image_domain", LEGACY_DOMAIN) != metadata.get("image_domain", LEGACY_DOMAIN):
        raise ValueError("Case/request image domains differ")
    if metadata.get("requested_actions") != [request["target"].tolist() for _ in range(request["horizon"])]:
        raise ValueError("Declared future actions differ from the fixed request")
    expected = {"history_raw", history_key, "past_actions"}
    if arrays[history_key].shape != (4, 3, 96, 96):
        raise ValueError("Malformed transformed real history")
    for index, prediction in enumerate(metadata["predictions"]):
        name = prediction["array"]
        if name != f"prediction_{index:03d}" or prediction["role"] not in ("frozen", "adapted") or arrays[name].shape != (request["horizon"], 3, 96, 96):
            raise ValueError("Malformed captured prediction")
        expected.add(name)
    execution = metadata.get("execution")
    if execution is not None:
        prediction_index = execution["prediction_index"]
        if type(prediction_index) is not int or not 0 <= prediction_index < len(metadata["predictions"]):
            raise ValueError("Executed prediction reference differs")
        if execution["role"] != metadata["predictions"][prediction_index]["role"]:
            raise ValueError("Executed checkpoint role differs")
        count = len(execution["steps"])
        if not 0 <= count <= request["horizon"]:
            raise ValueError("Malformed executed prefix")
        previous = request["initial_state"]
        for index, step in enumerate(execution["steps"]):
            if _snapshot(step["before_state"]) != previous:
                raise ValueError("Executed state prefix is discontinuous")
            after = _snapshot(step["after_state"])
            if after["control_step"] != previous["control_step"]+1 or step["control_step"] != after["control_step"]:
                raise ValueError("Executed control-step count differs")
            if step["done"] and index != count-1:
                raise ValueError("Controls were recorded after physical termination")
            previous = after
            if not np.array_equal(np.asarray(step["action"], dtype=np.float64), request["target"]):
                raise ValueError("Executed action differs from captured target")
        for name in ("actual_raw", actual_key):
            if arrays[name].shape != (count, 3, 96, 96):
                raise ValueError("Actual frames differ from executed controls")
            expected.add(name)
    if set(arrays) != expected:
        raise ValueError("Unknown or missing manual case arrays")
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Manual case arrays must be finite float32")
        if name in ("history_raw", history_key, "actual_raw", actual_key) and ((value < 0).any() or (value > 1).any()):
            raise ValueError("Real observation arrays outside [0,1]")
    return bundle


def save_case_bundle(path, bundle):
    """Write a new JSON+NPZ bundle; no inference, physics or overwrite."""
    _validate_case(bundle)
    arrays = {name: value.copy() for name, value in bundle["arrays"].items()}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    metadata = deepcopy(bundle["metadata"])
    metadata["payload"] = dict(path="arrays.npz", bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    metadata["array_identities"] = {name: dict(shape=list(value.shape), dtype=value.dtype.str,
        sha256=hashlib.sha256(value.tobytes()).hexdigest()) for name, value in arrays.items()}
    encoded = (json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    with (path / "arrays.npz").open("xb") as stream:
        if stream.write(payload) != len(payload): raise OSError("Short manual case payload write")
    with (path / "case.json").open("xb") as stream:
        if stream.write(encoded) != len(encoded): raise OSError("Short manual case manifest write")
    return dict(path=str(path), manifest="case.json", bytes=len(encoded)+len(payload),
        manifest_sha256=hashlib.sha256(encoded).hexdigest(), request_key=metadata["request"]["request_key"],
        comparison_roles=sorted({prediction["role"] for prediction in metadata["predictions"]}),
        executed_controls=len(metadata["execution"]["steps"]) if metadata["execution"] is not None else 0)


def load_case_bundle(path):
    """Verify a saved bundle and expose copied arrays; does not execute replay."""
    path = Path(path)
    metadata = json.loads((path / "case.json").read_bytes())
    declaration = metadata["payload"]
    if declaration["path"] != "arrays.npz": raise ValueError("Unknown manual case payload path")
    payload = (path / "arrays.npz").read_bytes()
    if len(payload) != declaration["bytes"] or hashlib.sha256(payload).hexdigest() != declaration["sha256"]:
        raise ValueError("Manual case payload identity differs")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(metadata["array_identities"]):
            raise ValueError("Manual case array membership differs")
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, value in arrays.items():
        identity = metadata["array_identities"][name]
        if list(value.shape) != identity["shape"] or value.dtype.str != identity["dtype"] or hashlib.sha256(value.tobytes()).hexdigest() != identity["sha256"]:
            raise ValueError("Manual case array identity differs")
    return _validate_case(dict(metadata=metadata, arrays=arrays))


def source_path(name):
    for root in (ROOT, ROOT.parent / "Tdyssey"):
        path = root / "third_party" / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Existing third_party/{name} checkout is missing; no download is attempted")


def make_environment(shape, *, source_root=None):
    from .official import load_official_source

    cls = make_official_shape_env_class(load_official_source(source_path("diffusion_policy") if source_root is None else source_root))
    env = cls(shape=shape, legacy=True)
    env.seed(5478)
    env.reset()
    initial = env.capture_state()
    # Deliberately simple, disclosed manual scene; never a held evaluation scene.
    y = 240 if shape == "T" else 285
    initial = replace(initial, object_body=BodyState(Pose((256, y), 0)),
                      pusher_body=BodyState(Pose((256, 405), 0)), pusher_target=None,
                      control_step=0)
    observation = env.restore_state(initial)
    return env, observation


class ManualSession:
    """Only execute steps physics. Cache: eight recent cases, at most three
    previews each (latest per role plus the preview referenced by execution).
    Superseded unexecuted previews and least-recent cases are explicitly pruned.
    """

    CASE_CACHE_LIMIT = 8
    _RUNTIME_FIELDS = ("prediction_trace", "prediction_usage", "checkpoint_identity")

    def __init__(self, env_factory=make_environment, *, image_domain=LEGACY_DOMAIN):
        if image_domain not in (LEGACY_DOMAIN, OFFICIAL_DOMAIN):
            raise ValueError("Unknown manual image domain")
        self.image_domain = image_domain
        self.factory = env_factory
        self.env = None
        self.revision = 0
        self.role, self.horizon, self.noise_seed = "frozen", 1, 5478
        self.controls = self.resets = self.forecasts = 0
        self.pending = self.prediction = self.actual = self.measurement = None
        self._cases = {}
        self.case_evictions = 0
        self._issued_comparison = None
        self.last_case_key = self.last_accepted_request = self._issued_request_key = None
        self.target = None
        self.done = False
        self.reset("T")

    def invalidate(self):
        self.revision += 1
        self.pending = self.prediction = self.actual = self.measurement = None
        self._issued_request_key = None
        self._issued_comparison = None

    def reset(self, shape=None):
        shape = shape or self.shape
        if shape not in ("T", "L"):
            raise ValueError("Manual demo supports T and L")
        if self.env is not None:
            self.env.close()
        self.shape = shape
        self.env, self.observation = self.factory(shape)
        self.resets += 1
        self.images = deque([self.observation["image"].copy()], maxlen=4)
        self.actions = deque(maxlen=3)
        self.history_states = deque([_snapshot(self.env.capture_state())], maxlen=4)
        self.history_steps = deque(maxlen=3)
        self.done = False
        self.target = None
        # Four real observations, with three recorded hold actions between them.
        hold = np.asarray(self.observation["agent_pos"], dtype=np.float64)
        for _ in range(3):
            self._step(hold)
        self.invalidate()

    def configure(self, *, role=None, horizon=None):
        if role is not None:
            if role not in ("frozen", "adapted"):
                raise ValueError("Unknown checkpoint")
            self.role = role
        if horizon is not None:
            if self.image_domain == OFFICIAL_DOMAIN and horizon != 1:
                raise ValueError("Official original-color baseline supports H1 only")
            if horizon not in (1, 4):
                raise ValueError("Preview horizon must be 1 or 4 control steps")
            self.horizon = horizon
        self.invalidate()

    def select(self, target):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all() or (target < 0).any() or (target > 511).any():
            raise ValueError("Target must be a finite 2D workspace point in [0,511]")
        self.target = target.copy()
        self.invalidate()

    def request(self):
        if self.done:
            raise ValueError("The real episode has succeeded; reset before another action")
        if self.target is None:
            raise ValueError("Choose a target in the left panel first")
        if len(self.images) != 4 or len(self.actions) != 3:
            raise ValueError("Four real frames and three intervening actions are required")
        initial = _snapshot(self.env.capture_state())
        if initial != self.history_states[-1]:
            raise ValueError("Real state changed outside the captured manual history")
        request = dict(revision=self.revision, role=self.role, shape=self.shape,
                    image_domain=self.image_domain,
                    horizon=self.horizon, noise_seed=self.noise_seed,
                    target=self.target.copy(), images=np.stack(self.images),
                    past_actions=np.asarray(self.actions, dtype=np.float32),
                    goal=Snapshot.from_dict(initial).goal, initial_state=initial,
                    history_states=deepcopy(list(self.history_states)),
                    history_steps=deepcopy(list(self.history_steps)))
        request["request_key"] = _request_key(request)
        _validate_request(request)
        self._issued_request_key = request["request_key"]
        return request

    def accept(self, request, raw, measurement):
        if (request["revision"] != self.revision or request["role"] != self.role
                or request["shape"] != self.shape or request["horizon"] != self.horizon
                or request["noise_seed"] != self.noise_seed
                or not np.array_equal(request["target"], self.target)):
            raise ValueError("Prediction belongs to an obsolete scene/action/checkpoint")
        _validate_request(request)
        if request["request_key"] != self._issued_request_key:
            raise ValueError("Prediction differs from the issued physical request")
        raw = np.asarray(raw)
        if raw.shape != (self.horizon, 3, 96, 96) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError("Prediction is missing, malformed or nonfinite")
        measurement = _json_copy(measurement)
        self._archive_prediction(request, raw, measurement)
        self._issued_comparison = None
        self.pending, self.prediction, self.measurement = deepcopy(request), raw.copy(), deepcopy(measurement)
        self.actual = None
        self.forecasts += self.horizon

    def _archive_prediction(self, request, raw, measurement, *, comparison=False):
        """Bounded archive; retain the exact executed preview when indices move."""
        key = request["request_key"]
        runtime = {name: _json_copy(request.get(name)) for name in self._RUNTIME_FIELDS}
        case = self._cases.pop(key, None)
        if case is None:
            fixed = {name: deepcopy(value) for name, value in request.items()
                     if name not in (*self._RUNTIME_FIELDS, "comparison_only", "comparison_revision")}
            case = dict(request=fixed, predictions=[], execution=None, discarded_predictions=0)
        case["predictions"].append(dict(role=request["role"], revision=request["revision"],
            accepted_revision=self.revision, raw=raw.copy(), measurement=deepcopy(measurement),
            comparison_only=comparison, **runtime))
        latest = {prediction["role"]: index for index, prediction in enumerate(case["predictions"])}
        keep = set(latest.values())
        execution = case["execution"]
        if execution is not None:
            keep.add(execution["prediction_index"])
        indices = sorted(keep)
        if execution is not None:
            execution["prediction_index"] = indices.index(execution["prediction_index"])
        case["discarded_predictions"] += len(case["predictions"])-len(indices)
        case["predictions"] = [case["predictions"][index] for index in indices]
        self._cases[key] = case
        while len(self._cases) > self.CASE_CACHE_LIMIT:
            del self._cases[next(iter(self._cases))]
            self.case_evictions += 1
        self.last_case_key, self.last_accepted_request = key, deepcopy(request)

    def _comparison_case(self):
        key = self.last_case_key
        if key not in self._cases or self.env is None:
            raise ValueError("No recent accepted case is available for comparison")
        case = self._cases[key]
        execution = case["execution"]
        expected = (execution["steps"][-1]["after_state"]
                    if execution is not None and execution["steps"] else case["request"]["initial_state"])
        expected_image = (execution["raw"][-1] if execution is not None and execution["raw"]
                          else case["request"]["images"][-1])
        if (_snapshot(self.env.capture_state()) != expected
                or not np.array_equal(self.observation["image"], expected_image)):
            raise ValueError("Live physical state differs from the captured comparison endpoint")
        return case

    def request_for_comparison(self, role):
        """Use archived inputs only; no model call, restore, control or live action."""
        if role not in ("frozen", "adapted"):
            raise ValueError("Unknown comparison checkpoint")
        case = self._comparison_case()
        request = deepcopy(case["request"])
        request.update(role=role, comparison_only=True, comparison_revision=self.revision)
        _validate_request(request)
        self.pending = None
        self._issued_request_key = None
        self._issued_comparison = (request["request_key"], role, self.revision)
        return request

    def accept_comparison(self, request, raw, measurement):
        """Archive a counterfactual preview; it can never become a live pending action."""
        expected = (request["request_key"], request["role"], self.revision)
        if (request.get("comparison_only") is not True or self._issued_comparison != expected
                or request.get("comparison_revision") != self.revision
                or request["request_key"] != self.last_case_key):
            raise ValueError("Comparison belongs to an obsolete request")
        _validate_request(request)
        case = self._comparison_case()
        raw = np.asarray(raw)
        if raw.shape != (request["horizon"], 3, 96, 96) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError("Comparison prediction is malformed or nonfinite")
        measurement = _json_copy(measurement)
        self._archive_prediction(request, raw, measurement, comparison=True)
        self.pending = self._issued_request_key = self._issued_comparison = None
        self.role, self.horizon = request["role"], request["horizon"]
        self.target = request["target"].copy()
        self.prediction, self.measurement = raw.copy(), deepcopy(measurement)
        execution = case["execution"]
        storage = "model" if self.image_domain == OFFICIAL_DOMAIN else "t1"
        self.actual = np.stack(execution[storage]) if execution is not None and execution[storage] else None
        if self.actual is not None and len(self.actual) != self.horizon:
            self.measurement = None
        self.forecasts += request["horizon"]

    def _step(self, action):
        action = np.asarray(action, dtype=np.float64)
        before = deepcopy(self.history_states[-1])
        observation, reward, done, info = self.env.step(action)
        self.observation = observation
        self.images.append(observation["image"].copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.controls += 1
        self.done = bool(done)
        after = _snapshot(self.env.capture_state())
        row = dict(reward=float(reward), done=self.done, contacts=int(info["n_contacts"]),
            action=action.tolist(), control_step=after["control_step"],
            before_state=before, after_state=after)
        self.history_states.append(after)
        self.history_steps.append(deepcopy(row))
        return row

    def execute(self):
        if self.pending is None or self.pending["revision"] != self.revision or self.done:
            raise ValueError("Generate a current prediction before executing")
        request = self.pending
        self.pending = None  # An action cannot be executed twice by key repeat.
        execution = dict(status="running", role=request["role"],
            prediction_index=len(self._cases[request["request_key"]]["predictions"])-1,
            steps=[], raw=[], t1=[], result=None)
        storage = "model" if self.image_domain == OFFICIAL_DOMAIN else "t1"
        if storage == "model":
            execution.pop("t1")
            execution[storage] = []
        self._cases[request["request_key"]]["execution"] = execution
        actual, rows = [], []
        try:
            for _ in range(request["horizon"]):
                rows.append(self._step(request["target"]))
                raw_frame = self.observation["image"].copy()
                actual.append(display_observation(raw_frame, self.image_domain))
                execution["steps"].append(deepcopy(rows[-1]))
                execution["raw"].append(raw_frame)
                execution[storage].append(actual[-1].copy())
                if self.done:
                    break
        except BaseException as error:
            execution.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
            raise
        self.actual = np.stack(actual)
        matched = len(actual) - 1
        if len(actual) != request["horizon"]:
            # The decoder described the requested final horizon, not this
            # earlier real success boundary now displayed on screen.
            self.measurement = None
        state = self.env.capture_state()
        truth = goal_metrics(official_shape(self.shape), state.object_body.pose, state.goal)
        result = dict(steps=rows, executed_controls=len(rows), matched_horizon=matched + 1,
                      prediction_role=request["role"], request_key=request["request_key"],
                      truth_coverage=truth.coverage, truth_success=self.done,
                      pixel_mae=float(np.abs(self.prediction[matched] - self.actual[matched]).mean()),
                      note="Manual exploration only; image MAE is not a dynamics acceptance score")
        execution.update(status="completed", result=deepcopy(result))
        self.revision += 1
        return result

    def comparison_predictions(self, case_key=None):
        """Latest accepted preview per role for one fixed request; no new calls."""
        key = case_key or self.last_case_key
        if key not in self._cases:
            return {}
        return {prediction["role"]: dict(raw=prediction["raw"].copy(),
                    measurement=deepcopy(prediction["measurement"]), request_key=key,
                    **{name: deepcopy(prediction.get(name)) for name in self._RUNTIME_FIELDS})
                for prediction in self._cases[key]["predictions"]}

    def capture_case(self, case_key=None):
        """Detach the accepted request and its saved outcomes, even after execute."""
        key = case_key or self.last_case_key
        if key not in self._cases:
            raise ValueError("Accept a prediction before saving a reproducible case")
        case = self._cases[key]
        request = case["request"]
        request_metadata = {name: deepcopy(value) for name, value in request.items()
                            if name not in ("images", "past_actions", "target", "goal",
                                "comparison_only", "comparison_revision", *self._RUNTIME_FIELDS)}
        request_metadata["target"] = request["target"].tolist()
        metadata = dict(kind="manual_world_model_case.v1", training_data=False,
            usage="manual exploration only; no training/evaluation-set claim", request=request_metadata,
            latest_accepted_role=case["predictions"][-1]["role"],
            requested_actions=[request_metadata["target"][:] for _ in range(request["horizon"])],
            t1_sha256=T1_SHA256, predictions=[], execution=None,
            cache_policy=dict(maximum_recent_cases=self.CASE_CACHE_LIMIT,
                previews="latest per role plus exact executed reference", maximum_previews_per_case=3,
                discarded_superseded_previews=case["discarded_predictions"],
                evicted_cases=self.case_evictions),
            current_revision_at_capture=self.revision,
            live_request_eligible=bool(self.pending is not None and not self.done
                and self.pending["request_key"] == key and self.pending["revision"] == self.revision))
        domain = request.get("image_domain", LEGACY_DOMAIN)
        if domain == OFFICIAL_DOMAIN:
            metadata.update(kind="manual_world_model_case.v2", image_domain=domain,
                t1_sha256=None, model_input_transform="identity official RGB; no T1",
                prediction_output="remote helper clips native sample to [0,1]; H1 only")
        history_key, actual_key, storage = _case_view_keys(metadata)
        arrays = dict(history_raw=request["images"].copy(), past_actions=request["past_actions"].copy())
        arrays[history_key] = np.stack([display_observation(frame, domain) for frame in request["images"]])
        for index, prediction in enumerate(case["predictions"]):
            name = f"prediction_{index:03d}"
            arrays[name] = prediction["raw"].copy()
            metadata["predictions"].append(dict(role=prediction["role"], revision=prediction["revision"],
                accepted_revision=prediction["accepted_revision"], comparison_only=prediction["comparison_only"],
                array=name, measurement=deepcopy(prediction["measurement"]),
                **{field: deepcopy(prediction.get(field)) for field in self._RUNTIME_FIELDS}))
        execution = case["execution"]
        if execution is not None:
            metadata["execution"] = {name: deepcopy(value) for name, value in execution.items()
                                     if name not in ("raw", "t1", "model")}
            for name, source in (("actual_raw", "raw"), (actual_key, storage)):
                arrays[name] = (np.stack(execution[source]) if execution[source]
                                else np.empty((0, 3, 96, 96), dtype=np.float32))
        return _validate_case(dict(metadata=metadata, arrays=arrays))

    def save_case(self, path, case_key=None):
        """Save captured data only; never calls an environment or predictor."""
        return save_case_bundle(path, self.capture_case(case_key))

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class PinnedPredictor:
    """One active model; reuse the audited model-only loader and native sampler."""

    def __init__(self):
        self.role = self.model = self.sampler = self.assets = None
        self.loads = 0
        self.load_reservations = 0
        self.forecast_reservations = self.forecast_returns = self.euler_reservations = 0
        self.forecast_cap = self.model_load_cap = None
        self.last_trace, self.last_partial_predictions = [], []
        self.peak_allocated = self.peak_reserved = 0

    def usage(self):
        return dict(model_load_reservations=self.load_reservations, model_loads=self.loads,
            forecast_reservations=self.forecast_reservations, forecast_returns=self.forecast_returns,
            euler_reservations=self.euler_reservations,
            note="Reservations remain spent on partial sampling, decoding or UI rejection")

    def _rng_hash(self):
        capture = getattr(self.sampler, "capture", None)
        return hashlib.sha256(capture()).hexdigest() if callable(capture) else None

    def check_cuda(self):
        import torch

        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())
        if self.peak_allocated > 512 * 1024**2 or self.peak_reserved > 1024**3:
            raise MemoryError("Manual preview exceeded its 512MiB allocated / 1GiB reserved CUDA bounds")

    def load(self, role):
        if self.role == role:
            return
        self.close()
        from .gpc_evaluation_assets import verify_evaluation_assets, load_model_only_state
        from .gpc_supervised_adaptation import make_real_denoiser
        from .gpc_adaptation import NativeSampler
        import torch

        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no download or CPU fallback is attempted")
        if self.assets is None:
            self.assets = verify_evaluation_assets(ROOT)
        if self.model_load_cap is not None and self.load_reservations >= self.model_load_cap:
            raise RuntimeError("Model-load reservation cap reached; no retry")
        self.load_reservations += 1
        state = load_model_only_state(role, self.assets)
        try:
            self.model = make_real_denoiser(source_path("gpc_code"),
                self.assets["weights"]["frozen"]["path"], "cuda",
                dict(execute=True, evaluation_only=True,
                     validated_checkpoint_identity=self.assets["weights"][role]), state_dict=state)
        finally:
            del state
        self.model.eval().requires_grad_(False)
        self.sampler = NativeSampler(self.model, self.check_cuda)
        self.role = role
        self.loads += 1
        self.check_cuda()

    def predict(self, request):
        if request.get("image_domain", LEGACY_DOMAIN) != LEGACY_DOMAIN:
            raise ValueError("Legacy predictor refuses official original-color requests")
        import torch
        from .wm_image_measurement_v2 import decode
        from .wm_compat_engine import compact_measurement

        started = time.monotonic()
        self.last_trace, self.last_partial_predictions = [], []
        self.load(request["role"])
        context = np.stack([transform_t1(frame)[0] for frame in request["images"]])
        past = list(request["past_actions"].copy())
        raw = []
        # Same fixed sampling noise for checkpoint comparisons; external RNG restored.
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
                self.sampler.seed(request["noise_seed"])
                for horizon in range(1, request["horizon"] + 1):
                    if self.forecast_cap is not None and self.forecast_reservations >= self.forecast_cap:
                        raise RuntimeError("Forecast reservation cap reached; no extra model call")
                    absolute = np.asarray(past + [request["target"]], dtype=np.float32)
                    window = absolute / np.float32(511) * np.float32(2) - np.float32(1)
                    record = dict(horizon=horizon, status="reserved",
                        input_sha256=hashlib.sha256(context.tobytes()).hexdigest(),
                        action_sha256=hashlib.sha256(window.tobytes()).hexdigest(),
                        actions_absolute=absolute.tolist(), actions_normalized=window.tolist(),
                        rng_entry_sha256=self._rng_hash(), euler_reserved=3)
                    self.last_trace.append(record)
                    self.forecast_reservations += 1
                    self.euler_reservations += 3
                    try:
                        image = self.sampler.sample(context, window)
                        self.forecast_returns += 1
                        record.update(status="returned", rng_exit_sha256=self._rng_hash())
                        if image.shape != (3, 96, 96) or image.dtype != np.float32 or not np.isfinite(image).all():
                            raise ValueError("Malformed or nonfinite prediction")
                        record.update(status="validated", raw_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                            raw_min=float(image.min()), raw_max=float(image.max()))
                    except BaseException as error:
                        record.update(status="failed", error=type(error).__name__+": "+str(error))
                        raise
                    raw.append(image)
                    self.last_partial_predictions.append(image.copy())
                    # Raw feedback only. T1 is never applied to a generated frame.
                    context = np.concatenate((context[1:], image[None]), axis=0)
                    past = past[1:] + [request["target"]]
            measurement = compact_measurement(decode(raw[-1], request["shape"], request["goal"]))
        finally:
            request["prediction_trace"] = deepcopy(self.last_trace)
            request["prediction_usage"] = self.usage()
            request["checkpoint_identity"] = (deepcopy(self.assets["weights"][request["role"]])
                if self.assets is not None else None)
        return np.stack(raw), measurement, time.monotonic() - started

    def close(self):
        self.role = self.sampler = self.model = None
        gc.collect()
        import sys
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def event(stream, kind, **details):
    stream.write(json.dumps(dict(time=time.time(), kind=kind, **details),
                            ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
