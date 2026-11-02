"""Metadata-only formal TRAIN sampling; geometry labels never enter the model.

Every180 slots contain18 letters with7 moving,1 intermediate,1 near-static
contact and1 near-static no-contact draw each. Batches consume this stream
continuously. Independent trajectory and window queues may repeat only after
their respective queue is exhausted; those are training exposures, not new data.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

LETTERS = tuple("ABCEFHIJLMOPSTUVXY")
STRATA = ("moving", "intermediate", "near_static_contact", "near_static_no_contact")
QUOTAS = dict(zip(STRATA, (7, 1, 1, 1)))
SPACES = ("center", "left", "right", "top", "bottom", "left-top", "right-top", "left-bottom", "right-bottom")
TRAIN_REFERENCE_SHA256 = "23865f38e19eb3b3fddf14b17c96f06586468bd04516e2403aa743f42c557d7e"
POLICY = dict(kind="native-rgb-formal-70-10-10-10.v1", seed=5478, letters=list(LETTERS),
    strata=list(STRATA), per_letter_quotas=QUOTAS, cycle_slots=180,
    slot_rng="numpy.PCG64 seed5478; independent of per-queue RNGs and model RNG",
    queue_rng="numpy.PCG64 from SHA256(seed,queue kind,letter,stratum,trajectory)",
    trajectory_order="balanced shuffled queue; exhaust before reshuffle",
    window_order="per trajectory+stratum shuffled queue; exhaust before reshuffle",
    motion=dict(moving_translation_world_px_gte=2., moving_rotation_deg_gte=1.,
        near_static_translation_world_px_lte=.5, near_static_rotation_rad_lte=.005,
        H1_history_transitions=4, minimum_anchor=3),
    input_scope="Only dataset index is selected; RGB/action model input is unchanged")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def candidate_rows_from_records(records):
    """Match NativeRGBDataset's catalogue without constructing it or rendering.

    The independently audited state/geometry validity must agree with declared
    four-transition history validity. A mismatch fails rather than shifting IDs.
    """
    from .native_rgb_motion_statistics import transition_rows
    result = []
    for record in records:
        entry, arrays = record["entry"], record["arrays"]
        if entry.get("split") != "train" or entry.get("stage") not in ("2A", "2B") or entry["shape"] not in LETTERS:
            raise ValueError("Only fixed formal2A+2B TRAIN records may form pools")
        rows = transition_rows(entry, arrays, record.get("steps"))
        declared = arrays["valid_transition"]
        for anchor in range(3, len(declared)):
            eligible = all(bool(v) for v in declared[anchor-3:anchor+1])
            if eligible != rows[anchor]["valid_h1_candidate"]:
                raise ValueError("Independent H1 validity differs from the dataset catalogue")
            if eligible:
                result.append(dict(rows[anchor], dataset_index=len(result), anchor=anchor))
    return result


def build_pool_manifest(rows, *, data_identity):
    """Freeze TRAIN index membership and physical-data identity; draw nothing."""
    data_identity = deepcopy(data_identity)
    for name in ("manifest_sha256", "audit_sha256", "split_sha256", "train_reference_manifest_sha256"):
        value = data_identity.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Explicit hash-bound formal data identity required: " + name)
    if data_identity["train_reference_manifest_sha256"] != TRAIN_REFERENCE_SHA256:
        raise ValueError("Only the frozen2A+2B formal TRAIN reference is approved")
    groups = {letter: {name: defaultdict(list) for name in STRATA} for letter in LETTERS}
    windows, seen, pairs, trajectory_letters = [], set(), set(), {}
    for row in rows:
        index, anchor = row["dataset_index"], row["anchor"]
        letter, trajectory, name = row["letter"], row["trajectory_id"], row["index_stratum"]
        if (type(index) is not int or index < 0 or index in seen or type(anchor) is not int or anchor < 3
                or letter not in LETTERS or name not in STRATA or not isinstance(trajectory, str) or not trajectory
                or row.get("split") != "train" or row.get("valid_h1_candidate") is not True
                or (trajectory, anchor) in pairs or trajectory_letters.get(trajectory, letter) != letter):
            raise ValueError("Invalid, duplicate or non-TRAIN H1 pool row")
        seen.add(index)
        pairs.add((trajectory, anchor))
        trajectory_letters[trajectory] = letter
        groups[letter][name][trajectory].append(index)
        windows.append(dict(dataset_index=index, anchor=anchor, letter=letter, trajectory_id=trajectory,
                            index_stratum=name, split="train", valid_h1_candidate=True))
    windows.sort(key=lambda row: row["dataset_index"])
    if [row["dataset_index"] for row in windows] != list(range(len(windows))):
        raise ValueError("Pool indices must cover the entire ordered TRAIN H1 catalogue")
    for letter in LETTERS:
        for name in STRATA:
            if not groups[letter][name]:
                raise ValueError("Empty letter-stratum; no borrowing/replacement is allowed: " + letter + "/" + name)
            groups[letter][name] = {key: sorted(value) for key, value in sorted(groups[letter][name].items())}
    identity = dict(kind="native-rgb-formal-sampling-identity.v1", policy=POLICY["kind"],
        policy_sha256=canonical_hash(POLICY), pools_sha256=canonical_hash(groups),
        data_sha256=canonical_hash(data_identity), catalogue_sha256=canonical_hash(windows), seed=5478)
    value = dict(kind="native-rgb-formal-training-pools.v1", policy=deepcopy(POLICY), identity=identity,
        data_identity=data_identity, windows=windows, groups=groups, total_windows=len(windows),
        strata_counts=dict(Counter(row["index_stratum"] for row in windows)))
    value["manifest_sha256"] = canonical_hash(value)
    return value


def validate_dataset_catalogue(dataset, pool_manifest):
    """Validate an already-created dataset's metadata; never access __getitem__."""
    if dataset.split != "train":
        raise ValueError("Formal sampler accepts TRAIN only")
    root = Path(dataset.root)
    expected = pool_manifest["data_identity"]
    if (_file_hash(root / "manifest.json") != expected["manifest_sha256"]
            or _file_hash(root / "physics-audit.json") != expected["audit_sha256"]
            or canonical_hash(dataset.manifest["splits"]) != expected["split_sha256"]
            or dataset.audit.get("physics_collection_gate_passed") is not True):
        raise ValueError("Dataset manifest/audit/split identity differs from formal pools")
    if len(dataset.windows) != len(pool_manifest["windows"]):
        raise ValueError("Formal sampler must cover all original TRAIN H1 windows")
    for index, (entry_index, anchor) in enumerate(dataset.windows):
        entry, row = dataset.entries[entry_index], pool_manifest["windows"][index]
        if (entry["split"] != "train" or entry["shape"] != row["letter"]
                or entry["trajectory_id"] != row["trajectory_id"] or anchor != row["anchor"]):
            raise ValueError("TRAIN dataset index/trajectory/anchor differs: " + str(index))
    return True


def support_summary(rows):
    """Expose scarce independent support and space/behavior gaps, without draws."""
    rows = list(rows)
    report, cells = {}, []
    for letter in LETTERS:
        report[letter] = {}
        for name in STRATA:
            selected = [row for row in rows if row["letter"] == letter and row["index_stratum"] == name]
            by_space = {space: [row for row in selected if row.get("requested_spatial_class") == space] for space in SPACES}
            value = dict(unique_windows=len(selected), independent_trajectories=len({row["trajectory_id"] for row in selected}),
                initial_space={space: dict(windows=len(values), independent_trajectories=len({r["trajectory_id"] for r in values})) for space, values in by_space.items()},
                actual_spatial_before=dict(Counter(str(row.get("actual_spatial_class_before", "unknown")) for row in selected)),
                actual_boundary_behavior=dict(Counter(str(row.get("boundary_behavior", "unknown")) for row in selected)),
                contact=dict(Counter(str(row.get("contact", "unknown")) for row in selected)),
                motion=dict(Counter(str(row.get("motion_class", "unknown")) for row in selected)))
            report[letter][name] = value
            cells.extend(dict(letter=letter, stratum=name, initial_space=space) for space, values in by_space.items() if not values)
    return dict(total_windows=len(rows), strata_counts=dict(Counter(row["index_stratum"] for row in rows)),
        independent_trajectories_by_stratum={name: len({r["trajectory_id"] for r in rows if r["index_stratum"] == name}) for name in STRATA},
        per_letter_stratum=report, empty_letter_stratum_initial_space_cells=cells,
        lowest_support=sorted([dict(letter=l, stratum=s, unique_windows=v["unique_windows"], independent_trajectories=v["independent_trajectories"])
            for l, values in report.items() for s, v in values.items()], key=lambda r: (r["unique_windows"], r["independent_trajectories"], r["letter"], r["stratum"]))[:12],
        boundary_scope="Geometry/recorded contact proxies only; no wall arbiter or physical success inferred")


class FormalWindowOrder:
    """Explicit formal sampler API: take/count, cursor, identity and full state."""
    def __init__(self, pool_manifest, seed=5478):
        import numpy as np
        if type(seed) is not int or seed != 5478:
            raise ValueError("Formal slot stream requires the frozen independent seed5478")
        rebuilt = build_pool_manifest(pool_manifest["windows"], data_identity=pool_manifest["data_identity"])
        if pool_manifest != rebuilt:
            raise ValueError("Formal pool/policy/catalogue/data hash or membership changed")
        self._manifest = rebuilt
        self._np = np
        self.cursor = self.slot_cycle = self.slot_position = 0
        self.slot_rng = np.random.default_rng(seed)
        self.slot_order = self._new_slots()
        self.trajectory_queues, self.window_queues = {}, {}
        for letter in LETTERS:
            for name in STRATA:
                key = letter + "|" + name
                groups = rebuilt["groups"][letter][name]
                self.trajectory_queues[key] = self._new_queue(list(groups), ["trajectory", letter, name])
                self.window_queues[key] = {trajectory: self._new_queue(indices, ["window", letter, name, trajectory])
                    for trajectory, indices in groups.items()}

    @classmethod
    def from_dataset(cls, dataset, pool_manifest, seed=5478, indices=None):
        if indices is not None:
            raise ValueError("Formal sampler uses all TRAIN candidates; no subset indices")
        validate_dataset_catalogue(dataset, pool_manifest)
        return cls(pool_manifest, seed)

    def identity(self):
        return deepcopy(self._manifest["identity"])

    def window_metadata(self, index):
        if type(index) is not int or not 0 <= index < len(self._manifest["windows"]):
            raise ValueError("Formal dataset index outside the TRAIN catalogue")
        return deepcopy(self._manifest["windows"][index])

    def _new_slots(self):
        slots = [[letter, name] for letter in LETTERS for name in STRATA for _ in range(QUOTAS[name])]
        return [slots[i] for i in self.slot_rng.permutation(len(slots)).tolist()]

    def _new_queue(self, values, label):
        seed = int(canonical_hash([5478, *label])[:32], 16)
        rng = self._np.random.default_rng(seed)
        return dict(order=rng.permutation(values).tolist(), position=0, epoch=0, rng=rng)

    @staticmethod
    def _draw_queue(queue, values):
        if queue["position"] == len(queue["order"]):
            queue.update(order=queue["rng"].permutation(values).tolist(), position=0, epoch=queue["epoch"] + 1)
        result = queue["order"][queue["position"]]
        queue["position"] += 1
        return result

    def take(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("Positive formal draw count required")
        result = []
        for _ in range(count):
            if self.slot_position == 180:
                self.slot_cycle += 1
                self.slot_position = 0
                self.slot_order = self._new_slots()
            letter, name = self.slot_order[self.slot_position]
            key = letter + "|" + name
            groups = self._manifest["groups"][letter][name]
            trajectory = self._draw_queue(self.trajectory_queues[key], list(groups))
            index = self._draw_queue(self.window_queues[key][trajectory], groups[trajectory])
            result.append(index)
            self.slot_position += 1
            self.cursor += 1
        return result

    def state_dict(self):
        def queue(value):
            return dict(order=deepcopy(value["order"]), position=value["position"], epoch=value["epoch"], rng=deepcopy(value["rng"].bit_generator.state))
        state = dict(kind="native-rgb-formal-sampling-state.v1", identity=self.identity(), cursor=self.cursor,
            slot_cycle=self.slot_cycle, slot_position=self.slot_position, slot_order=deepcopy(self.slot_order),
            slot_rng=deepcopy(self.slot_rng.bit_generator.state),
            trajectory_queues={key: queue(value) for key, value in self.trajectory_queues.items()},
            window_queues={key: {t: queue(value) for t, value in groups.items()} for key, groups in self.window_queues.items()})
        state["state_sha256"] = canonical_hash(state)
        return state

    def load_state_dict(self, state):
        candidate = deepcopy(state)
        checksum = candidate.pop("state_sha256", None)
        if checksum != canonical_hash(candidate) or candidate["identity"] != self.identity():
            raise ValueError("Formal sampler state/policy/pool/data identity differs")
        expected_keys = set(self.state_dict()) - {"state_sha256"}
        if set(candidate) != expected_keys or candidate["kind"] != "native-rgb-formal-sampling-state.v1":
            raise ValueError("Unknown/incomplete formal sampling state")
        cycle, position, cursor = (candidate[k] for k in ("slot_cycle", "slot_position", "cursor"))
        expected_slots = Counter((letter, name) for letter in LETTERS for name in STRATA for _ in range(QUOTAS[name]))
        if (any(type(v) is not int for v in (cycle, position, cursor)) or cycle < 0 or not 0 <= position <= 180
                or cursor != cycle * 180 + position or Counter(tuple(v) for v in candidate["slot_order"]) != expected_slots):
            raise ValueError("Invalid slot cycle/order/cursor")

        def rng(value):
            result = self._np.random.default_rng(0)
            if value.get("bit_generator") != "PCG64":
                raise ValueError("Unexpected formal RNG algorithm")
            result.bit_generator.state = deepcopy(value)
            return result

        def queue(value, values):
            if (set(value) != {"order", "position", "epoch", "rng"} or sorted(value["order"]) != sorted(values)
                    or type(value["position"]) is not int or not 0 <= value["position"] <= len(values)
                    or type(value["epoch"]) is not int or value["epoch"] < 0):
                raise ValueError("Invalid trajectory/window queue or cursor")
            return dict(order=deepcopy(value["order"]), position=value["position"], epoch=value["epoch"], rng=rng(value["rng"]))

        if set(candidate["trajectory_queues"]) != set(self.trajectory_queues) or set(candidate["window_queues"]) != set(self.window_queues):
            raise ValueError("Incomplete formal queues")
        trajectories, windows = {}, {}
        for key in self.trajectory_queues:
            letter, name = key.split("|")
            groups = self._manifest["groups"][letter][name]
            trajectories[key] = queue(candidate["trajectory_queues"][key], list(groups))
            if set(candidate["window_queues"][key]) != set(groups):
                raise ValueError("Incomplete trajectory window queues")
            windows[key] = {t: queue(candidate["window_queues"][key][t], indices) for t, indices in groups.items()}
            trajectory_queue = trajectories[key]
            drawn = trajectory_queue["epoch"] * len(groups) + trajectory_queue["position"]
            prefix = Counter(tuple(v) for v in candidate["slot_order"][:position])
            if drawn != cycle * QUOTAS[name] + prefix[(letter, name)]:
                raise ValueError("Trajectory cursor/exposure does not match the slot stream")
            active_trajectories = set(trajectory_queue["order"][:trajectory_queue["position"]])
            for trajectory, indices in groups.items():
                window_queue = windows[key][trajectory]
                window_draws = window_queue["epoch"] * len(indices) + window_queue["position"]
                if window_draws != trajectory_queue["epoch"] + int(trajectory in active_trajectories):
                    raise ValueError("Window cursor/exposure does not match its trajectory queue")
        slot_rng = rng(candidate["slot_rng"])
        self.cursor, self.slot_cycle, self.slot_position = cursor, cycle, position
        self.slot_order, self.slot_rng = deepcopy(candidate["slot_order"]), slot_rng
        self.trajectory_queues, self.window_queues = trajectories, windows
