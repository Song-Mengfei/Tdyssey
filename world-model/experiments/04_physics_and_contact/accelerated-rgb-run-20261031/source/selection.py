"""Saved-only, family-isolated clean-window selection for the authorized run.

No simulator, model, renderer, Torch, or optimizer is constructed here.  Privileged
saved states/contact observations are used only to audit/select RGB/action rows.
The first suspicious control and its entire dependent suffix remain quarantined,
including contact-recovery candidates.  Old diagnostic data is never discovered.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import zlib

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LETTERS = tuple("TLOCXB")
HORIZONS = (1, 4, 8, 16)
SCALE = 96. / 512.
OBVIOUS = {1: (1., 3.), 4: (2., 5.), 8: (4., 10.), 16: (8., 20.)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _storage(root, config, needed):
    # The owner's common governor measures the same existing experiment set;
    # there is deliberately no inherited 3GiB per-stage check here.
    if (Path(__file__).parent / "common.py").exists():
        from common import check_storage
        check_storage(root, config, reserve_bytes=needed, full=False)


def write(path, value, root=None, config=None):
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    if root is not None:
        _storage(root, config or {}, len(payload))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
    temporary.replace(path)


def centers(letter, states):
    from tdyssey.envs.official import official_shape
    local = np.asarray(official_shape(letter).centroid, np.float64)
    c, s = np.cos(states[:, 4]), np.sin(states[:, 4])
    return states[:, 2:4] + np.column_stack((c * local[0] - s * local[1],
                                            s * local[0] + c * local[1]))


def _positive_contacts(data, n):
    """Control flags plus actual post-solve positive-impulse microstep time."""
    positive = np.asarray(data.get("contact_positive", np.zeros(n, bool)), bool)
    if positive.shape != (n,):
        raise ValueError("Positive-contact control flags are misaligned")
    ow = np.zeros(n, bool)
    totals = dict(observed_seconds=0., positive_ao_seconds=0.,
                  positive_ow_seconds=0., telemetry_available=False)
    if "telemetry_zlib" not in data:
        return positive, ow, totals
    payload = zlib.decompress(data["telemetry_zlib"].tobytes())
    if "telemetry_sha256" in data and hashlib.sha256(payload).digest() != data["telemetry_sha256"].tobytes():
        raise ValueError("Native telemetry checksum mismatch")
    decoded = json.loads(payload)
    telemetry = decoded.get("telemetry", []) if isinstance(decoded, dict) else decoded
    if len(telemetry) != n:
        raise ValueError("Native telemetry/control timeline mismatch")
    observed_ao = np.zeros(n, bool)
    for i, control in enumerate(telemetry):
        for substep in control.get("substeps", []):
            if not substep.get("executed", False):
                continue
            dt = float(substep.get("dt", .01))
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Invalid actual native duration")
            totals["observed_seconds"] += dt
            kinds = set()
            for contact in substep.get("contacts", []):
                if (contact.get("phase") == "post_solve" and
                        np.linalg.norm(contact.get("total_impulse", [0., 0.])) > 0.):
                    kinds.add(contact.get("kind"))
            if "agent_object" in kinds:
                totals["positive_ao_seconds"] += dt
                observed_ao[i] = True
            if "object_wall" in kinds:
                totals["positive_ow_seconds"] += dt
                ow[i] = True
    if "contact_positive" in data and not np.array_equal(positive, observed_ao):
        raise ValueError("Saved AO flags disagree with native positive impulses")
    totals["telemetry_available"] = True
    return observed_ao, ow, totals


def load_attempt(path, metadata):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    actions = np.asarray(data["requested_actions"], np.float64)
    passed = np.asarray(data["passed_actions"], np.float64)
    n = len(actions)
    if actions.shape != (n, 2) or passed.shape != (n, 2) or not np.array_equal(actions, passed):
        raise ValueError(f"Requested/passed action mismatch: {path}")
    if not np.isfinite(actions).all() or np.any(actions < 0) or np.any(actions > 512):
        raise ValueError(f"Action outside original physical domain: {path}")
    states, rgb = np.asarray(data["states"]), np.asarray(data["rgb"])
    valid = np.asarray(data["control_valid"], bool)
    if (states.shape != (n + 1, 15) or rgb.shape != (n + 1, 3, 96, 96)
            or rgb.dtype != np.uint8 or valid.shape != (n,)):
        raise ValueError(f"State/RGB/action alignment mismatch: {path}")
    clean = metadata.get("clean_control_count")
    if type(clean) is not int or not 0 <= clean <= n:
        raise ValueError(f"Missing or invalid explicit clean prefix: {path}")
    if not np.isfinite(states[:clean + 1]).all() or not valid[:clean].all():
        raise ValueError(f"Invalid control inside claimed clean prefix: {path}")
    for key in ("first_warning_control", "first_suspicious_control", "first_hard_stop_control"):
        if metadata.get(key) is not None and clean > int(metadata[key]):
            raise ValueError(f"Clean prefix crosses {key}: {path}")
    if "control_time_seconds" in data:
        times = np.asarray(data["control_time_seconds"], np.float64)
        if (times.shape != (n + 1,) or not np.isfinite(times).all()
                or np.any(np.abs(np.diff(times[:clean + 1]) - .1) > 1e-9)):
            raise ValueError(f"Clean controls lack complete true 10Hz frames: {path}")
    positive, ow, contact_totals = _positive_contacts(data, n)
    data.update(actions=actions, states=states, rgb=rgb, valid=valid,
                clean=clean, positive=positive, ow=ow, contact_totals=contact_totals)
    return data


def _committed(metadata, anchor, horizon):
    segments = metadata.get("segments", metadata.get("blocks", []))
    return any(int(seg.get("start_control", -1)) <= anchor and
               anchor + horizon <= int(seg.get("end_control", -1))
               for seg in segments)


def _metric(data, centroid, anchor, horizon):
    stop = anchor + horizon
    path = centroid[anchor:stop + 1]
    state = data["states"][anchor:stop + 1]
    trans = float(np.linalg.norm(path[-1] - path[0]) * SCALE)
    rotation = float(np.rad2deg(state[-1, 4] - state[0, 4]))
    pusher = float(np.linalg.norm(state[-1, :2] - state[0, :2]) * SCALE)
    contact = bool(data["positive"][anchor:stop].any())
    moving = trans >= .1 or abs(rotation) >= .5
    event = ("contact_and_move" if contact and moving else
             "contact_blocked_or_micro" if contact else
             "uncontacted_object_motion" if moving else
             "empty_pusher_motion" if pusher >= .1 else "quiet_or_micro")
    bitstatic = bool(all(np.array_equal(frame, data["rgb"][anchor])
                         for frame in data["rgb"][anchor + 1:stop + 1]))
    exact_quiet_context = (bool(np.all(data["states"][anchor - 3:stop + 1] ==
                                      data["states"][anchor - 3])) and
                           bool(np.all(data["rgb"][anchor - 3:stop + 1] ==
                                       data["rgb"][anchor - 3])))
    obvious_t, obvious_r = OBVIOUS[horizon]
    return dict(translation_imagepx=trans,
                cumulative_translation_imagepx=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum() * SCALE),
                signed_rotation_deg=rotation, abs_rotation_deg=abs(rotation),
                cumulative_rotation_deg=float(np.abs(np.rad2deg(np.diff(state[:, 4]))).sum()),
                pusher_translation_imagepx=pusher, contact_positive=contact,
                object_wall_contact_positive=bool(data["ow"][anchor:stop].any()),
                diagnostic_contact_and_motion=contact and moving,
                obvious_translation=trans >= obvious_t, obvious_rotation=abs(rotation) >= obvious_r,
                bitstatic_future=bitstatic, exact_quiet_context=exact_quiet_context,
                moving=moving, event=event)


def _row(metadata, raw, data, centroid, anchor, horizon):
    metrics = _metric(data, centroid, anchor, horizon)
    identity = f"{metadata['family_id']}:{metadata.get('attempt_id', raw.stem)}:a{anchor}:H{horizon}"
    return dict(sample_id=identity, raw_path=str(raw.resolve()),
                raw_sha256=metadata.get("_selection_raw_sha256"),
                attempt_id=metadata.get("attempt_id", raw.stem),
                family_id=metadata["family_id"], letter=metadata["letter"],
                task=metadata["task"], split=metadata["split"],
                anchor=anchor, horizon=horizon,
                committed_future=_committed(metadata, anchor, horizon),
                model_action_compatible=bool(np.all(data["actions"][anchor - 3:anchor + horizon] <= 511.)),
                **metrics)


def _key(row, data):
    a, h = row["anchor"], row["horizon"]
    digest = hashlib.sha256()
    for value in (row["family_id"], row["attempt_id"], str(h)):
        digest.update(value.encode("utf-8")); digest.update(b"\0")
    digest.update(data["rgb"][a - 3:a + h + 1].tobytes())
    digest.update(data["actions"][a - 3:a + h].astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _thin_static_runs(rows, representatives=3):
    """Retain start/middle/end of long *entirely unchanged* contexts.

    Distinct pusher/object motion is never thinned.  This is deterministic index
    selection, not repeated weighted sampling, and does not delete raw records.
    """
    output, run = [], []
    def flush():
        if run:
            keep = set(np.linspace(0, len(run) - 1, min(representatives, len(run)), dtype=int).tolist())
            output.extend(row for i, row in enumerate(run) if i in keep)
            run.clear()
    for row in rows:
        if row["exact_quiet_context"]:
            if run and row["anchor"] != run[-1]["anchor"] + 1:
                flush()
            run.append(row)
        else:
            flush(); output.append(row)
    flush()
    return output


def distribution(values):
    values = np.asarray(values, np.float64)
    return dict(N=len(values), **dict(zip(("P10", "P50", "P90", "max"),
                map(float, np.quantile(values, [.1, .5, .9, 1.]))))) if len(values) else dict(
                    N=0, P10=None, P50=None, P90=None, max=None)


def aggregate(rows):
    keys = ("translation_imagepx", "cumulative_translation_imagepx", "abs_rotation_deg",
            "cumulative_rotation_deg", "pusher_translation_imagepx")
    return dict(windows=len(rows), families=len({r["family_id"] for r in rows}),
                letters=dict(Counter(r["letter"] for r in rows)),
                events=dict(Counter(r["event"] for r in rows)),
                contact_positive_windows=sum(r["contact_positive"] for r in rows),
                object_wall_contact_windows=sum(r["object_wall_contact_positive"] for r in rows),
                diagnostic_contact_and_motion_windows=sum(r["diagnostic_contact_and_motion"] for r in rows),
                obvious_translation_windows=sum(r["obvious_translation"] for r in rows),
                obvious_rotation_windows=sum(r["obvious_rotation"] for r in rows),
                clockwise_rotation_windows=sum(r["signed_rotation_deg"] >= .5 for r in rows),
                counterclockwise_rotation_windows=sum(r["signed_rotation_deg"] <= -.5 for r in rows),
                bitstatic_future_windows=sum(r["bitstatic_future"] for r in rows),
                exact_quiet_context_windows=sum(r["exact_quiet_context"] for r in rows),
                **{key: distribution([r[key] for r in rows]) for key in keys})


def _grouped(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups["/".join(str(row[key]) for key in fields)].append(row)
    return {key: aggregate(value) for key, value in sorted(groups.items())}


def select(root: Path, config: dict) -> dict:
    root = Path(root).resolve()
    attempt_root = root / "collection" / "attempts"
    output = root / "selection"
    if (output / "summary.json").exists():
        existing = read(output / "summary.json")
        if all(sha(output / name) == value for name, value in existing["index_sha256"].items()):
            return existing
        raise ValueError("An already frozen selection index changed")
    metadata_paths = sorted(attempt_root.glob("*.json"))
    sources, raw_missing, families = [], [], {}
    natural, train, val = [], [], []
    duplicate_count = static_thinned = 0
    per_h_action_excluded = Counter()
    seen_keys = {}
    actual_contact = dict(observed_seconds=0., positive_ao_seconds=0., positive_ow_seconds=0.,
                          attempts_with_telemetry=0, attempts_without_telemetry=0)
    suspicious = Counter()
    for meta_path in metadata_paths:
        metadata = read(meta_path)
        # Claim/worker/status JSON is never a trajectory metadata document.
        if not all(key in metadata for key in ("family_id", "letter", "task", "split")):
            continue
        if metadata["letter"] not in LETTERS or metadata["split"] not in ("train", "seen_val"):
            raise ValueError("A held-out letter or unapproved split entered this run")
        family = str(metadata["family_id"])
        if family in families and families[family] != metadata["split"]:
            raise ValueError("One scene family crosses training/validation splits")
        families[family] = metadata["split"]
        raw = meta_path.with_suffix(".npz")
        if not raw.exists():
            raw_missing.append(dict(attempt_id=metadata.get("attempt_id", meta_path.stem),
                                    status=metadata.get("status"), metadata=str(meta_path)))
            continue
        data = load_attempt(raw, metadata)
        centroid = centers(metadata["letter"], data["states"])
        metadata["_selection_raw_sha256"] = sha(raw)
        sources.append(dict(raw_path=str(raw), raw_sha256=metadata["_selection_raw_sha256"],
                            metadata_path=str(meta_path), metadata_sha256=sha(meta_path),
                            family_id=family, split=metadata["split"],
                            actual_controls=len(data["actions"]), clean_controls=data["clean"]))
        for event in metadata.get("events", []):
            if isinstance(event, dict):
                suspicious[str(event.get("outcome", event.get("status", event.get("kind", "unknown"))))] += 1
        contact = data["contact_totals"]
        actual_contact["attempts_with_telemetry" if contact["telemetry_available"] else "attempts_without_telemetry"] += 1
        for key in ("observed_seconds", "positive_ao_seconds", "positive_ow_seconds"):
            actual_contact[key] += contact[key]
        attempt_train = []
        for horizon in HORIZONS:
            for anchor in range(3, data["clean"] - horizon + 1):
                if not data["valid"][anchor - 3:anchor + horizon].all():
                    continue
                row = _row(metadata, raw, data, centroid, anchor, horizon)
                natural.append(row)
                if not row["model_action_compatible"]:
                    per_h_action_excluded[horizon] += 1
                    continue
                if horizon == 1 and metadata["split"] == "train":
                    key = _key(row, data)
                    if key in seen_keys:
                        seen_keys[key]["source_occurrences"] += 1
                        duplicate_count += 1
                        continue
                    row["source_occurrences"] = 1
                    seen_keys[key] = row
                    attempt_train.append(row)
                elif horizon == 16 and metadata["split"] == "seen_val" and row["committed_future"]:
                    # Validation remains naturally distributed, no train-based selection.
                    val.append(row)
        retained = _thin_static_runs(attempt_train)
        static_thinned += len(attempt_train) - len(retained)
        train.extend(retained)
    if not sources:
        raise ValueError("No new saved trajectories were available for selection")
    if not train:
        raise ValueError("No complete clean model-compatible training windows")
    meaningful_motion = any(row["moving"] and row["split"] == "train" and
                            row["model_action_compatible"] for row in natural)
    train.sort(key=lambda r: (r["family_id"], r["attempt_id"], r["anchor"]))
    val.sort(key=lambda r: (r["family_id"], r["attempt_id"], r["anchor"]))
    train_families = {row["family_id"] for row in train}
    val_families = {row["family_id"] for row in val}
    if train_families & val_families:
        raise ValueError("Selected family overlap")
    actual_contact["positive_ao_ratio_of_observed_native_time"] = (
        actual_contact["positive_ao_seconds"] / actual_contact["observed_seconds"]
        if actual_contact["observed_seconds"] else None)
    summary = dict(kind="accelerated-clean-rgb-selection.v1", run_id=config.get("run_id"),
        ready_for_training=meaningful_motion,
        blocked_reason=None if meaningful_motion else "No meaningful new object motion in clean model-compatible training prefixes",
        training_eligible=True, old_data_relabelled=False, old_data_used=False,
        selected_train_windows=len(train), selected_seen_val_h16_windows=len(val),
        selected_train_families=len(train_families), selected_seen_val_families=len(val_families),
        train_families=sorted(train_families), seen_val_families=sorted(val_families),
        raw_trajectory_files=len(sources), raw_missing_attempts=raw_missing,
        total_planned_family_split_observed=len(families), source_records=sources,
        clean_controls=sum(row["clean_controls"] for row in sources),
        actual_controls_in_raw=sum(row["actual_controls"] for row in sources),
        recovered_and_other_suspicious_events_quarantined=dict(suspicious),
        exact_duplicate_train_windows_removed=duplicate_count,
        entirely_static_train_contexts_thinned=static_thinned,
        static_selection_rule="Dedup exact family/attempt/H/history/future/actions; keep at most start/middle/end of each entirely unchanged contiguous context run; retain every distinct object/pusher-motion window. No oversampling.",
        action_contract=dict(physical_input_domain=[0,512], model_input_domain=[0,511],
            normalization="existing a/511*2-1 unchanged", clipped=False,
            excluded_clean_windows_by_horizon={str(h): per_h_action_excluded[h] for h in HORIZONS}),
        model_inputs="RGB and actions only; saved state/contact/family/letter are privileged selection/audit metadata",
        exposure_rule="Each selected unique training row exactly once during this one new-data epoch; no weighted replacement",
        window_rule="RGB[a-3:a+1] history; actions[a-3:a+H]; future RGB[a+1:a+H+1]; no padding/reset/event crossing",
        quarantine_rule="Only controls [0,clean_control_count) are usable; first warning/hard control and entire later suffix excluded even if contact recovered",
        horizons=list(HORIZONS), natural_clean_by_horizon=_grouped(natural, ("horizon",)),
        model_compatible_clean_by_horizon=_grouped([r for r in natural if r["model_action_compatible"]], ("horizon",)),
        by_split_letter_task_horizon=_grouped(natural, ("split","letter","task","horizon")),
        by_letter_event_horizon=_grouped(natural, ("letter","event","horizon")),
        selected_train=aggregate(train), selected_train_by_letter_task=_grouped(train,("letter","task")),
        common_h16_seen_val=aggregate(val), common_h16_seen_val_by_letter=_grouped(val,("letter",)),
        actual_contact=actual_contact,
        motion_definition="Net/cumulative translation of actual shape area centroid in 96px units; net signed and cumulative absolute unwrapped physical angle in degrees. Not symmetry-canonical visual orientation.",
        diagnostic_motion_definition="contact AND (translation>=.1 imagepx OR abs rotation>=.5deg); report only, not sole admission rule",
        obvious_thresholds={str(h): dict(translation_imagepx=t,rotation_deg=r) for h,(t,r) in OBVIOUS.items()},
        caveats=["Natural windows overlap; window counts are not independent families.",
                 "Six already-seen letters only; no unseen-letter generalization claim.",
                 "Contact ratio is actual preserved native microstep time; unframed/missing telemetry must be reconciled with collection ledger.",
                 "No minimum coverage rate gate; absent boundary/recontact/strong rotation support is disclosed."])
    write(output / "train-index.json", train, root, config)
    write(output / "seen-val-index.json", val, root, config)
    summary["index_sha256"] = {name: sha(output / name) for name in ("train-index.json","seen-val-index.json")}
    write(output / "summary.json", summary, root, config)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = select(args.root, read(args.root / "runtime-config.json"))
    print(json.dumps({key:value[key] for key in ("ready_for_training","selected_train_windows",
                     "selected_seen_val_h16_windows","selected_train_families")}, ensure_ascii=False))
