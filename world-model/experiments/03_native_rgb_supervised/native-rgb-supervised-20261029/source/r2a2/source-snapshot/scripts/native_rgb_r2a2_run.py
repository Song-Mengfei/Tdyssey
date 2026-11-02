"""One separately authorized 64-slot paired physics probe, never a trainer.

No simulation step is allowed in prepare, tests, rendering or report generation.
The run entry points are single-use: gate (first pair), remaining (after gate).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

EXPERIMENT = ROOT / "artifacts/native-rgb-supervised-20261029"
OUT = EXPERIMENT / "r2a2"
REQUEST_SHA = "4a34ac57fe3583b64103a57dc951434a1c3e782c1d712137b58c2832c3c3a22a"
ARMS = ("D-half-life-0.1s", "D-half-life-0.3s")
DAMPINGS = (.0009765625, .09921256574801246)
SOURCE_FILES = ["scripts/native_rgb_r2a2_run.py", "scripts/native_rgb_r2a2_media.py",
    "scripts/native_rgb_r2a2_report.py", "tdyssey/native_rgb_r2a2_env.py",
    "tdyssey/native_rgb_env.py", "tdyssey/envs/official.py", "tdyssey/geometry.py",
    "tdyssey/letters.py", "tdyssey/native_rgb_motion_statistics.py",
    "tdyssey/native_rgb_supervised_sampling.py", "tdyssey/native_rgb_data.py",
    "tdyssey/collection.py", "tdyssey/state.py", "tdyssey/official.py", "tdyssey/source_export.py",
    "tests/test_native_rgb_r2a2_run.py", "tests/test_native_rgb_r2a2_env.py",
    "tdyssey/native_rgb_r2a_env.py", "scripts/native_rgb_r2a_report.py",
    "scripts/native_rgb_r2a_media.py", "scripts/native_rgb_r2a_run.py",
    "manifests/official_dp.json"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode()


def json_safe(value):
    """Keep failure evidence strict JSON without concealing nonfinite values."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NONFINITE:" + repr(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).resolve().relative_to(OUT.resolve()).as_posix()


def disk_files():
    old = read(EXPERIMENT / "r1/storage-and-resource-audit.json")
    paths = set()
    for raw in old["roots"]:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Previously audited artifact/source disappeared: {p}")
        if p.is_dir():
            paths.update(f.resolve() for f in p.rglob("*") if f.is_file())
        else:
            paths.add(p.resolve())
    for pattern in ("tdyssey/native_rgb*.py", "scripts/native_rgb*.py",
                    "tests/test_native_rgb*.py", "configs/native_rgb*.json"):
        paths.update(p.resolve() for p in ROOT.glob(pattern) if p.is_file())
    return paths


def storage_check(label, reserve_next_bytes=0):
    files = disk_files()
    total = sum(p.stat().st_size for p in files)
    probe = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    result = dict(at_utc=utc(), label=label, combined_bytes=total,
        atomic_reserve_bytes=256 * 1024**2, combined_limit_bytes=3 * 1024**3,
        r2a2_artifact_bytes=probe, r2a2_limit_bytes=128 * 1024**2,
        reserve_next_bytes=reserve_next_bytes, disk_free_bytes=shutil.disk_usage(OUT).free,
        files=len(files))
    result["passed"] = (total + result["atomic_reserve_bytes"] + reserve_next_bytes
        <= result["combined_limit_bytes"] and probe + reserve_next_bytes
        <= result["r2a2_limit_bytes"] and result["disk_free_bytes"] > 512 * 1024**2)
    if not result["passed"]:
        raise RuntimeError("Storage reserve/cap reached: " + json.dumps(result))
    return result


def verify_sources():
    for row in read(OUT / "source-identity.json")["files"]:
        if sha(row["source"]) != row["sha256"] or sha(OUT / row["snapshot"]) != row["sha256"]:
            raise RuntimeError("Frozen source/config changed: " + row["source"])
    for row in read(OUT / "prior-resource-identities.json")["files"]:
        if sha(row["path"]) != row["sha256"]:
            raise RuntimeError("Original physics/model resource ledger changed: " + row["path"])
    for row in read(OUT / "runtime-identity.json")["files"]:
        if sha(row["path"]) != row["sha256"]:
            raise RuntimeError("Installed physics runtime changed: " + row["path"])
    for row in read(OUT / "frozen-baseline-references.json")["parent_review_files"]:
        if sha(row["path"]) != row["sha256"]:
            raise RuntimeError("Frozen parent review artifact changed: " + row["path"])


def prepare():
    if (OUT / "resource-ledger.json").exists() or (OUT / "source-identity.json").exists():
        raise FileExistsError("R2A2 preparation already exists; no overwrite or second launch")
    state = read(EXPERIMENT / "stage-state.json")
    if not (state["current_stage"] == "R2A2" and state["r2a2_execution_authorized"]
            and state["budget"]["r2a2_new_controls_cap"] == 512):
        raise RuntimeError("Current state does not authorize R2A2-512")
    original = OUT / "approved-requests.json"
    if sha(original) != REQUEST_SHA:
        raise RuntimeError("The parent-approved proposal changed")
    requests = read(original)
    rows = [r for r in requests["requests"] if r["phase"] == "R2A2"]
    if len(rows) != 64 or sum(r["max_controls"] for r in rows) != 512:
        raise RuntimeError("Only the exact frozen64 R2A2 requests are authorized")
    for i in range(0, 64, 2):
        a, b = rows[i:i+2]
        if not (a["protocol"] == ARMS[0] and b["protocol"] == ARMS[1]
                and a["family_id"] == b["family_id"]
                and a["action_variant"] == b["action_variant"]
                and a["initial_parameters_sha256"] == b["initial_parameters_sha256"]
                and a["actions_absolute"] == b["actions_absolute"]
                and len(a["actions_absolute"]) == len(b["actions_absolute"]) == 8):
            raise RuntimeError("Approved protocol pair is not exactly matched")
        if any(not math.isfinite(x) or x < 0 or x > 511
               for action in a["actions_absolute"] for x in action):
            raise RuntimeError("Invalid frozen absolute target")
    shutil.copyfile(original, OUT / "frozen-requests.json")
    config = dict(kind="native-rgb-R2A2-512-paired-physics.v1", max_controls=512,
        slots=64, controls_per_slot=8, prior_supervised_controls=40192,
        prior_all_native_controls=74752, physics_source=str((ROOT.parent /
            "Tdyssey/third_party/diffusion_policy").resolve()),
        original_protocol_sha256="fd358dba8f26a616ec6c55b3f6d532681bf5036e2cf79fe5a467001e3a77ae96",
        candidate_protocols={ARMS[0]:dict(damping=DAMPINGS[0],half_life_seconds=.1),
            ARMS[1]:dict(damping=DAMPINGS[1],half_life_seconds=.3)},
        sole_physical_difference="space.damping only; two finite half-life candidates; originalD0/D1 offline only", force_limit=10000.,
        pre_step_speed_limit=200., kp=100., kv=20., control_dt=.1, physics_dt=.01,
        substeps=10, penetration_limit_world_px=2., request_sha256=REQUEST_SHA,
        model_calls=0, optimizer_updates=0, gpu_work=0, R2B_authorized=False,
        old_unused320_spent=False, replay_authorized=False, new_independent_families=0,
        reused_independent_families=16, baseline_protocols_not_rerun=True,
        relatively_sustained_contact_H1_substeps_gte=8,
        relatively_sustained_contact_H4_aggregate_substeps_gte=32,
        contact_group_scope="R2A2 descriptive groups, not sole acceptance and not retroactive R2A criteria",
        storage_candidate_limit=128 * 1024**2, shared_limit=3 * 1024**3,
        atomic_reserve=256 * 1024**2)
    atomic_json(OUT / "frozen-config.json", config)
    source_pairs = [(ROOT / name, name) for name in SOURCE_FILES]
    for name in ("diffusion_policy/env/pusht/pusht_env.py",
                 "diffusion_policy/env/pusht/pymunk_override.py"):
        source_pairs.append((Path(config["physics_source"]) / name, "official-source/" + name))
    for name in ("frozen-requests.json", "frozen-config.json", "PARENT_REVIEW.md",
                 "prior-resource-identities.json", "runtime-identity.json", "preflight-checks.json",
                 "frozen-baseline-references.json"):
        source_pairs.append((OUT / name, "R2A2-inputs/" + name))
    identities = []
    for source, name in source_pairs:
        dest = OUT / "source-snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(source.read_bytes())
        identities.append(dict(source=str(source.resolve()), snapshot=relative(dest),
            sha256=sha(source), bytes=source.stat().st_size))
    atomic_json(OUT / "source-identity.json", dict(kind="R2A2-source-before-first-control.v1",
        at_utc=utc(), files=identities))
    slots = [dict(slot_id=r["request_id"], request_id=r["request_id"],
        family_id=r["family_id"], action_variant=r["action_variant"],
        protocol=r["protocol"], letter=r["letter"], status="not_executed",
        attempted=False, initialization_attempted=False, charged_controls=0,
        completed_controls=0, actual_controls=0, partial_substeps=0,
        trajectory_file=f"trajectories/{r['request_id']}/trajectory.npz",
        metadata_file=f"trajectories/{r['request_id']}/metadata.json") for r in rows]
    atomic_json(OUT / "index.json", dict(kind="R2A2-all64-slot-registry.v1",
        collection_closed=False, status="prepared", gate_passed=False,
        planned_slots=64, planned_controls=512, planned_H1=320, planned_H4=128, slots=slots))
    atomic_json(OUT / "resource-ledger.json", dict(kind="R2A2-linked-physical-cost-ledger.v1",
        max_controls=512, prior_supervised_controls=40192, prior_all_native_controls=74752,
        charged_controls=0, completed_controls=0, executed_substeps=0, engine_step_calls=0,
        engine_call_errors_with_unknown_advance=0,
        pending_control=None, failures=[], closed=False, started_at_utc=None,
        model_calls=0, optimizer_attempts=0, GPU_work=0,
        no_old_budget_reset=True, old_unused320_spent=False))
    atomic_json(OUT / "pre-execution-storage.json", storage_check("prepared", 2*1024**2))
    print(json.dumps(dict(prepared=True, slots=64, new_control_cap=512, hidden_steps=0)))


def guard_charge(ledger, row, control):
    if (ledger["closed"] or ledger["pending_control"] is not None
            or ledger["charged_controls"] >= ledger["max_controls"]
            or row["charged_controls"] >= 8 or control != row["charged_controls"]):
        raise RuntimeError("No retry/over-budget/unresolved physical action is allowed")
    ledger["charged_controls"] += 1
    row["charged_controls"] += 1
    ledger["pending_control"] = dict(slot_id=row["slot_id"], control_index=control)


def physics_identity(identity):
    # Environment explicitly excludes arm/version/hash bookkeeping from physics.
    if "physics" not in identity:
        raise ValueError("Actual environment identity must expose physics")
    return identity["physics"]


def compare_pair(initials):
    import numpy as np
    a, b = initials
    if not np.array_equal(a["state"], b["state"]):
        raise RuntimeError("Actual paired initial state15 mismatch")
    if not np.array_equal(a["rgb"], b["rgb"]):
        raise RuntimeError("Actual paired initial raw RGB mismatch")
    pa = json.loads(json.dumps(physics_identity(a["identity"])))
    pb = json.loads(json.dumps(physics_identity(b["identity"])))
    da, db = pa["space"].pop("damping"), pb["space"].pop("damping")
    if da != DAMPINGS[0] or db != DAMPINGS[1] or pa != pb:
        raise RuntimeError("Actual non-damping physics parameters are not identical")
    return dict(actual_states_bitwise_equal=True, actual_initial_RGB_equal=True,
        non_damping_physics_equal=True, damping=[da, db],
        common_physics_sha256=hashlib.sha256(canonical(pa)).hexdigest(),
        state15_sha256=hashlib.sha256(a["state"].tobytes()).hexdigest())


def compare_old_baselines(initials, request):
    """Read existing P0/P1 only; never construct or integrate an old arm."""
    import numpy as np
    refs=read(OUT/"frozen-baseline-references.json")
    rows=[r for r in refs["slots"] if r["family_id"]==request["family_id"]
          and r["action_variant"]==request["action_variant"]]
    if len(rows)!=2 or {r["protocol"] for r in rows}!={"P0-frozen","P1-damping1-candidate"}:
        raise RuntimeError("Missing frozen old P0/P1 references for the exact scene/action")
    common=json.loads(json.dumps(physics_identity(initials[0]["identity"])))
    common["space"].pop("damping")
    proof=[]
    for row in rows:
        if sha(row["trajectory_file"])!=row["trajectory_sha256"] or sha(row["metadata_file"])!=row["metadata_sha256"]:
            raise RuntimeError("Old P0/P1 arrays/metadata changed")
        meta=read(row["metadata_file"])
        with np.load(row["trajectory_file"],allow_pickle=False) as z:
            if not np.array_equal(z["actions"],np.asarray(request["actions_absolute"],dtype=np.float64)):
                raise RuntimeError("New targets differ from actually executed old P0/P1 actions")
            for initial in initials:
                if not np.array_equal(z["states"][0],initial["state"]) or not np.array_equal(z["rgb"][0],initial["rgb"]):
                    raise RuntimeError("New actual initial state/RGB differs from old P0/P1")
        physics=json.loads(json.dumps(physics_identity(meta["initial_identity"])))
        old_damping=physics["space"].pop("damping")
        if old_damping!=(0. if row["protocol"]=="P0-frozen" else 1.) or physics!=common:
            raise RuntimeError("New actual non-damping parameters differ from old P0/P1")
        proof.append(dict(request_id=row["request_id"],protocol=row["protocol"],
            actual_initial_state_and_RGB_equal=True,actual_actions_equal=True,
            non_damping_physics_equal=True,trajectory_sha256=row["trajectory_sha256"],metadata_sha256=row["metadata_sha256"]))
    if request["source_request_id"] not in {r["request_id"] for r in rows if r["protocol"]=="P0-frozen"}:
        raise RuntimeError("Source request identity is not the matching old P0")
    return dict(old_P0_P1_offline_only=True,all_four_actual_initial_states_and_RGB_equal=True,
        all_four_actual_actions_equal=True,all_four_non_damping_physics_equal=True,baseline_proofs=proof)


def control_record(info, substeps, request, control, completed):
    import numpy as np
    categories = {"agent-object": set(), "object-wall": set(), "agent-wall": set()}
    arcs = set()
    point_samples=[]
    for i, sub in enumerate(substeps):
        for contact in sub.get("contacts", []):
            category = str(contact.get("category", contact.get("kind"))).replace("_", "-")
            if category in categories:
                categories[category].add(i)
            if category == "agent-object":
                for arc in contact.get("actual_contact_arc_ids", contact.get("arc_ids", [])):
                    arcs.add(str(arc))
                for point in contact.get("points", []):
                    if point.get("arc_id") is not None:
                        arcs.add(str(point["arc_id"]))
                    if point.get("object_world") is not None and contact.get("phase")=="post_solve":
                        point_samples.append(dict(substep_index=i,phase=contact.get("phase"),
                            object_world=point["object_world"],object_local=point.get("object_local"),
                            arc_ids=point.get("arc_ids",[])))
    p = [s.get("penetrations", s.get("penetration")) for s in substeps]
    unknown_penetrations=any(v is None or len(v)!=3 or any(x is None for x in v) for v in p)
    maxima = ([None,None,None] if unknown_penetrations else
        np.max(np.asarray(p, dtype=float), axis=0).tolist() if p else [0., 0., 0.])
    return dict(control_index=control, completed=bool(completed),
        valid_transition=bool(completed and info.get("valid", True)),
        valid=bool(completed and info.get("valid", True)), substep_count=len(substeps),
        max_penetrations=maxima, max_substep_penetrations=maxima,
        penetration=maxima, unknown_penetrations=unknown_penetrations,
        contact=len(categories["agent-object"]),
        agent_object_contact_substeps=len(categories["agent-object"]),
        object_wall_contact_substeps=len(categories["object-wall"]),
        agent_wall_contact_substeps=len(categories["agent-wall"]),
        actual_contact_arc_ids=sorted(arcs), contact_arc_bins=sorted(arcs),
        actual_contact_point_samples=point_samples,
        action_phase=request["action_schedule"][control],
        action_absolute=request["actions_absolute"][control])


def save_slot(row, meta, states, actions, images, attempted_actions, telemetry,
              partial_states=None, partial_rgb=None):
    import numpy as np
    directory = OUT / "trajectories" / row["slot_id"]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "trajectory.npz"
    with target.with_suffix(".npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, states=np.asarray(states,dtype=np.float64).reshape(-1,15),
            actions=np.asarray(actions,dtype=np.float64).reshape(-1,2),
            rgb=np.asarray(images,dtype=np.uint8).reshape(-1,3,96,96),
            attempted_actions=np.asarray(attempted_actions,dtype=np.float64).reshape(-1,2),
            partial_states=np.asarray(partial_states if partial_states is not None else [],dtype=np.float64).reshape(-1,15),
            partial_rgb=np.asarray(partial_rgb if partial_rgb is not None else [],dtype=np.uint8))
    os.replace(target.with_suffix(".npz.tmp"),target)
    logs = directory / "telemetry.json.gz"
    with gzip.open(logs.with_suffix(".gz.tmp"), "wt", encoding="utf-8") as stream:
        json.dump(json_safe(telemetry),stream,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    os.replace(logs.with_suffix(".gz.tmp"),logs)
    meta.update(status=row["status"], charged_controls=row["charged_controls"],
        completed_controls=row["completed_controls"], actual_controls=row["actual_controls"],
        partial_substeps=row["partial_substeps"],
        frame_sha256=[hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest() for im in images],
        frame_hash_layout="uint8 CHW C-order raw bytes", trajectory_sha256=sha(target),
        telemetry_file=relative(logs),telemetry_sha256=sha(logs), physical_span_seconds=len(actions)*.1,
        partial_control_span_seconds=(row["partial_substeps"]*.01 if meta.get("partial_time_exact",True) else None),
        partial_engine_call_allowance_seconds=row["partial_substeps"]*.01,
        final_endpoint_display_hold_seconds=.1, midtrajectory_resets=0)
    atomic_json(directory / "metadata.json",meta)
    row.update(trajectory_sha256=sha(target),metadata_sha256=sha(directory / "metadata.json"))


def check_saved(row):
    import numpy as np
    meta=read(OUT/row["metadata_file"])
    with np.load(OUT/row["trajectory_file"],allow_pickle=False) as z:
        n=len(z["actions"])
        if (z["states"].shape!=(n+1,15) or z["rgb"].shape!=(n+1,3,96,96)
                or z["states"].dtype!=np.float64 or z["actions"].dtype!=np.float64):
            raise RuntimeError("Ncontrols/N+1frames/state/action alignment mismatch")
        if not np.array_equal(z["actions"],np.asarray(meta["request"]["actions_absolute"][:n],dtype=np.float64)):
            raise RuntimeError("Saved actions differ from frozen executed absolute targets")
        if any(hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()!=expected
               for f,expected in zip(z["rgb"],meta["frame_sha256"])):
            raise RuntimeError("Raw frame SHA mismatch")
    with gzip.open(OUT/meta["telemetry_file"],"rt",encoding="utf-8") as stream:
        logs=json.load(stream)
    if len(logs)!=row["charged_controls"]:
        raise RuntimeError("Every charged control must have its log, including failures")
    for i,log in enumerate(logs):
        if log["completed"] and (len(log["substeps"])!=10 or log["action"]!=meta["request"]["actions_absolute"][i]):
            raise RuntimeError("Control100Hz substeps/action identity mismatch")
        for j,sub in enumerate(log["substeps"]):
            if sub.get("dt",sub.get("dt_seconds",.01))!=.01:
                raise RuntimeError("Physical timestep changed")
            if j and sub["pre_state"]!=log["substeps"][j-1]["post_state"]:
                raise RuntimeError("Substep state continuity mismatch")
    return dict(slot_id=row["slot_id"],states=n+1,full_controls=n,
        raw_frames=n+1,charged=row["charged_controls"],substeps=sum(len(x["substeps"]) for x in logs),passed=True)


def run(mode):
    import numpy as np
    from tdyssey.native_rgb_r2a2_env import make_r2a2_env
    verify_sources()
    state=read(EXPERIMENT/"stage-state.json")
    index=read(OUT/"index.json");ledger=read(OUT/"resource-ledger.json")
    if state["current_stage"]!="R2A2" or not state["r2a2_execution_authorized"]:
        raise RuntimeError("No current R2A2 authorization")
    if ledger["closed"] or ledger["pending_control"] is not None or index["collection_closed"]:
        raise RuntimeError("Closed/failed/interrupted probe cannot resume or replay")
    if mode=="gate" and (ledger["charged_controls"]!=0 or index["status"]!="prepared"):
        raise RuntimeError("The initial integrity pair can execute only once")
    if mode=="remaining" and not (index["gate_passed"] and index["status"]=="gate_passed"):
        raise RuntimeError("Remaining frozen slots require the successful first-pair gate")
    if mode=="remaining" and not read(OUT/"gate-once-claim.json").get("finished_successfully"):
        raise RuntimeError("The initial gate process must have finished before remaining work")
    claim_path=OUT/(mode+"-once-claim.json")
    claim=dict(mode=mode,at_utc=utc(),pid=os.getpid(),finished_successfully=False,
        no_restart_or_replay=True)
    with claim_path.open("x",encoding="utf-8") as stream:
        json.dump(claim,stream,indent=2)
    config=read(OUT/"frozen-config.json");proposal=read(OUT/"frozen-requests.json")
    requests={r["request_id"]:r for r in proposal["requests"] if r["phase"]=="R2A2"}
    families={f["family_id"]:f for f in proposal["scene_families"] if f["phase"]=="R2A2"}
    ledger["started_at_utc"]=ledger["started_at_utc"] or utc()
    state["status"]="running_R2A2_gate" if mode=="gate" else "running_R2A2_remaining"
    atomic_json(EXPERIMENT/"stage-state.json",state)
    pair_start,pair_end=(0,2) if mode=="gate" else (2,64)
    current_row=None;halt=None
    try:
        for start in range(pair_start,pair_end,2):
            verify_sources();storage=storage_check(f"before-pair-{start//2}",2*1024**2)
            pair_rows=index["slots"][start:start+2]
            initials=[]
            for row in pair_rows:
                current_row=row
                if row["status"]!="not_executed":raise RuntimeError("No slot can be replaced/replayed")
                row.update(attempted=True,initialization_attempted=True,status="initializing")
                atomic_json(OUT/"index.json",index)
                req=requests[row["request_id"]];fam=families[req["family_id"]]
                env=make_r2a2_env(config["physics_source"],req["letter"],req["protocol"])
                env.seed(fam["seed"])
                rgb=env.reset_scene(**fam["initialization"]["reset_scene_arguments"])
                state0=env.state().copy();identity=env.initial_identity()
                if (state0.shape!=(15,) or not np.isfinite(state0).all()
                        or not np.allclose(state0,fam["initialization"]["expected_initial_state15"],rtol=0,atol=1e-9)
                        or np.any(state0[5:]!=0) or env.agent.angular_velocity!=0
                        or env.agent.torque!=0 or np.max(env.penetrations())>.1
                        or np.asarray(rgb).shape!=(3,96,96)):
                    raise RuntimeError("Actual initial state/zero velocities/force/geometry differs from proposal")
                meta=dict(request=req,family=fam,initial_identity=identity,
                    actual_initial_state=state0.tolist(),actual_reset_arguments=fam["initialization"]["reset_scene_arguments"],
                    protocol_sha256=identity.get("protocol_sha256"),initial_storage=storage,
                    control_records=[],initial_rgb_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
                    geometry_state_is_audit_only=True)
                row["status"]="initialized_not_executed"
                save_slot(row,meta,[state0],[],[rgb],[],[])
                initials.append(dict(env=env,state=state0,rgb=np.asarray(rgb),identity=identity,meta=meta))
            comparison=compare_pair(initials)
            comparison.update(compare_old_baselines(initials,requests[pair_rows[0]["request_id"]]))
            atomic_json(OUT/"pairs"/(f"pair-{start//2:02d}-initial-check.json"),comparison)
            for row,initial in zip(pair_rows,initials):
                current_row=row;req=requests[row["request_id"]];env=initial["env"];meta=initial["meta"]
                meta["actual_pair_initial_check"]=comparison
                states=[initial["state"]];images=[initial["rgb"]];actions=[];attempted=[];telemetry=[]
                row["status"]="running";partial_states=[];partial_rgb=None
                for control,action in enumerate(req["actions_absolute"]):
                    guard_charge(ledger,row,control)
                    atomic_json(OUT/"resource-ledger.json",ledger);atomic_json(OUT/"index.json",index)
                    attempted.append(action);info={};completed=False
                    try:
                        rgb,info=env.step(np.asarray(action,dtype=np.float64))
                        if not info["valid"] or info["physics_substeps"]!=10 or env.control_step!=control+1:
                            raise RuntimeError("Full control physical validity/timing mismatch")
                        completed=True
                    except Exception as exc:
                        steps=env.last_substeps
                        partial_states=[sub["post_state"] for sub in steps if sub.get("post_state") is not None]
                        meta["partial_state_substep_indices"]=[i for i,sub in enumerate(steps) if sub.get("post_state") is not None]
                        meta["unavailable_partial_state_substeps"]=[i for i,sub in enumerate(steps) if sub.get("post_state") is None]
                        meta["partial_time_exact"]=not any(sub.get("engine_exception") for sub in steps)
                        row.update(status="failed_partial_control",partial_substeps=len(steps),error=repr(exc))
                        meta["failure"]={"control_index":control,"reason":repr(exc),"at_utc":utc()}
                        if np.isfinite(env.state()).all():
                            try:partial_rgb=env.render_clean()
                            except Exception as render_exc:
                                meta["failure"]["partial_render_error"]=repr(render_exc)
                        raise
                    finally:
                        steps=env.last_substeps
                        row["actual_controls"]+=int(bool(steps))
                        ledger["engine_step_calls"]+=len(steps)
                        ledger["executed_substeps"]+=sum(bool(sub.get("executed")) and not sub.get("engine_exception") for sub in steps)
                        ledger["engine_call_errors_with_unknown_advance"]+=sum(bool(sub.get("engine_exception")) for sub in steps)
                        record=control_record(info,steps,req,control,completed)
                        meta["control_records"].append(record)
                        telemetry.append(dict(control_index=control,action=action,completed=completed,substeps=steps))
                        if completed:
                            states.append(env.state().copy());images.append(rgb);actions.append(action)
                            row["completed_controls"]+=1;ledger["completed_controls"]+=1
                            ledger["pending_control"]=None
                        save_slot(row,meta,states,actions,images,attempted,telemetry,partial_states,partial_rgb)
                        atomic_json(OUT/"resource-ledger.json",ledger);atomic_json(OUT/"index.json",index)
                row["status"]="completed";save_slot(row,meta,states,actions,images,attempted,telemetry)
                check=check_saved(row);atomic_json(OUT/"checks"/(row["slot_id"]+".json"),check)
                row["post_storage"]=storage_check("after-"+row["slot_id"])
                atomic_json(OUT/"index.json",index)
                print(json.dumps(dict(slot=row["slot_id"],status=row["status"],charged=ledger["charged_controls"],substeps=ledger["executed_substeps"])),flush=True)
            if start==0:
                gate=dict(passed=True,at_utc=utc(),actual_initial_pair=comparison,
                    saved_checks=[check_saved(r) for r in pair_rows],charged_controls=ledger["charged_controls"],
                    executed_substeps=ledger["executed_substeps"],no_additional_test_controls=True)
                if gate["charged_controls"]!=16 or gate["executed_substeps"]!=160:
                    raise RuntimeError("First-pair budget/log integrity failure")
                atomic_json(OUT/"gate-integrity-check.json",gate)
                index.update(gate_passed=True,status="gate_passed")
                atomic_json(OUT/"index.json",index)
    except BaseException as exc:
        halt=dict(at_utc=utc(),reason=repr(exc),slot_id=current_row["slot_id"] if current_row else None,
            no_restart_or_replacement=True)
        ledger["failures"].append(halt)
        if current_row is not None and current_row["status"] not in ("failed_partial_control","completed"):
            current_row.update(status="initialization_or_IO_failure",error=repr(exc))
        index.update(status="stopped_on_first_error",collection_closed=True,halt=halt)
        ledger.update(closed=True,stopped_at_utc=utc())
        atomic_json(OUT/"halt.json",halt)
    else:
        if mode=="remaining":
            index.update(status="completed_all64",collection_closed=True)
            ledger.update(closed=True,stopped_at_utc=utc())
    finally:
        ledger["supervised_plus_charged_controls"]=40192+ledger["charged_controls"]
        ledger["all_native_plus_charged_controls"]=74752+ledger["charged_controls"]
        ledger["supervised_plus_completed_controls"]=40192+ledger["completed_controls"]
        ledger["all_native_plus_completed_controls"]=74752+ledger["completed_controls"]
        atomic_json(OUT/"resource-ledger.json",ledger);atomic_json(OUT/"index.json",index)
        state=read(EXPERIMENT/"stage-state.json")
        state.update(status="R2A2_cold_ready_for_report" if index["collection_closed"] else "R2A2_gate_cold_pending_integrity_review",
            r2a2_collection_closed=index["collection_closed"],r2a2_gate_passed=index["gate_passed"],
            next_action="Offline evidence/videos/report only then actual standing-authorized callback and STOP" if index["collection_closed"] else "Inspect saved first-pair gate; only if passed execute remaining frozen R2A2 slots once")
        state["resources"]["r2a2_charged_controls"]=ledger["charged_controls"]
        state["resources"]["r2a2_completed_controls"]=ledger["completed_controls"]
        state["resources"]["r2a2_executed_substeps"]=ledger["executed_substeps"]
        atomic_json(EXPERIMENT/"stage-state.json",state)
    verify_sources()
    claim.update(finished_at_utc=utc(),finished_successfully=halt is None,
        collection_status=index["status"],charged_controls=ledger["charged_controls"])
    atomic_json(claim_path,claim)
    print(json.dumps(dict(status=index["status"],collection_closed=index["collection_closed"],
        charged=ledger["charged_controls"],completed=ledger["completed_controls"],substeps=ledger["executed_substeps"],
        torch_imported="torch" in sys.modules,model_calls=0,GPU_work=0)),flush=True)
    if halt: return 2
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","gate","remaining"])
    args=parser.parse_args()
    if args.mode=="prepare":prepare();return 0
    return run(args.mode)


if __name__=="__main__":
    raise SystemExit(main())
