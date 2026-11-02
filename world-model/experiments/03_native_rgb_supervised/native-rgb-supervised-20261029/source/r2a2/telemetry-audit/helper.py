"""Closed-file R2A2 decay audit; old R2A inputs are read-only references.

No simulator, renderer, model, optimizer or dataset iterator is constructed.
"""
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
ARMS = {"P0-frozen": 0., "P1-damping1-candidate": 1.,
    "D-half-life-0.1s": .0009765625, "D-half-life-0.3s": .09921256574801246}
ENTRY_EPS = 1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            d.update(chunk)
    return d.hexdigest()


def stats(values):
    values = list(values)
    a = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    out = dict(n=len(values), finite=len(a), unknown_or_nonfinite=len(values)-len(a))
    if len(a):
        out.update(min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
            **{f"p{q}": float(np.percentile(a, q)) for q in (5, 50, 90, 95, 99)})
    return out


def vec(value, n):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return a if a.shape == (n,) and np.isfinite(a).all() else None


def diff(a, b):
    return float(np.max(np.abs(a-b)))


def kind(event):
    return event.get("kind", event.get("category", "unknown")).replace("-", "_")


def ratio_record(start, end, omega_start, omega_end, elapsed, damping):
    speed = float(np.linalg.norm(start))
    angular = abs(float(omega_start))
    result = dict(elapsed_seconds=elapsed, expected_retention=damping**elapsed,
        linear_entry_speed=speed, angular_entry_abs_speed=angular,
        linear_entry_class="exact_zero" if speed == 0. else "near_zero_nonzero" if speed <= ENTRY_EPS else "nonzero",
        angular_entry_class="exact_zero" if angular == 0. else "near_zero_nonzero" if angular <= ENTRY_EPS else "nonzero",
        linear_vector_residual_norm=float(np.linalg.norm(end-start*(damping**elapsed))),
        angular_signed_residual=float(omega_end-omega_start*(damping**elapsed)),
        linear_retention=None, linear_projected_retention=None, angular_signed_retention=None,
        linear_observed_half_life_seconds=None, angular_observed_half_life_seconds=None)
    if speed > ENTRY_EPS:
        q = float(np.linalg.norm(end)/speed)
        result.update(linear_retention=q, linear_projected_retention=float(end@start/(speed*speed)))
        if 0. < q < 1.:
            result["linear_observed_half_life_seconds"] = elapsed*math.log(.5)/math.log(q)
    if angular > ENTRY_EPS:
        q = float(omega_end/omega_start)
        result["angular_signed_retention"] = q
        if 0. < q < 1.:
            result["angular_observed_half_life_seconds"] = elapsed*math.log(.5)/math.log(q)
    return result


def audit(base):
    base = Path(base).resolve()
    metrics = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(Counter)
    ratios = defaultdict(lambda: defaultdict(list))
    impulses = defaultdict(lambda: defaultdict(list))
    checks, anomalies, inputs, slots, episodes = {}, [], [], [], []
    for cohort in (base.parent/"r2a", base):
        index, ledger = read(cohort/"index.json"), read(cohort/"resource-ledger.json")
        if index.get("collection_closed") is not True or ledger.get("closed") is not True:
            raise RuntimeError("Refusing hot collection files; no polling/waiting")
        source_path = cohort/"source-identity.json"
        source = read(source_path)
        for row in source["files"]:
            for path in (Path(row["source"]), cohort/row["snapshot"]):
                if sha(path) != row["sha256"]:
                    raise ValueError(f"Frozen source mismatch: {path}")
        inputs.append(dict(cohort=cohort.name, index_sha256=sha(cohort/"index.json"),
            ledger_sha256=sha(cohort/"resource-ledger.json"), source_identity_sha256=sha(source_path)))
        cohort_steps = cohort_logs = 0
        for slot in index["slots"]:
            if not slot.get("initialization_attempted"):
                slots.append(dict(cohort=cohort.name, slot=slot["slot_id"], status=slot["status"]))
                continue
            meta_path = cohort/slot["metadata_file"]
            if not meta_path.exists():
                anomalies.append(dict(slot=slot["slot_id"], reason="missing attempted-slot metadata"))
                continue
            if sha(meta_path) != slot["metadata_sha256"]:
                raise ValueError("Metadata SHA mismatch")
            meta = read(meta_path)
            logs_path, npz_path = cohort/meta["telemetry_file"], cohort/slot["trajectory_file"]
            if sha(logs_path) != meta["telemetry_sha256"] or sha(npz_path) != meta["trajectory_sha256"]:
                raise ValueError("Telemetry/trajectory SHA mismatch")
            with gzip.open(logs_path, "rt", encoding="utf-8") as stream:
                logs = json.load(stream)
            req, physics = meta["request"], meta["initial_identity"]["physics"]
            arm, family = req["protocol"], req["family_id"]
            damping = float(physics["space"]["damping"])
            if arm not in ARMS or damping != ARMS[arm]:
                raise ValueError("Unexpected candidate; no damping search is allowed")
            gravity = np.asarray(physics["space"]["gravity"])
            agent_mass, object_mass = float(physics["agent"]["mass"]), float(physics["object"]["mass"])
            moment = physics["object"]["moment"]
            kp, kv = float(physics["kp"]), float(physics["kv"])
            force_cap = float(physics["base_control_protocol"]["force_limit"])
            speed_cap = float(physics["base_control_protocol"]["pre_step_speed_limit"])
            counts[arm]["slots"] += 1
            counts[arm]["charged_controls"] += len(logs)
            counts[arm]["completed_controls"] += int(slot["completed_controls"])
            cohort_logs += len(logs)
            run, previous_kinds, contact_history = [], set(), []
            last_post = None

            def finish_run():
                if not run:
                    return
                first, last = run[0], run[-1]
                row = ratio_record(first["entry"][7:9], last["post"][7:9], first["entry"][9], last["post"][9],
                    len(run)*.01, damping)
                row.update(arm=arm, family=family, letter=req["letter"], slot=slot["slot_id"],
                    substeps=len(run), start_control=first["control"], start_substep=first["substep"],
                    end_control=last["control"], end_substep=last["substep"],
                    after_contact_kinds=first["preceding_kinds"],
                    observed_detachment=bool(set(first["preceding_kinds"]).intersection(("agent_object", "object_wall"))))
                episodes.append(row)
                ratios[arm]["free_runs"].append(row)
                run.clear()

            for log in logs:
                action, control = np.asarray(log["action"]), int(log["control_index"])
                control_free, control_valid = [], True
                if len(log["substeps"]) != 10 and log.get("completed"):
                    anomalies.append(dict(slot=slot["slot_id"], control=control, reason="completed control lacks10substeps"))
                for sub in log["substeps"]:
                    ref = dict(slot=slot["slot_id"], control=control, substep=sub["index"])
                    counts[arm]["substeps"] += 1
                    cohort_steps += 1
                    contacts = sub.get("contacts", [])
                    kinds = {kind(event) for event in contacts}
                    contact_history.append("agent_object" in kinds)
                    pre, entry, post = [vec(sub.get(key), 15) for key in ("pre_state", "integrator_pre_state", "post_state")]
                    pen = vec(sub.get("penetrations"), 3)
                    if any(value is None for value in (pre, entry, post, pen)) or not sub.get("executed") or sub.get("engine_exception"):
                        anomalies.append(dict(**ref, reason="unknown/nonfinite state or engine advance"))
                        control_valid = False
                        finish_run()
                        previous_kinds, last_post = kinds, None
                        continue
                    if float(sub["dt"]) != .01 or (last_post is not None and not np.array_equal(last_post, pre)):
                        anomalies.append(dict(**ref, reason="time/state continuity mismatch"))
                    last_post = post
                    raw = agent_mass*(kp*(action-pre[:2])-kv*pre[5:7])
                    expected_force = raw.copy()
                    force_norm = float(np.linalg.norm(raw))
                    if force_norm > force_cap:
                        expected_force *= force_cap/force_norm
                    expected_velocity = pre[5:7].copy()
                    speed = float(np.linalg.norm(expected_velocity))
                    if speed > speed_cap:
                        expected_velocity *= speed_cap/speed
                    measured_force = np.asarray(sub["clipped_body_force"])
                    errors = dict(force_unclipped=diff(raw, np.asarray(sub["exact_formula_reconstruction"]["unclipped_force"])),
                        force_clipped=diff(expected_force, measured_force),
                        cap_velocity=diff(expected_velocity, np.asarray(sub["velocity_after_cap"])),
                        cap_entry=diff(entry[5:7], np.asarray(sub["velocity_after_cap"])),
                        force_entry=diff(entry[10:12], measured_force))
                    if max(errors.values()) > 1e-8:
                        anomalies.append(dict(**ref, reason="force/cap mismatch", errors=errors))
                    for key, value in errors.items():
                        metrics[arm][key+"_maxabs_error"].append(value)
                    counts[arm]["force_clipped"] += int(force_norm > force_cap)
                    counts[arm]["speed_clipped"] += int(speed > speed_cap)
                    counts[arm]["penetration_gt2"] += int(max(pen) > 2.)
                    for key, value in (("agent_speed_pre_cap", speed), ("agent_speed_post_cap", np.linalg.norm(entry[5:7])),
                            ("agent_speed_post", np.linalg.norm(post[5:7])), ("object_COM_speed_post", np.linalg.norm(post[7:9])),
                            ("object_omega_abs_post", abs(post[9])), ("max_penetration", max(pen))):
                        metrics[arm][key].append(float(value))
                    free_agent = not kinds.intersection(("agent_object", "agent_wall"))
                    no_object_contact = not kinds.intersection(("agent_object", "object_wall"))
                    if free_agent:
                        expected = entry[5:7]*(damping**.01)+(gravity+measured_force/agent_mass)*.01
                        metrics[arm]["no_agent_contact_full_forced_linear_residual"].append(float(np.linalg.norm(post[5:7]-expected)))
                    if no_object_contact:
                        expected = entry[7:9]*(damping**.01)+(gravity+entry[12:14]/object_mass)*.01
                        omega_expected = entry[9]*(damping**.01)+(0. if moment == "infinity" else entry[14]/float(moment))*.01
                        metrics[arm]["no_object_contact_full_linear_residual"].append(float(np.linalg.norm(post[7:9]-expected)))
                        metrics[arm]["no_object_contact_full_angular_residual"].append(abs(float(post[9]-omega_expected)))
                    pure = no_object_contact and np.all(entry[12:14] == 0.) and entry[14] == 0. and np.all(gravity == 0.)
                    if pure:
                        ratio = ratio_record(entry[7:9], post[7:9], entry[9], post[9], .01, damping)
                        ratio.update(arm=arm, family=family, **ref)
                        ratios[arm]["single_substeps"].append(ratio)
                        item = dict(entry=entry, post=post, control=control, substep=sub["index"], preceding_kinds=sorted(previous_kinds))
                        run.append(item)
                        control_free.append(item)
                    else:
                        finish_run()
                    previous_kinds = kinds
                    for contact_kind in ("agent_object", "object_wall", "agent_wall"):
                        counts[arm][contact_kind+"_substeps"] += int(contact_kind in kinds)
                    for event in contacts:
                        if event.get("phase") != "post_solve":
                            if "total_impulse" in event:
                                anomalies.append(dict(**ref, reason="impulse was recorded before post_solve"))
                            continue
                        impulse = vec(event.get("total_impulse"), 2)
                        if impulse is None:
                            anomalies.append(dict(**ref, reason="unknown impulse"))
                        else:
                            impulses[arm][kind(event)].append(float(np.linalg.norm(impulse)))
                if control_valid and log.get("completed") and len(log["substeps"]) == 10 and len(control_free) == 10:
                    first, last = control_free[0], control_free[-1]
                    ratio = ratio_record(first["entry"][7:9], last["post"][7:9], first["entry"][9], last["post"][9], .1, damping)
                    ratio.update(family=family, slot=slot["slot_id"], control=control)
                    ratios[arm]["complete_controls"].append(ratio)
            finish_run()
            runs, run_length = [], 0
            for value in contact_history+[False]:
                if value:
                    run_length += 1
                elif run_length:
                    runs.append(run_length)
                    run_length = 0
            counts[arm]["AO_contact_episodes"] += len(runs)
            counts[arm]["AO_lost_contact_transitions"] += sum(a and not b for a,b in zip(contact_history, contact_history[1:]))
            counts[arm]["AO_recontacts_after_observed_episode"] += max(0, len(runs)-1)
            metrics[arm]["AO_contact_episode_seconds"].extend(n*.01 for n in runs)
            slots.append(dict(cohort=cohort.name, arm=arm, slot=slot["slot_id"], letter=req["letter"], family=family,
                status=slot["status"], metadata_sha256=sha(meta_path), telemetry_sha256=sha(logs_path), trajectory_sha256=sha(npz_path)))
        checks[cohort.name+":charged_controls_match"] = cohort_logs == ledger["charged_controls"]
        checks[cohort.name+":substep_calls_match"] = cohort_steps == ledger["engine_step_calls"]
        checks[cohort.name+":no_new_model_gpu_optimizer"] = all(ledger[k] == 0 for k in ("model_calls", "GPU_work", "optimizer_attempts"))
    ratio_summary = {}
    for arm in ARMS:
        ratio_summary[arm] = {}
        for population in ("single_substeps", "complete_controls", "free_runs"):
            rows = ratios[arm][population]
            ratio_summary[arm][population] = dict(records=len(rows), independent_slots=len({r["slot"] for r in rows}),
                independent_families=len({r["family"] for r in rows}),
                entry_classes_linear=dict(Counter(r["linear_entry_class"] for r in rows)),
                entry_classes_angular=dict(Counter(r["angular_entry_class"] for r in rows)),
                **{key: stats(r[key] for r in rows) for key in ("linear_retention", "linear_projected_retention",
                    "angular_signed_retention", "linear_vector_residual_norm", "angular_signed_residual",
                    "linear_observed_half_life_seconds", "angular_observed_half_life_seconds")})
    checks["no_telemetry_invariant_anomalies"] = not anomalies
    checks["torch_not_imported_by_audit"] = "torch" not in sys.modules
    return dict(kind="R2A2-independent-cold-decay-audit.v1", passed=all(checks.values()), checks=checks,
        inputs=inputs, helper_sha256=sha(__file__), counts={arm:dict(c) for arm,c in counts.items()}, anomalies=anomalies,
        force_cap_velocity_statistics={arm:{key:stats(v) for key,v in m.items()} for arm,m in metrics.items()},
        pure_object_retention=ratio_summary, observed_free_episodes=episodes,
        post_only_native_impulse_norm={arm:{k:stats(v) for k,v in types.items()} for arm,types in impulses.items()}, slots=slots,
        expected={arm:dict(damping=d, single_substep_retention=d**.01, full_control_retention=d**.1) for arm,d in ARMS.items()},
        interpretation=dict(pure_free_object="No AO/OW pre or post event; exact zero measured object force/torque and gravity; collisions/unknowns terminate every free run",
            forced_agent="Full native update with measured clipped applied force; PD-driven agent speed is never interpreted as free half-life",
            ratios="Meaningful entry norm or absolute omega >1e-12; exact-zero and nonzero-nearzero entry classes remain separately counted, never assigned ratio0/1",
            controls="Only all10pure free substeps within an actually completed raw0.1s control; not H1/H4 history-window counts",
            episodes="All contiguous observed pure-free runs; initial-free runs separate from runs immediately following AO/OW contact; no collisions crossed or extrapolated",
            half_life="Fitted from observed positive retention strictly between0and1; q0 is immediate annihilation, q1 no attenuation, neither assigned a finite fitted half-life",
            position_caveat="Velocity retention does not assert identical displacement retention; native position integration also has bias-velocity terms following contacts",
            contact="Only post_solve impulse events aggregated; episodes describe actual100Hz event runs, not controller-intended recovery",
            bounds="All speeds and penetrations observed only for frozen8-control probes; not global stability guarantees or final protocol selection",
            supplemental_scope="Main reporter owns motion/rotation joint targets and signed-lever/arc-family support; this audit changes no acceptance threshold",
            work="Read-only old/new telemetry; no new physics/model/GPU/render/optimizer/training-iterator or damping search"))


def report(result):
    lines = ["# Independent finite-damping decay audit", "", f"Checks passed: **{result['passed']}**; anomalies={len(result['anomalies'])}.", "",
        "The object half-life analysis is restricted to recorded zero-force/zero-torque/zero-gravity substeps without any object-agent or object-wall event. "
        "The agent uses the full forced update, including measured PD force; agent motion is not called free decay.", ""]
    for arm in ARMS:
        c = result["counts"].get(arm, {})
        metrics = result["force_cap_velocity_statistics"].get(arm, {})
        lines.extend([f"## {arm}", "", f"Actual substeps={c.get('substeps',0)}, force clips={c.get('force_clipped',0)}, speed clips={c.get('speed_clipped',0)}; "
            f"expected q(.01s)={result['expected'][arm]['single_substep_retention']}, q(.1s)={result['expected'][arm]['full_control_retention']}.", ""])
        for population, row in result["pure_object_retention"][arm].items():
            l, a = row["linear_retention"], row["angular_signed_retention"]
            lines.append(f"- {population}: {row['records']} records/{row['independent_families']} initial families; "
                f"linear nonzero-entry n={l['finite']}, q range={l.get('min')}..{l.get('max')}; "
                f"angular nonzero-entry n={a['finite']}, q range={a.get('min')}..{a.get('max')}; zero/near-zero entries counted separately in JSON.")
        lines.append("")
        for key in ("no_agent_contact_full_forced_linear_residual", "no_object_contact_full_linear_residual", "no_object_contact_full_angular_residual",
                "agent_speed_post", "object_COM_speed_post", "object_omega_abs_post", "max_penetration"):
            r = metrics.get(key, {})
            lines.append(f"- {key}: n={r.get('finite',0)}, median={r.get('p50')}, max={r.get('max')}.")
        lines.extend(["", f"AO episodes={c.get('AO_contact_episodes',0)}, loss transitions={c.get('AO_lost_contact_transitions',0)}, "
            f"recontacts after an observed episode={c.get('AO_recontacts_after_observed_episode',0)}.", ""])
    lines.extend(["## Interpretation limits", "", "Zero-entry values do not define a retention ratio. A collision ends an episode; no decay estimate spans it. "
        "D0 destroys prior velocity at an update; D1 has no finite damping half-life. Finite candidates are tested against their specified native D**elapsed rule, not tuned to these observations. "
        "Velocity retention is not a displacement-retention claim: native position integration also includes bias velocity following contacts. "
        "Observed coasting, contact retention, wall impulses and velocity peaks must be assessed together; this audit does not select ideal physics or establish learnability or shape preservation. "
        "Original R2A files were only read; all outputs are in the new R2A2 telemetry-audit directory.", ""])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-closed", action="store_true", required=True)
    args = p.parse_args()
    result = audit(BASE)
    (BASE/"telemetry-audit/statistics.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    (BASE/"telemetry-audit/REPORT.md").write_text(report(result), encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"], checks=len(result["checks"]), anomalies=len(result["anomalies"]))))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
