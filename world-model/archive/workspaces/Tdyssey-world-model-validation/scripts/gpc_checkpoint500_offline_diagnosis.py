"""Freeze JSON-only selections, then inspect existing pixels without model work."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "artifacts/gpc-checkpoint500-offline-diagnosis-20261026"
FORMAL = "artifacts/gpc-checkpoint500-evaluation-20261026"
KEYS = {
    "config": ("configs/gpc_checkpoint500_evaluation.json", "cd586ed72153ad5d246b4109d18b8ee1faca5dfd7312c721c2c470ca0369e6e2"),
    "schedule": ("artifacts/wm-sampling-b-only-20261026/training-schedule.json", "f7217a2d09393798529b00e72fcd971e8d7be11cab7a9c769704a9b30ddd004a"),
    "data": ("artifacts/gpc-data-stage-a-20261026/adaptation-data.json", "2a28136ca9c3b111d202f507ebab16c992b4d26a9c48d388956e390c6b2b069e"),
    "benchmark": ("artifacts/gpc-data-stage-b2-20261026/benchmark.json", "dde6b5b8c31c8aac17f341d08f8f12eef78fe0e8ca4ebfe88e289c6289e759ea"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    if len(data) > 4*1024**2:
        raise ValueError("Individual diagnostic JSON exceeds 4 MiB")
    with path.open("xb") as stream:
        stream.write(data)
    return dict(path=str(path), bytes=len(data), sha256=digest(data))


def identity(path, expected=None):
    data = path.read_bytes()
    value = dict(path=path.relative_to(ROOT).as_posix(), bytes=len(data), sha256=digest(data))
    if expected and value["sha256"] != expected:
        raise ValueError("Sealed identity mismatch: "+str(path))
    return value, data


def stats(values):
    values = sorted(v for v in values if type(v) in (int, float) and math.isfinite(v))
    def quantile(q):
        point = (len(values)-1)*q
        low, high = math.floor(point), math.ceil(point)
        return values[low]+(values[high]-values[low])*(point-low)
    return dict(n=len(values), minimum=values[0] if values else None,
        median=quantile(.5) if values else None, p90=quantile(.9) if values else None,
        maximum=values[-1] if values else None, mean=sum(values)/len(values) if values else None)


def branch_key(row):
    return tuple(row[k] for k in ("stratum", "shape", "anchor", "candidate"))


def choose_branches(rows):
    """One lexicographic branch per physical cell, never per image outcome."""
    branches = {}
    for row in rows:
        previous = branches.setdefault(branch_key(row), row)
        if previous["contact_by_h4"] != row["contact_by_h4"]:
            raise ValueError("Physical label differs between repeats")
    cells = []
    for shape in ("T", "L"):
        for stratum in ("regression", "independent"):
            for contact in (False, True):
                eligible = sorted((r for r in branches.values() if r["shape"] == shape
                    and r["stratum"] == stratum and r["contact_by_h4"] is contact),
                    key=lambda r:(r["anchor"], r["candidate"]))
                cells.append(dict(shape=shape, stratum=stratum, contact_by_h4=contact,
                    available_branches=len(eligible), selected=branch_key(eligible[0]) if eligible else None))
    return cells


def choose_train(entries):
    result = []
    for shape in ("T", "L"):
        for contact in (False, True):
            seen = set()
            for row in entries:
                key = (row["trajectory_id"], row["anchor"])
                if row["shape"] == shape and row["contact"] is contact and key not in seen:
                    result.append(row)
                    seen.add(key)
                    if len(seen) == 2:
                        break
    return result


def summarize(groups):
    result = []
    numeric = ("component_mass_fraction", "area_ratio", "full_soft_iou", "roi_soft_iou",
               "outside_roi_template_mass", "squared_residual", "fitted_blur_sigma")
    for key, group in sorted(groups.items(), key=lambda item:str(item[0])):
        shapes = [r["measurement"]["shape"] for r in group]
        result.append(dict(cell=list(key), prediction_rows=len(group),
            physical_branches=len({branch_key(r) for r in group}),
            physical_anchors=len({branch_key(r)[:3] for r in group}),
            object_valid=sum(r["measurement"]["state"]["valid"] for r in group),
            pusher_valid=sum(r["measurement"]["pusher"]["valid"] for r in group),
            coverage_valid=sum(r["measurement"]["coverage"]["valid"] for r in group),
            coverage_null=sum(r["measurement"]["coverage"]["estimate"] is None for r in group),
            termination=dict(Counter(r["measurement"]["termination"]["status"] for r in group)),
            reasons=dict(Counter(v["reason"] for v in shapes)),
            metrics={name:dict(stats(v.get(name) for v in shapes),
                null_or_missing=sum(v.get(name) is None for v in shapes)) for name in numeric}))
    return result


def freeze(output):
    bindings, metadata = {}, {}
    for name, (path, sha) in KEYS.items():
        bindings[name], data = identity(ROOT/path, sha)
        metadata[name] = json.loads(data)
    inventory_path = ROOT/"artifacts/gpc-checkpoint500-evaluation-review-20261026/output-inventory.json"
    bindings["formal_inventory"], data = identity(inventory_path, "e91fd505fd5b51dc26b7983b6724721e02ffbc23630b84c560ad216333e2d8ac")
    inventory = json.loads(data)
    benchmark = metadata["benchmark"]
    regression = ROOT/benchmark["regression"]["path"]
    bindings["regression_report"], data = identity(regression/"report.json", benchmark["regression"]["report_sha256"])
    regression_rows = json.loads(data)["branches"]
    rows, groups, branch_turn = [], defaultdict(list), {}
    for path in sorted((ROOT/FORMAL/"measurements").glob("*.json")):
        relative = path.relative_to(ROOT/FORMAL).as_posix()
        file_id, data = identity(path, inventory[relative]["sha256"])
        if file_id["bytes"] != inventory[relative]["bytes"]:
            raise ValueError("Measurement byte count differs")
        row = json.loads(data)
        row["measurement_path"] = file_id["path"]
        row["file_sha256"], row["file_bytes"] = file_id["sha256"], file_id["bytes"]
        if row["horizon"] == 4:
            angle = row["truth"]["pose"]["angle"]-row["latest"]["pose"]["angle"]
            branch_turn[branch_key(row)] = abs(math.degrees(math.atan2(math.sin(angle), math.cos(angle)))) > 5
        rows.append(row)
    if len(rows) != 3996 or len({branch_key(r) for r in rows}) != 111:
        raise ValueError("Incomplete formal measurement/physical grid")
    for row in rows:
        key = tuple(row[k] for k in ("role", "stratum", "shape", "mode", "horizon", "contact_by_h4"))
        groups[key+(branch_turn[branch_key(row)],)].append(row)
    dump(output/"measurement-summary.json", dict(kind="checkpoint500_offline_measurement_summary.v1",
        rows=len(rows), physical_branches=111, retained_invalid=True, scores_recomputed=False,
        cell_fields=["weight", "stratum", "shape", "mode", "horizon", "contact_by_h4", "truth_large_rotation_by_h4"],
        cells=summarize(groups)))
    cells, selected = choose_branches(rows), []
    for cell in cells:
        if cell["selected"] is None:
            continue
        key = tuple(cell["selected"])
        row = next(r for r in rows if branch_key(r) == key)
        if cell["stratum"] == "independent":
            source = next(c for c in benchmark["cases"] if c["anchor"] == row["anchor"] and c["candidate"] == row["candidate"])
            payload = dict(source["payload"])
            payload["path"] = (Path(KEYS["benchmark"][0]).parent/payload["path"]).as_posix()
            goal = source["goal"]
        else:
            source = next(c for c in regression_rows if c["anchor"] == row["anchor"] and c["candidate"] == row["candidate"])
            payload, _ = identity(regression/f"{row['anchor']}-{row['candidate']}-physical.npz")
            goal = source["initial_state"]["goal"]
        predictions = [r for r in rows if branch_key(r) == key and r["seed"] == 5478
            and (r["mode"], r["horizon"]) in (("AR", 1), ("AR", 4), ("TF", 4))]
        if len(predictions) != 6:
            raise ValueError("Fixed pixel prediction selection is incomplete")
        for prediction in predictions:
            prediction["prediction_file_identity"] = inventory[prediction["raw_path"].replace("\\", "/")]
        selected.append(dict(shape=row["shape"], stratum=row["stratum"], contact_by_h4=row["contact_by_h4"],
            anchor=row["anchor"], candidate=row["candidate"], payload=payload, goal=goal,
            latest=row["latest"], truth=source["truth"] if "truth" in source else [s["truth"] for s in source["actual"]],
            predictions=predictions))
    train_entries = choose_train(metadata["schedule"]["entries"])
    train_ids = {r["trajectory_id"] for r in train_entries}
    declarations = {r["trajectory_id"]:r for r in metadata["data"]["episodes"]}
    data_parent = Path(KEYS["data"][0]).parent
    catalog_path = data_parent/metadata["data"]["catalogs"][0]["path"]
    bindings["catalog"], data = identity(ROOT/catalog_path, metadata["data"]["catalogs"][0]["sha256"])
    catalog, train_sources = json.loads(data), {}
    for run in catalog["runs"]:
        wanted = {i["trajectory_id"] for i in run["episodes"]} & train_ids
        if not wanted:
            continue
        run_path = data_parent/run["path"]
        manifest_id, data = identity(ROOT/run_path/"manifest.json", run["manifest_sha256"])
        bindings[run["path"]+"_manifest"] = manifest_id
        for episode in json.loads(data)["episodes"]:
            identifier = episode["trajectory_id"]
            if identifier not in wanted:
                continue
            if declarations[identifier]["role"] != "train":
                raise ValueError("Non-train payload requested")
            files = {}
            for item in episode["files"]:
                file_id = dict(item)
                file_id["path"] = (run_path/file_id.pop("name")).as_posix()
                files["payload" if file_id["path"].endswith(".npz") else "record"] = file_id
            train_sources[identifier] = files
    windows = [dict(r, **train_sources[r["trajectory_id"]]) for r in train_entries]
    manifest = dict(kind="checkpoint500_offline_fixed_samples.v1", source_commit="a461829c6bdf967c0bfab60e621e8c286465c68a",
        bindings=bindings, branch_rule="T/L x regression/independent x contact false/true; lexicographic(anchor,candidate) first; no empty-cell replacement",
        train_rule="first two distinct (trajectory_id,anchor) in original frozen schedule order per shape/contact cell",
        seed=5478, modes=[["AR", 1], ["AR", 4], ["TF", 4]], branch_cells=cells,
        branches=selected, train_windows=windows, raw_decode_caps=dict(predictions=48, case_frames=64, train_frames=40),
        no_scoring_changes=True, held_payload_access=0)
    result = dump(output/"fixed-samples.json", manifest)
    print(json.dumps(dict(status="frozen", sample_manifest=result, cells=cells, training_windows=[
        {k:r[k] for k in ("shape", "contact", "trajectory_id", "anchor", "cursor")} for r in windows])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "pixels"))
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if output != (ROOT/OUTPUT).resolve():
        raise ValueError("Only the fresh fixed diagnostic output is permitted")
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "freeze":
        freeze(output)
    else:
        if not args.manifest_sha256:
            raise ValueError("Pixels require the prior fixed sample identity")
        _, data = identity(output/"fixed-samples.json", args.manifest_sha256)
        sys.path.insert(0, str(ROOT))
        from gpc_checkpoint500_offline_pixels import run_pixels
        result = run_pixels(ROOT, output, json.loads(data))
        if "torch" in sys.modules:
            raise RuntimeError("Forbidden runtime imported")
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
