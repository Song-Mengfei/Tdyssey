"""Compare v1/v2 image interfaces using saved development RGB only.

No physics, renderer, checkpoint, policy or world-model inference is invoked.
The decoder receives RGB, shape ID and fixed goal only; truth is posthoc labels.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":"), allow_nan=False)
                             if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def output_manifest(directory):
    """Bind all saved diagnosis payloads and report exact final output bytes."""
    files={path.relative_to(directory).as_posix():dict(bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(directory.rglob("*")) if path.is_file()}
    payload_bytes=sum(row["bytes"] for row in files.values())
    manifest=dict(kind="tdyssey.wm_interface_v2_output_manifest.v1",files=files,
        payload_bytes_excluding_this_manifest=payload_bytes,manifest_bytes=0,total_output_bytes=payload_bytes)
    for _ in range(20):
        encoded=(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
        length=len(encoded)
        if manifest["manifest_bytes"]==length:
            break
        manifest.update(manifest_bytes=length,total_output_bytes=payload_bytes+length)
    else:
        raise RuntimeError("Output byte manifest size did not stabilize")
    with (directory/"output-manifest.json").open("xb") as stream:
        stream.write(encoded)
    return manifest


def cell(value):
    return "unavailable" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(cell(v).replace("|", "\\|") for v in row) + " |" for row in rows)])


def markdown(results):
    summary = results["summary"]
    error_rows = [r for r in results["strata"] if r["source"] in ("calibration", "forecast")
                  and r["horizon"] in (None, 1, 4) and r["stratum"]=="all"]
    confusion_rows = [r for r in results["classifications"] if r["source"]=="calibration" and r["stratum"]=="near_095"]
    ranking_rows = [r for r in summary["rank_summary"] if r["horizon"] in (1,4)]
    return "\n".join(["# Image-interface v2 development diagnosis", "",
        "All reused calibration, branch and model frames were inspected development data. This report does not establish independent validation, generated-image uncertainty calibration, success precision or policy improvement.", "",
        f"Expected image rows: {summary['expected_image_rows']}; available: {summary['available_image_rows']}. RGB hash caches reduce repeated measurement without removing any row.", "",
        "Decoder inputs contain only saved RGB, known shape and fixed goal. Simulator truth supplies posthoc error labels. Missing object, shape, pusher and score eligibility are independent fields.", "",
        "## Observed image errors", "", table(["Source", "Shape", "Version", "H", "Metric", "Anchor mean", "Measured / expected rows"],
            [[r["source"],r["shape"],r["version"],r["horizon"],r["metric"],r["anchor_equal_observed_mean"],
              f"{r['measured_image_rows']} / {r['expected_image_rows']}"] for r in error_rows]), "",
        "Means use the measurable subset. Their denominators and coverage are reported; a small recovered subset cannot pass a validation gate.", "",
        "## Near-threshold real calibration classification", "", table(["Shape", "Version", "Decision", "TP", "TN", "FP", "FN", "Unknown", "Precision"],
            [[r["shape"],r["version"],r["method"],r["TP"],r["TN"],r["FP"],r["FN"],r["unknown"],r["precision"]] for r in confusion_rows]), "",
        "No predicted-positive cases means precision is unavailable. An absolute-error development interval is an assumption and has no probability/confidence interpretation on model predictions.", "",
        "## Raw endpoint and stopping-consistent rankings", "", table(["Shape", "Version", "H", "Mode", "Complete predictions / anchors", "Scored / abstained pairs", "Observed pair accuracy", "Conservative pair credit", "Observed regret", "Conservative upper", "Hold", "Uniform"],
            [[r["shape"],r["version"],r["horizon"],r["scoring_mode"],f"{r['prediction_complete_anchors']} / {r['expected_anchors']}",
              f"{r['scored_pairs']} / {r['abstained_pairs']}",r["anchor_equal_observed_pair_accuracy"],r["anchor_equal_conservative_pair_accuracy"],
              r["observed_selected_regret"],r["conservative_regret_upper"],r["hold_regret_independent"],r["uniform_expectation_regret_independent"]] for r in ranking_rows]), "",
        "Observed pair accuracy is unavailable when no eligible pair is scored. Abstained pairs are counted separately; zero credit for those pairs belongs only to the explicitly labeled conservative engineering bound.", "",
        "Raw endpoint ranks are diagnostic, with physical-absorption flags preserved. Stopping-consistent scoring requires a complete unambiguous predicted stopping prefix; true futures never repair it. Hold/uniform and privileged observed-pose persistence retain their independent physical denominators.", "",
        "## Posthoc case evidence", "", table(["Category", "Matching image rows", "Examples retained"],
            [[name,values["total_matching_rows"],len(values["examples"])] for name,values in results["cases"].items()]), "",
        "Cases are the first six matches in fixed inventory order, selected only after decoding for explanation. They are not new scene selection, test examples or evidence of a validation pass.", "",
        "No additional reset, restore, render, controller step, world-model/policy inference or optimizer update occurred.", ""])


def case_previews(cases, directory):
    """Scientific previews from existing RGB arrays, not environment rendering."""
    import numpy as np
    from PIL import Image, ImageDraw
    from tdyssey.envs.official import official_shape
    from tdyssey.geometry import Pose
    from tdyssey.wm_interface_diagnostics import load_rgb
    paths = []
    for category, group in cases.items():
        for index, example in enumerate(group["examples"]):
            image = load_rgb(example)
            if image is None:
                continue
            if image.shape==(3,96,96):
                image = np.moveaxis(image,0,-1)
            pixels = np.clip(image*255 if np.issubdtype(image.dtype,np.floating) else image,0,255).astype(np.uint8)
            canvas = Image.new("RGB",(768,360),"white")
            draw = ImageDraw.Draw(canvas)
            for pane, version in enumerate(("v1","v2")):
                frame = Image.fromarray(pixels).resize((288,288),Image.Resampling.NEAREST)
                x0 = pane*384
                canvas.paste(frame,(x0,40))
                view = example[version+"_view"]
                pose = view.get("pose")
                if pose and view.get("object_valid"):
                    for polygon in official_shape(example["shape"]).polygons(Pose(**pose)):
                        points=[(x0+x/512*288,40+y/512*288) for x,y in polygon]
                        draw.line(points+[points[0]],fill=(240,70,20),width=2)
                draw.text((x0+2,2),version+" "+example["id"],fill="black")
                draw.text((x0+2,17),"coverage="+cell(view.get("coverage"))+" term="+str(view.get("termination")),fill="black")
                draw.text((x0+2,330),"object="+str(view.get("object_valid"))+" pusher="+str(view.get("pusher_valid")),fill="black")
                labels=example.get("error_labels") or {}
                draw.text((x0+2,345),"truth coverage="+cell(labels.get("truth_coverage"))+" success="+cell(labels.get("actual_success")),fill="black")
            filename=f"{category}-{index:02d}.png"
            canvas.save(directory/filename)
            paths.append(filename)
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/wm-interface-v2-development")
    parser.add_argument("--no-previews",action="store_true")
    args=parser.parse_args()
    run_dir, output=args.run_dir.resolve(),args.output_dir.resolve()
    if output==run_dir or output.is_relative_to(run_dir) or run_dir.is_relative_to(output):
        parser.error("Development output must be separate from immutable original run")
    if output.exists() and any(output.iterdir()):
        parser.error("Refusing to overwrite an existing development diagnosis")
    from tdyssey.frozen_wm_decoder import decode as decode_v1
    from tdyssey.wm_image_measurement_v2 import decode as decode_v2
    from tdyssey.wm_interface_diagnostics import analyse
    tracked=(Path(__file__), ROOT/"tdyssey/wm_interface_diagnostics.py",ROOT/"tdyssey/wm_image_measurement_v2.py",
             ROOT/"tdyssey/wm_measurement_contract.py",ROOT/"tdyssey/frozen_wm_decoder.py",ROOT/"configs/frozen_wm_decoder_v1.json",
             ROOT/"configs/wm_image_measurement_v2.json")
    source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    def progress(done,total,calls):
        print(json.dumps(dict(measured_rows=done,expected_rows=total,decoder_calls=calls)),flush=True)
    results=analyse(run_dir,decoder_v1=decode_v1,decoder_v2=decode_v2,progress=progress)
    results["summary"].update(source_sha256=source_hashes,created_at=datetime.now(timezone.utc).isoformat())
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"inventory.json",dict(expected_rows=results["summary"]["expected_image_rows"],
        available_rows=results["summary"]["available_image_rows"],
        expected_by_source=results["summary"]["expected_by_source"],
        available_by_source=results["summary"]["available_by_source"],
        rows=[{key:row.get(key) for key in ("id","source","shape","anchor","candidate","noise_seed","horizon",
              "path","key","index","input_available","rgb_sha256")} for row in results["images"]]))
    with (output/"per-image.jsonl").open("x",encoding="utf-8") as stream:
        for row in results["images"]:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    for name,key in (("per-image","flat_images"),("stratified-errors","strata"),("termination-confusion","classifications"),
                     ("sequence-scores","sequences"),("rankings","rankings"),("baselines","baselines")):
        write_csv(output/(name+".csv"),results[key])
    write_csv(output/"ranking-summary.csv",results["summary"]["rank_summary"])
    write_json(output/"cases.json",results["cases"])
    if not args.no_previews:
        (output/"cases").mkdir()
        results["summary"]["case_previews"]=case_previews(results["cases"],output/"cases")
    (output/"summary.md").write_text(markdown(results),encoding="utf-8")
    for path,digest in source_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("Decoder or offline source changed during diagnosis: "+path)
    forbidden=[name for name in ("torch","pymunk","pygame","diffusion_policy") if name in sys.modules]
    if forbidden:
        raise RuntimeError("Unexpected forbidden runtime imports: "+str(forbidden))
    results["summary"]["forbidden_runtime_imports"]=forbidden
    results["summary"]["output_bytes_before_summary"]=sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    results["summary"]["output_bytes_scope"]="before summary.json and output-manifest.json; exact final bytes appear in output-manifest.json"
    write_json(output/"summary.json",results["summary"])
    manifest=output_manifest(output)
    print(json.dumps(dict(output_dir=str(output),available_image_rows=results["summary"]["available_image_rows"],
                         expected_image_rows=results["summary"]["expected_image_rows"],total_output_bytes=manifest["total_output_bytes"]),sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
