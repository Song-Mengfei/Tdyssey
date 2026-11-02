"""Saved-only synthesis of the approved two-epoch O/C continuation.

This module never imports Torch, constructs a model/simulator, or scores a new
prediction.  It groups existing machine metrics and renders a few saved raw
prediction comparisons. Display clipping is not fed back to any model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

HORIZONS = (1, 4, 8, 16)
LETTERS = tuple("TLOCXB")
FIELDS = ("raw_mask_iou", "object_position_error_image_px",
          "pusher_position_error_image_px", "raw_contour_error_image_px",
          "visible_angle_error_deg", "holes_difference", "area_fraction_error",
          "rgb_full_mae", "motion_vector_error_image_px",
          "underrotation_deg", "absolute_underrotation_deg")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _guard(root, config, extra):
    from common import check_storage
    return check_storage(root, config, reserve_bytes=int(extra))


def _write(path, value, root, config):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _guard(root, config, len(payload))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def _metric_value(row, name):
    metrics = row.get("metrics", {})
    value = metrics.get(name)
    if value is None and name == "visible_angle_error_deg":
        value = metrics.get("legacy", {}).get("angle_error_deg")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def summarize(rows):
    means = {}
    for field in FIELDS:
        values = [_metric_value(row, field) for row in rows]
        values = [value for value in values if value is not None]
        means[field] = dict(mean=float(np.mean(values)) if values else None,
                            available=len(values), unknown=len(rows) - len(values))
    failures = Counter(reason for row in rows
                       for reason in row.get("metrics", {}).get("failure_reasons", []))
    assessable = sum(bool(row.get("metrics", {}).get("legacy_shape_assessable", False)) for row in rows)
    return dict(denominator=len(rows), families=len({row.get("family_id", row["sample_id"]) for row in rows}),
                means=means, pusher_missing=failures.get("empty_pusher", 0),
                object_missing=failures.get("empty_object", 0),
                shape_assessable=assessable, shape_unknown=len(rows) - assessable,
                shape_failed=sum(row.get("metrics", {}).get("legacy_shape_deformation_failure") is True for row in rows),
                hard_generation_failures=sum(bool(row.get("metrics", {}).get("hard_generation_failure", False)) for row in rows),
                failure_reasons=dict(failures))


def _model_rows(metrics):
    return [row for row in metrics.get("rows", []) if row.get("method") != "copy_last"]


def _check_formal_pairing(formal):
    identities = [{(row["sample_id"], int(row["horizon"])) for row in _model_rows(value)}
                  for value in formal.values()]
    if not identities or any(value != identities[0] for value in identities[1:]):
        raise ValueError("Formal epoch comparisons do not use the same saved sample/horizon identities")
    raw = [{row["sample_id"]: row for row in value.get("raw_files", [])} for value in formal.values()]
    if any(set(value) != set(raw[0]) for value in raw[1:]):
        raise ValueError("Formal epoch raw prediction identities differ")
    for sid in raw[0]:
        seeds = [value[sid].get("seeds") for value in raw]
        if any(value != seeds[0] for value in seeds[1:]):
            raise ValueError("Formal paired initial-noise seeds differ")
    return dict(paired=True, sample_horizons=len(identities[0]),
                sample_ids=len(raw[0]), same_raw_noise_seeds=True)


def _raw(metrics, sample_id):
    record = next((item for item in metrics.get("raw_files", []) if item["sample_id"] == sample_id), None)
    if record is None:
        raise ValueError("Missing saved raw prediction: " + sample_id)
    path = Path(record["path"])
    if record.get("sha256") and sha(path) != record["sha256"]:
        raise ValueError("Saved raw prediction changed: " + str(path))
    with np.load(path, allow_pickle=False) as archive:
        value = archive["raw"].copy()
    if value.shape != (16, 3, 96, 96) or not np.isfinite(value).all():
        raise ValueError("Incomplete/nonfinite saved sixteen-frame prediction")
    return value, dict(path=str(path.resolve()), sha256=sha(path))


def _truth(row):
    path = Path(row["raw_path"])
    if row.get("raw_sha256") and sha(path) != row["raw_sha256"]:
        raise ValueError("Frozen truth trajectory changed")
    a = int(row["anchor"])
    with np.load(path, allow_pickle=False) as archive:
        rgb = archive["rgb"][a:a + 17].copy()
        valid = archive["control_valid"][a - 3:a + 16]
    if rgb.shape != (17, 3, 96, 96) or not valid.all():
        raise ValueError("Comparison lacks a real clean complete H16 truth sequence")
    return rgb[1:].astype(np.float32) / np.float32(255), rgb[0].astype(np.float32) / np.float32(255)


def _display(frame):
    return np.rint(np.clip(np.asarray(frame).transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)


def _frames(sample_id, columns):
    import cv2
    size = 192; height = 250; width = size * len(columns)
    output = []
    for step in range(16):
        image = np.full((height, width, 3), 255, np.uint8)
        cv2.putText(image, f"{sample_id} | t+{step+1}={(step+1)/10:.1f}s", (5, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, .34, (20,20,20), 1)
        for col, (title, frames) in enumerate(columns):
            parts = title.split("|")
            for line, part in enumerate(parts[:2]):
                cv2.putText(image, part, (col * size + 5, 29 + line * 13),
                            cv2.FONT_HERSHEY_SIMPLEX, .36, (20,20,20), 1)
            image[54:246, col * size:(col + 1) * size] = cv2.resize(
                _display(frames[step]), (size,size), interpolation=cv2.INTER_NEAREST)
        output.append(image)
    return output


def _media(root, config, name, sample_id, columns, *, video=False):
    import cv2
    folder = root / "report"; folder.mkdir(parents=True, exist_ok=True)
    frames = _frames(sample_id, columns)
    board = np.concatenate([frames[h-1] for h in HORIZONS], axis=0)
    success, png = cv2.imencode(".png", cv2.cvtColor(board, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Prediction comparison PNG encode failed")
    _guard(root, config, len(png))
    path = folder / (name + ".png"); path.write_bytes(png.tobytes())
    artifact = dict(sample_id=sample_id, board=str(path), board_sha256=sha(path),
                    columns=[value[0] for value in columns], horizons=list(HORIZONS),
                    display_only_quantization=True)
    if video:
        import imageio_ffmpeg
        target = folder / (name + "-10fps.mp4")
        _guard(root, config, 4 * 1024**2)
        height, width = frames[0].shape[:2]
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", "10", "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
        options = {"creationflags":subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        result = subprocess.run(cmd, input=b"".join(frame.tobytes() for frame in frames),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
        if result.returncode:
            raise RuntimeError("Comparison video failed: " + result.stderr.decode(errors="replace")[-400:])
        capture = cv2.VideoCapture(str(target)); count = 0
        fps = capture.get(cv2.CAP_PROP_FPS)
        while capture.read()[0]:
            count += 1
        capture.release()
        if count != 16 or abs(fps - 10) > 1e-3:
            raise RuntimeError("Saved comparison video has incorrect true timing")
        artifact.update(video=str(target), video_sha256=sha(target), frames=count, fps=fps)
    return artifact


def _fmt(group, field, digits=3):
    item = group["means"][field]
    return "unknown" if item["mean"] is None else f'{item["mean"]:.{digits}f} ({item["available"]})'


def _table(groups, *, title="Epoch"):
    lines = [f"| {title} | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        lines.append(f'| {group["epoch"]} | {group["horizon"]} | {group["denominator"]} | '
                     f'{_fmt(group,"raw_mask_iou")} | {_fmt(group,"object_position_error_image_px")} | '
                     f'{_fmt(group,"visible_angle_error_deg")} | {_fmt(group,"raw_contour_error_image_px")} | {_fmt(group,"holes_difference")} | '
                     f'{group["pusher_missing"]} | {group["shape_unknown"]} |')
    return lines


def _html_table(groups, title="Epoch"):
    text = _table(groups, title=title)
    cells = [line.strip("|").split("|") for line in text]
    return ("<table><thead><tr>" + "".join("<th>"+html.escape(x.strip())+"</th>" for x in cells[0])
            + "</tr></thead><tbody>" + "".join("<tr>"+"".join("<td>"+html.escape(x.strip())+"</td>" for x in row)+"</tr>" for row in cells[2:]) + "</tbody></table>")


def _reference(value, root):
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return path
    if isinstance(value, dict):
        for key in ("metrics", "metrics_path", "path"):
            if value.get(key):
                return _reference(value[key], root)
    return None


def _diagnostic_metrics(result, mode, root):
    """The diagnostics adapter may provide values directly or saved file paths."""
    value = result.get(mode)
    if isinstance(value, dict) and "rows" in value:
        return value, None
    path = _reference(value, root)
    if path is not None and path.exists():
        return read(path), str(path)
    return None, str(path) if path else None


def _loss_tables(records):
    lines=["| Epoch | Split / letter | Fixed samples | Residual MSE |", "|---|---|---:|---:|"]
    cells=[]
    for epoch,value in sorted(records.items()):
        for key in ("train","seen_val","train:O","train:C","seen_val:O","seen_val:C"):
            result=value.get("means",{}).get(key)
            if not result:
                continue
            mean=result.get("mean"); text="unknown" if mean is None else f"{float(mean):.8f}"
            cells.append((str(epoch),key,str(result.get("samples",0)),text))
            lines.append(f'| {epoch} | {key} | {result.get("samples",0)} | {text} |')
    table="<table><thead><tr><th>Epoch</th><th>Split / letter</th><th>Fixed samples</th><th>Residual MSE</th></tr></thead><tbody>"
    table += "".join("<tr>"+"".join("<td>"+html.escape(x)+"</td>" for x in row)+"</tr>" for row in cells)
    return lines,table+"</tbody></table>"


def generate_report(root: Path, config: dict, engine_result: dict) -> dict:
    root = Path(root).resolve(); report = root / "report"; report.mkdir(parents=True, exist_ok=True)
    ar1 = Path(config.get("ar1_root", config.get("data_root",root.parent / "accelerated-rgb-run-20261031")))
    formal_paths = engine_result.get("formal48", {"1":str(ar1/"evaluation/after/metrics.json"),
        "2":str(root/"evaluation/epoch2/metrics.json"), "3":str(root/"evaluation/epoch3/metrics.json")})
    formal = {str(epoch):read(_reference(value, root)) for epoch,value in formal_paths.items()}
    pairing = _check_formal_pairing(formal)
    spec = read(ar1 / "evaluation-spec.json")
    spec_rows = {row["sample_id"]:row for row in spec["rows"]}
    strong_ids = sorted({row["sample_id"] for row in _model_rows(formal["1"])
                         if row["letter"] in "OC" and row["horizon"] == 16 and row.get("obvious_motion")})
    groups = []
    for epoch,metrics in sorted(formal.items()):
        rows = _model_rows(metrics)
        for letter in ("ALL", *LETTERS, "OC_strong8"):
            for horizon in HORIZONS:
                selected = [row for row in rows if row["horizon"] == horizon and
                            (letter == "ALL" or row["letter"] == letter or
                             letter == "OC_strong8" and row["sample_id"] in strong_ids)]
                groups.append(dict(epoch=epoch,letter=letter,horizon=horizon,**summarize(selected)))
    baseline = read(ar1 / "evaluation/before/metrics.json")
    for horizon in HORIZONS:
        rows = [row for row in baseline.get("rows", []) if row.get("method") == "copy_last" and row["horizon"] == horizon]
        groups.append(dict(epoch="copy_last",letter="ALL",horizon=horizon,**summarize(rows)))
    media = []; inputs = [dict(path=str(_reference(value,root)),sha256=sha(_reference(value,root)))
                           for value in formal_paths.values()]
    inputs += [dict(path=str(path),sha256=sha(path)) for path in
               (ar1/"evaluation-spec.json",ar1/"evaluation/before/metrics.json")]
    for letter in LETTERS:
        candidates = sorted(sid for sid,row in spec_rows.items() if row["letter"] == letter)
        preferred = [sid for sid in candidates if sid in strong_ids] if letter in "OC" else []
        if not candidates:
            continue
        sid = (preferred or candidates)[0]; row = spec_rows[sid]
        truth,last = _truth(row); columns = [("SAVED TRUTH",truth)]
        for epoch in ("1","2","3"):
            prediction,record = _raw(formal[epoch],sid); inputs.append(record)
            columns.append((f"EPOCH {epoch}|AR16 / denoise3",prediction))
        columns.append(("COPY LAST",np.repeat(last[None,:],16,axis=0)))
        artifact = _media(root,config,f"formal-{letter}-epoch1-2-3",sid,columns,video=letter in "OCB")
        artifact.update(kind="formal48_same_case",letter=letter,split="seen_val",
                        selection="First frozen strong-motion ID for O/C; first original fixed ID for other letters; not selected by prediction quality")
        media.append(artifact)
    diagnostics = engine_result.get("diagnostics", {})
    diag_groups = []; diag_records = {}; missing = []
    for epoch in ("1","3"):
        epoch_result = diagnostics.get(epoch, {})
        for mode in ("real_history", "autoregressive"):
            if mode not in epoch_result:
                epoch_result = dict(epoch_result, **{mode:str(root/"diagnostics"/f"epoch{epoch}"/mode/"metrics.json")})
            metrics,path = _diagnostic_metrics(epoch_result,mode,root)
            if metrics is None:
                missing.append(dict(epoch=epoch,mode=mode,path=path)); continue
            diag_records[(epoch,mode)] = metrics
            if path:
                inputs.append(dict(path=path,sha256=sha(path)))
            for split in ("train","seen_val"):
                for letter in ("ALL_OC","O","C"):
                    for horizon in HORIZONS:
                        selected = [row for row in _model_rows(metrics) if row.get("split")==split and
                                    row["horizon"]==horizon and (letter=="ALL_OC" or row["letter"]==letter)]
                        diag_groups.append(dict(epoch=epoch,mode=mode,split=split,letter=letter,horizon=horizon,**summarize(selected)))
    # Detailed diagnostic plots are added only when the same frozen raw case exists
    # in all four epoch/input conditions. An absent case is reported, never padded.
    diag_spec_path = root / "diagnostic-spec.json"
    if diag_spec_path.exists() and all((ep,mode) in diag_records for ep in ("1","3") for mode in ("real_history","autoregressive")):
        diag_spec = read(diag_spec_path)
        inputs.append(dict(path=str(diag_spec_path),sha256=sha(diag_spec_path)))
        rows = diag_spec.get("rows", [])
        for split in ("train","seen_val"):
            for letter in ("O","C"):
                cases = sorted((row for row in rows if row.get("split")==split and row["letter"]==letter),key=lambda x:x["sample_id"])
                if not cases:
                    missing.append(dict(split=split,letter=letter,reason="No frozen diagnostic H16 case"));continue
                row=cases[0];sid=row["sample_id"];truth,last=_truth(row)
                columns=[("SAVED TRUTH",truth)]
                for epoch,mode,label in (("1","real_history","E1 TRUE HISTORY|single step each time"),("1","autoregressive","E1 AR16|predicted history"),("3","real_history","E3 TRUE HISTORY|single step each time"),("3","autoregressive","E3 AR16|predicted history")):
                    raw,record=_raw(diag_records[(epoch,mode)],sid);inputs.append(record);columns.append((label,raw))
                artifact=_media(root,config,f"diagnostic-{split}-{letter}-true-vs-ar",sid,columns,video=True)
                artifact.update(kind="motion_biased_diagnostic",split=split,letter=letter)
                media.append(artifact)
    fixed_loss = {}; fixed_loss_missing = []
    for epoch in ("1","2","3"):
        path = _reference(diagnostics.get(epoch,{}).get("fixed_loss"),root)
        if path is None:
            path = root/"diagnostics"/f"epoch{epoch}"/"fixed-loss.json"
        if path.exists():
            fixed_loss[epoch] = read(path)
            inputs.append(dict(path=str(path),sha256=sha(path)))
        else:
            fixed_loss_missing.append(str(path))
            missing.append(dict(epoch=epoch,mode="fixed_residual_loss",path=str(path)))
    fixed_noise_ids=[value.get("noise_and_sample_identity") for value in fixed_loss.values()]
    fixed_noise_matched=bool(fixed_noise_ids) and all(value is not None and value==fixed_noise_ids[0] for value in fixed_noise_ids)
    if any(value is not None for value in fixed_noise_ids) and not fixed_noise_matched:
        raise ValueError("Fixed-residual-loss sample/noise identities differ across epochs")
    denoise_path = _reference(diagnostics.get("1",{}).get("denoise16"),root)
    if denoise_path is None:
        denoise_path = root/"diagnostics"/"epoch1"/"denoise16"/"metrics.json"
    denoise_groups = []; denoise_record = None
    if denoise_path.exists():
        denoise_record = read(denoise_path)
        inputs.append(dict(path=str(denoise_path),sha256=sha(denoise_path)))
        d16ids = {row["sample_id"] for row in denoise_record.get("raw_files",[])}
        if d16ids != set(strong_ids):
            raise ValueError("Sixteen-denoise diagnostic differs from the frozen eight original strong O/C cases")
        for label,rows in (("epoch1/denoise3",[row for row in _model_rows(formal["1"]) if row["sample_id"] in strong_ids]),
                           ("epoch1/denoise16",_model_rows(denoise_record))):
            for letter in ("ALL_OC","O","C"):
                for h in HORIZONS:
                    selected=[row for row in rows if row["horizon"]==h and (letter=="ALL_OC" or row["letter"]==letter)]
                    denoise_groups.append(dict(epoch=label,letter=letter,horizon=h,**summarize(selected)))
        for letter in ("O","C"):
            cases=[sid for sid in strong_ids if spec_rows[sid]["letter"]==letter]
            if not cases:
                continue
            sid=cases[0];truth,last=_truth(spec_rows[sid])
            three,record3=_raw(formal["1"],sid);sixteen,record16=_raw(denoise_record,sid)
            inputs.extend((record3,record16))
            artifact=_media(root,config,f"inference-only-{letter}-epoch1-denoise3-vs16",sid,
                [("SAVED TRUTH",truth),("E1 / DENOISE 3|reused formal output",three),
                 ("E1 / DENOISE 16|same recorded noise",sixteen),
                 ("COPY LAST",np.repeat(last[None,:],16,axis=0))],video=False)
            artifact.update(kind="epoch1_inference_only",letter=letter,split="seen_val")
            media.append(artifact)
    else:
        missing.append(dict(epoch="1",mode="denoise16",path=str(denoise_path)))
    limits = [
        "Only two additional full epochs were approved. No new physical sampling, architecture/loss/LR change, server or fourth epoch.",
        "Formal48 uses the unchanged seen-val cases, three-denoise sampler and recursive predictions. All six letters, including B regressions, remain in the tables.",
        "The O/C movement diagnostic deliberately favors recorded movement/contact. Training and seen-val are separate; this is not a natural-distribution or unseen-letter test.",
        "Real-history predictions each receive four prior true frames for that target time. Their t+16 image is a single-step diagnostic, never a sixteen-step closed-loop prediction.",
        "AR16 receives true history only at its initial anchor, then raw predictions, with no future truth feedback or shape repair.",
        "Position/IoU, angular error, raw contour and visible hole counts are distinct observations. Occlusion/symmetry/deformation can make angle or shape unknown; unknown is not a pass.",
        "Raw contour error mixes position, rotation and boundary shape. Corner counts or lower contour error alone cannot establish rigid-corner or hole preservation.",
        "Absolute angular alignment error alone cannot establish underrotation. A signed, assessable motion comparison is required; unavailable underrotation remains unknown.",
        "Three-vs-sixteen denoising is an inference-only comparison at the same epoch1 checkpoint on eight fixed O/C strong cases. It cannot be counted as training improvement or a new default.",
        "The sixteen-denoise diagnostic injects each case's original batch-four initial noise into a batch-one call. Batch-one versus batch-four floating-point kernel rounding may also differ; this is not a perfectly isolated causal estimate of denoising-step count.",
        "Display PNG/video clips and quantizes only for viewing. Saved raw floats and all existing machine metrics remain unchanged. No shape or penetration correction is claimed.",
        "These finite comparisons can support candidate causes; they do not prove model-capacity failure, convergence or a universal cure for autoregressive drift."]
    summary=dict(kind="oc-two-epoch-saved-report.v1",run_id=config.get("run_id"),
        engine_complete=engine_result.get("complete"),formal_pairing=pairing,formal_groups=groups,
        strong_oc_sample_ids=strong_ids,diagnostic_groups=diag_groups,
        fixed_residual_loss=fixed_loss,fixed_residual_loss_missing=fixed_loss_missing,
        fixed_loss_sample_noise_identity_matched=fixed_noise_matched,
        inference_only_denoise_groups=denoise_groups,
        inference_only_denoise_receipt=denoise_record,
        missing_diagnostics=missing,diagnostics=diagnostics,epochs=engine_result.get("epochs",[]),
        additional_exposures=engine_result.get("actual_additional_exposures"),
        successful_updates=engine_result.get("successful_updates"),AMP_skips=engine_result.get("AMP_skips"),
        rng_preservation=engine_result.get("rng_preservation"),media=media,limits=limits,
        saved_only_report=True,report_physics_calls=0,report_model_calls=0)
    _write(report/"summary.json",summary,root,config)
    lines=["# O/C continuation: new-data epochs 1 / 2 / 3", "",
           f'Additional exposures: {summary["additional_exposures"]}; successful updates: {summary["successful_updates"]}; AMP skips: {summary["AMP_skips"]}.',
           "Epoch1 formal predictions were reused from AR1. Epoch2/3 use the same cases/actions/noise and formal three-denoise sampler.","",
           "## Formal48: all six seen letters", ""]
    lines += _table([group for group in groups if group["letter"]=="ALL"])
    lines += ["", "## Per-letter H16, including regressions", ""]
    for letter in LETTERS:
        lines += ["",f"### {letter}",""] + _table([g for g in groups if g["letter"]==letter and g["horizon"]==16])
    lines += ["", "## Fixed strong O/C subset (eight original seen-val cases)", ""]
    lines += _table([group for group in groups if group["letter"]=="OC_strong8"])
    lines += ["", "## Movement diagnostics: real history versus AR16", "",
              "Each table keeps train and seen-val separate. Real-history t+16 is a separate one-step prediction using its own true preceding history; it is not a successful closed-loop H16.", ""]
    for split in ("train","seen_val"):
        for mode in ("real_history","autoregressive"):
            lines += ["",f"### {split} / {mode}",""] + _table([g for g in diag_groups if g["split"]==split and g["mode"]==mode and g["letter"]=="ALL_OC"])
    lines += ["","## Epoch1 inference-only: three versus sixteen denoising steps","",
              "The original three-step outputs are reused. This changes inference, not trained weights. Original per-case noise is injected; batch-one versus original batch-four rounding remains a computational caveat.",""]
    lines += _table([g for g in denoise_groups if g["letter"]=="ALL_OC"],title="Checkpoint / sampler")
    lines += ["","## Fixed samples and fixed diffusion-noise residual loss","",
              "Train and seen-val fixed-sample measurements are saved separately. These objective values use matched cases/noise; ordinary random-batch loss alone is not evidence of convergence.",""]
    loss_lines,loss_html=_loss_tables(fixed_loss)
    lines += loss_lines
    for epoch,value in fixed_loss.items():
        lines += [f'- Epoch {epoch}: [fixed-loss machine result]({root/"diagnostics"/f"epoch{epoch}"/"fixed-loss.json"})']
    lines += ["", "## Interpretation limits", ""] + ["- "+text for text in limits]
    if missing:
        lines += ["", "Missing diagnostics: "+json.dumps(missing,ensure_ascii=False)]
    lines += ["", "## Saved comparisons", ""]
    for item in media:
        lines += [f'- [{item.get("split","")} {item["letter"]}: {item["kind"]}]({item["board"]})']
    lines += ["",f'[Complete grouped machine report]({report/"summary.json"})']
    text="\n".join(lines)+"\n";_guard(root,config,len(text.encode("utf-8")));(root/"REPORT.md").write_text(text,encoding="utf-8")
    chunks=['<!doctype html><meta charset="utf-8"><title>O/C continuation epochs1–3</title>',
            '<style>body{font:15px system-ui;margin:24px auto;max-width:1200px;padding:0 20px}table{border-collapse:collapse;margin:20px 0;font-size:13px}th,td{padding:6px;border:1px solid #ccc}img,video{max-width:100%}p,li{line-height:1.5}details{margin:18px 0}</style>',
            '<h1>O/C continuation: epochs 1 / 2 / 3</h1>',
            '<p>Saved-only report. Zero new physical sampling. The formal comparison keeps three denoising steps, identical cases/actions/noise and raw recursive feedback.</p>',
            '<h2>Formal48: all letters</h2>',_html_table([g for g in groups if g['letter']=='ALL']),
            '<h2>Per-letter H16, including B and other regressions</h2>']
    for letter in LETTERS:
        chunks += [f'<h3>{letter}</h3>',_html_table([g for g in groups if g['letter']==letter and g['horizon']==16])]
    chunks += ['<h2>Fixed O/C strong subset</h2>',_html_table([g for g in groups if g['letter']=='OC_strong8'])]
    for split in ('train','seen_val'):
        for mode in ('real_history','autoregressive'):
            chunks += [f'<h2>{split}: {mode}</h2>',_html_table([g for g in diag_groups if g['split']==split and g['mode']==mode and g['letter']=='ALL_OC'])]
    chunks += ['<h2>Epoch1 inference only: denoise3 versus denoise16</h2>',
               '<p>Identical trained checkpoint; original3 outputs reused. Original per-case noise is injected, with a batch-one versus batch-four numerical-rounding caveat. This is not training gain.</p>',
               _html_table([g for g in denoise_groups if g['letter']=='ALL_OC'],title='Checkpoint / sampler'),
               '<h2>Fixed sample/noise residual objectives</h2>',
               '<p>Matched train and seen-val results are saved separately. Random batch-loss changes alone cannot establish convergence.</p>',loss_html]
    for epoch in fixed_loss:
        href=os.path.relpath(root/'diagnostics'/f'epoch{epoch}'/'fixed-loss.json',report).replace('\\','/')
        chunks += [f'<p><a href="{html.escape(href)}">Epoch{epoch} fixed-loss machine result</a></p>']
    chunks += ['<h2>Read these comparisons correctly</h2><ul>']+['<li>'+html.escape(x)+'</li>' for x in limits]+['</ul>']
    for item in media:
        chunks += [f'<h2>{html.escape(item["letter"]+" / "+item["split"]+" / "+item["kind"])}</h2>',
                   '<p>'+html.escape(item['sample_id'])+'</p>',f'<img src="{Path(item["board"]).name}">']
        if item.get('video'):
            chunks += [f'<video controls preload="metadata" src="{Path(item["video"]).name}"></video>']
    chunks += ['<p><a href="summary.json">Full grouped results and unavailable counts</a></p>']
    html_text="\n".join(chunks);_guard(root,config,len(html_text.encode("utf-8")));(report/'index.html').write_text(html_text,encoding='utf-8')
    manifest=dict(kind='oc-saved-report-manifest.v1',inputs=inputs,outputs=[dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(report.iterdir()) if path.is_file() and path.name!='final-manifest.json'],report=dict(path=str(root/'REPORT.md'),sha256=sha(root/'REPORT.md')))
    _write(report/'final-manifest.json',manifest,root,config)
    return dict(report=str(root/'REPORT.md'),index=str(report/'index.html'),summary=str(report/'summary.json'),
                manifest=str(report/'final-manifest.json'),media=media,missing_diagnostics=missing,
                report_physics_calls=0,report_model_calls=0)
