"""Offline saved-file case figures. No sampler, simulator, or scoring changes."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

SEED = 5478
RULES = {
    "version": 1, "seed": SEED, "shape": "L",
    "ordering": "lexicographic (anchor,candidate); selections deduplicated by branch",
    "categories": {
        "contact_h4_rejected": "all saved contact-by-H4 branches with B-AR H4 joint rejection",
        "rotation_gt5_h4_rejected": "all branches with wrapped true H4 rotation >5deg and B-AR H4 rejection",
        "AR_rejected_TF_recovered": "first two branches with rejected B-AR H4 and accepted B-TF H4",
        "AR_TF_both_rejected": "first two branches with both B-AR and B-TF H4 rejected",
        "noncontact_hold_control": "first two noncontact hold branches with B-AR and B-TF H4 accepted",
        "matched_hold_control": "available hold for first contact and first rotation selected anchor",
    },
    "display": "clip[0,1], round*255 uint8, nearest pixels; never fed to fitting/model/scoring",
    "fit_residual": "abs(frozen RGB chromatic alpha - original stored candidate rigid template), fixed scale[0,1]",
    "color_residual": "minimum RGB RMS/255 distance to fixed source/output palette segments, display scale[0,.10]",
    "truth_use": "posthoc stratum labels and separately colored outlines only; never template fitting/selection by visual appeal",
    "original_thresholds_unchanged": True,
}
RULES_SHA256 = hashlib.sha256(json.dumps(RULES, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _joint(row):
    m = row.get("measurement") or {}
    return (row.get("status") == "completed" and (m.get("state") or {}).get("valid") is True
            and (m.get("shape") or {}).get("valid") is True)


def select_cases(records, branches):
    """Select from saved numeric labels/flags before looking at any image."""
    rows = {(r["anchor"], r["candidate"], r["mode"], r["horizon"]): r for r in records
            if r.get("shape") == "L" and r.get("seed") == SEED and r.get("arm") == "B"}
    labels = {(r["anchor"], r["candidate"]): r for r in branches
              if r.get("shape") == "L" and r.get("status") == "complete"}
    groups = {name: [] for name in RULES["categories"]}
    for key in sorted(labels):
        branch = labels[key]
        ar, tf = rows.get((*key, "AR", 4)), rows.get((*key, "TF", 4))
        if ar is None or tf is None:
            continue
        contact = any(s.get("contacts", 0) > 0 for s in branch["actual"])
        first = branch["latest"]["pose"]["angle"]
        last = branch["actual"][3]["truth"]["pose"]["angle"]
        rotation = abs(math.degrees(math.atan2(math.sin(last-first), math.cos(last-first))))
        if contact and not _joint(ar): groups["contact_h4_rejected"].append(key)
        if rotation > 5 and not _joint(ar): groups["rotation_gt5_h4_rejected"].append(key)
        if not _joint(ar) and _joint(tf): groups["AR_rejected_TF_recovered"].append(key)
        if not _joint(ar) and not _joint(tf): groups["AR_TF_both_rejected"].append(key)
        if not contact and key[1] == "hold" and _joint(ar) and _joint(tf): groups["noncontact_hold_control"].append(key)
    for name in ("AR_rejected_TF_recovered", "AR_TF_both_rejected", "noncontact_hold_control"):
        groups[name] = groups[name][:2]
    for name in ("contact_h4_rejected", "rotation_gt5_h4_rejected"):
        if not groups[name]: continue
        key = (groups[name][0][0], "hold")
        if key in labels and (*key, "AR", 4) in rows and (*key, "TF", 4) in rows:
            groups["matched_hold_control"].append(key)
    selected = sorted({key for group in groups.values() for key in group})
    return [{"anchor": a, "candidate": c, "seed": SEED,
             "categories": [name for name, group in groups.items() if (a, c) in group]}
            for a, c in selected]


def display_rgb(raw):
    raw = np.asarray(raw)
    if raw.shape != (3, 96, 96) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("Expected finite saved CHW96 RGB")
    return np.moveaxis(np.rint(np.clip(raw, 0, 1)*255).astype(np.uint8), 0, -1)


def color_line_residual(rgb):
    """Display diagnostic, with no shape/goal/state-dependent color selection."""
    colors = dict(W=(255,255,255), Q=(211,211,211), P0=(65,105,225), P1=(78,126,255),
                  G0=(119,136,153), G1=(143,163,184), R0=(165,28,48), R1=(198,34,58),
                  V=(144,238,144), K=(0,0,0))
    segments = (("W","Q"),("W","P0"),("P0","P1"),("W","P1"),("W","G0"),
                ("G0","G1"),("W","G1"),("W","R0"),("R0","R1"),("W","R1"),
                ("W","V"),("W","K"),("V","G0"),("V","G1"),("K","R0"),("K","R1"))
    result = np.full(rgb.shape[:2], np.inf)
    for first, second in segments:
        u, v = np.asarray(colors[first]), np.asarray(colors[second])
        alpha = np.clip(np.sum((rgb-u)*(v-u), axis=-1)/np.sum((v-u)**2), 0, 1)
        result = np.minimum(result, np.sqrt(np.mean((rgb-u-alpha[...,None]*(v-u))**2, axis=-1))/255)
    return result


def frame_diagnostics(raw, measurement, shape_name="L"):
    """Reuse original object fit, extracting full frozen pusher diagnostics only."""
    from . import wm_image_measurement_v2 as frozen
    display_rgb(raw)  # Validation; measurement retains original raw values.
    rgb = np.moveaxis(np.clip(raw, 0, 1), 0, -1).astype(float)*255
    alpha, blue, palette = frozen.continuous_maps(rgb)
    pusher = frozen.measure_pusher(blue)
    state, shape = measurement.get("state") or {}, measurement.get("shape") or {}
    pose = state.get("candidate_pose") or state.get("pose")
    template = None
    if pose is not None:
        values = (*pose["position"], pose["angle"], shape.get("fitted_outline", 2.), shape.get("fitted_blur_sigma", 0.))
        template = frozen._template(frozen.official_shape(shape_name), frozen.WORLD_X, frozen.WORLD_Y, values, shape.get("palette", palette))
    old_pusher = (measurement.get("pusher") or {}).get("position")
    agreement = (pusher.get("valid") == (measurement.get("pusher") or {}).get("valid")
                 and (old_pusher is None or (pusher.get("position") is not None
                      and np.allclose(pusher["position"], old_pusher, atol=1e-6, rtol=0))))
    summary = dict(raw_min=float(raw.min()), raw_max=float(raw.max()),
                   display_clipped_channel_fraction=float(((raw < 0)|(raw > 1)).mean()),
                   display_clipped_pixel_fraction=float(np.any((raw < 0)|(raw > 1), axis=0).mean()),
                   offline_frozen_pusher=pusher, original_pusher_agreement=bool(agreement),
                   original_candidate_pose=deepcopy(pose), original_state_valid=state.get("valid"),
                   original_shape_reason=shape.get("reason"), original_soft_iou=shape.get("full_soft_iou"),
                   reconstructed_soft_iou=None if template is None else frozen.soft_iou(alpha, template))
    return dict(alpha=alpha, blue=blue, template=template,
                fit_residual=None if template is None else np.abs(alpha-template),
                color_residual=color_line_residual(rgb), summary=summary)


def _json(path):
    return json.loads(Path(path).read_bytes())


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_hash(value):
    return hashlib.sha256(value.dtype.str.encode()+repr(value.shape).encode()+value.tobytes()).hexdigest()


def _saved_raw(row, root):
    expected = "predictions/"+row["key"].replace("/", "-")+".npy"
    if row["raw_path"] != expected:
        raise ValueError("Saved prediction key/path mismatch")
    path = root/expected
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Saved prediction escaped its input root")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or _array_hash(value) != row["raw_sha256"]:
        raise ValueError("Saved prediction array hash/dtype mismatch")
    display_rgb(value)
    return value, path


def panel_labels(stream, horizon, measurement, summary):
    """Explicit cell identity prevents adjacent titles from looking shared."""
    state, shape, pusher = (measurement.get(name) or {} for name in ("state", "shape", "pusher"))
    identity = stream+" "+("latest" if horizon==0 else "H%d"%horizon)
    valid = lambda value: "OK" if value is True else "REJECT"
    value = lambda key: "NA" if shape.get(key) is None else "%.3f"%shape[key]
    title = (identity+" | S="+valid(state.get("valid"))+" P="+valid(pusher.get("valid"))
             +"\n"+shape.get("reason", "not_recorded")
             +"\nIoU="+value("full_soft_iou")+" area="+value("area_ratio"))
    caption = "raw %.3f..%.3f\nclip(ch) %.2f%%" % (summary["raw_min"],summary["raw_max"],100*summary["display_clipped_channel_fraction"])
    return title, caption


def _draw_pages(frames, branch, destination, label):
    os.environ.setdefault("MPLCONFIGDIR", str(destination/".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from . import wm_image_measurement_v2 as frozen
    streams = ("REAL", "A AR", "B AR", "B TF")
    pages = []
    for kind in ("rgb", "measurements", "fit-residual", "color-residual"):
        fig, axes = plt.subplots(4, 5, figsize=(16.5, 14.2), dpi=135)
        for ri, stream in enumerate(streams):
            for h in range(5):
                ax = axes[ri, h]; ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlim(-.5,511.5); ax.set_ylim(511.5,-.5)
                entry = frames.get((stream, h))
                if entry is None:
                    text = "H1 reused from AR\n(no separate TF call)" if stream=="B TF" and h==1 else "No separate frame"
                    ax.text(.5,.5,text,ha="center",va="center",fontsize=8,transform=ax.transAxes)
                    ax.set_facecolor("#f0f0f0")
                    ax.set_title(stream+" "+("initial context" if h==0 else "H%d"%h),fontsize=8.5)
                else:
                    raw, measurement, diagnostic = entry["raw"], entry["measurement"], entry["diagnostic"]
                    if kind in ("rgb", "measurements"):
                        image = display_rgb(raw)
                        if kind=="measurements": image = np.rint(image*.55+255*.45).astype(np.uint8)
                        ax.imshow(image,extent=(-.5,511.5,511.5,-.5),interpolation="nearest")
                    else:
                        field = diagnostic["fit_residual"] if kind=="fit-residual" else diagnostic["color_residual"]
                        if field is not None:
                            ax.imshow(field,extent=(-.5,511.5,511.5,-.5),interpolation="nearest",cmap="magma",vmin=0,vmax=1 if kind=="fit-residual" else .10)
                        else: ax.text(.5,.5,"No original object fit",ha="center",va="center",fontsize=8,transform=ax.transAxes)
                    if kind=="measurements":
                        shape = frozen.official_shape("L")
                        def outline(pose,color,style):
                            if pose is not None:
                                sdf=frozen.signed_distance(shape,frozen.WORLD_X,frozen.WORLD_Y,pose["position"],pose["angle"])
                                ax.contour(frozen.WORLD_X,frozen.WORLD_Y,sdf,levels=[0],colors=[color],linewidths=1.1,linestyles=style)
                        state = measurement.get("state") or {}
                        outline(state.get("candidate_pose") or state.get("pose"),"magenta","solid" if state.get("valid") else "dashed")
                        outline(branch["initial_state"]["goal"],"black","dotted")
                        outline(entry["truth"].get("pose"),"#00a650","dashed")
                        if diagnostic["alpha"].max()>.5:
                            ax.contour(frozen.WORLD_X,frozen.WORLD_Y,diagnostic["alpha"],levels=[.5],colors=["#e69f00"],linewidths=.6)
                        pusher=diagnostic["summary"]["offline_frozen_pusher"]
                        position=pusher.get("position") or pusher.get("candidate_position")
                        if position is not None:
                            ax.add_patch(Circle(position, frozen.CONFIG["pusher_gate"]["radius"], fill=False,
                                edgecolor="cyan" if pusher["valid"] else "#e69f00",linestyle="solid" if pusher["valid"] else "dashed",linewidth=1))
                        for component in pusher.get("meaningful_components",[]):
                            ax.plot(*component["position"],marker=".",color="cyan",markersize=3)
                        if entry["truth"].get("pusher") is not None:ax.plot(*entry["truth"]["pusher"],marker="x",color="#00a650",markersize=4)
                        if h:ax.plot(*branch["actual"][h-1]["action"],marker="+",color="#d55e00",markersize=6)
                    title, caption = panel_labels(stream,h,measurement,diagnostic["summary"])
                    ax.set_title(title,fontsize=8.5)
                    ax.set_xlabel(caption,fontsize=7.5)
                if h==0:ax.set_ylabel(stream,fontsize=11)
        subtitle = {"rgb":"Saved pixels: fixed display clip/round, no measurement or feedback changes",
                    "measurements":"Magenta: original rigid fit (dashed=rejected); orange: alpha .5; cyan: frozen pusher; green dashed/x: truth labels; black dotted: goal; orange +: action",
                    "fit-residual":"Absolute object-alpha minus ORIGINAL fitted template; fixed scale 0..1 (not RGB error/probability)",
                    "color-residual":"Fixed palette-line RGB RMS residual /255; fixed display scale 0..0.10 (independent of shape fit)"}[kind]
        fig.suptitle(label+" | "+kind+"\n"+subtitle,fontsize=9)
        fig.subplots_adjust(left=.045,right=.995,top=.91,bottom=.045,hspace=.60,wspace=.16)
        name=label.replace("/","-")+"-"+kind+".png";path=destination/name
        fig.savefig(path,metadata={"Software":"Tdyssey offline saved-file diagnostics"});plt.close(fig)
        pages.append(dict(kind=kind,path=str(path),bytes=path.stat().st_size,sha256=_digest(path)))
    return pages


def export_case_figures(run_dir, physical_dir, destination, *, figure_subdir="figures", manifest_name="visual-cases.json"):
    """Freeze selection before RGB reads, then export display-only CPU evidence."""
    run_dir, physical_dir, destination = map(Path,(run_dir, physical_dir, destination))
    root=run_dir/"results" if (run_dir/"results").is_dir() else run_dir
    physical_report=_json(physical_dir/"report.json")
    record_paths=sorted((root/"records").glob("L*-5478-*.json"))
    records=[_json(path) for path in record_paths]
    selected=select_cases(records,physical_report["branches"])
    if figure_subdir not in ("figures","figures-v2") or manifest_name not in ("visual-cases.json","visual-cases-v2.json"):
        raise ValueError("Undeclared diagnostic layout output name")
    figures=destination/figure_subdir;figures.mkdir(parents=True,exist_ok=True)
    selection=dict(rules=RULES,rules_sha256=RULES_SHA256,cases=selected,
                   diagnostic_source_sha256=_digest(Path(__file__)),
                   frozen_measurement_source_sha256=_digest(Path(__file__).with_name("wm_image_measurement_v2.py")),
                   frozen_measurement_config_sha256=_digest(Path(__file__).resolve().parents[1]/"configs/wm_image_measurement_v2.json"),
                   numeric_inputs_sha256={str(path):_digest(path) for path in record_paths},
                   physical_report_sha256=_digest(physical_dir/"report.json"),RGB_reads_before_rule_freeze=0)
    if figure_subdir=="figures-v2":
        original=destination/"figures/selection-frozen.json";old=_json(original)
        if old["rules_sha256"]!=RULES_SHA256 or old["cases"]!=selected:
            raise ValueError("Layout repair changed the frozen scientific case selection")
        selection["original_frozen_selection"]=dict(path=str(original),sha256=_digest(original))
    frozen_path=figures/"selection-frozen.json"
    if frozen_path.exists():
        if _json(frozen_path)!=selection:raise ValueError("Existing frozen case selection differs")
    else:
        with frozen_path.open("x",encoding="utf-8") as stream:json.dump(selection,stream,indent=2,sort_keys=True)
    lookup={(r["anchor"],r["candidate"],r["mode"],r["arm"],r["horizon"]):r for r in records}
    branches={(b["anchor"],b["candidate"]):b for b in physical_report["branches"]}
    manifest=dict(kind="wm_l_saved_visual_cases.v1",selection_file=str(frozen_path),selection_sha256=_digest(frozen_path),
                  rules_sha256=RULES_SHA256,status="selection_frozen",cases=[],new_model_loads=0,new_model_forwards=0,
                  environment_creations=0,resets=0,controls=0,physics_steps=0,policy_calls=0,training_updates=0,
                  full_decoder_calls=0,offline_frozen_pusher_measurements=0)
    target=destination/manifest_name
    def save():
        with target.open("w",encoding="utf-8",newline="\n") as stream:json.dump(manifest,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    save()  # No NPZ/NPY/RGB has been read yet.
    for selected_case in selected:
        anchor,candidate=selected_case["anchor"],selected_case["candidate"];branch=branches[anchor,candidate]
        physical_path=physical_dir/f"{anchor}-{candidate}-physical.npz"
        with np.load(physical_path,allow_pickle=False) as arrays:actual=arrays["actual"].copy();latest=arrays["history"][-1].copy()
        frames={};summaries=[]
        def add(stream,h,raw,measurement,truth,source):
            diagnostic=frame_diagnostics(raw,measurement)
            frames[stream,h]=dict(raw=raw,measurement=measurement,diagnostic=diagnostic,truth=truth)
            summaries.append(dict(stream=stream,horizon=h,source=str(source),**diagnostic["summary"]))
            summaries[-1]["panel_title"]=panel_labels(stream,h,measurement,diagnostic["summary"])[0]
            manifest["offline_frozen_pusher_measurements"]+=1
        path=root/"real"/f"{anchor}-{candidate}-history-3-A.json"
        add("REAL",0,latest,_json(path)["measurement"],branch["latest"],path)
        for h in range(1,5):
            path=root/"real"/f"{anchor}-{candidate}-future-H{h}-A.json"
            add("REAL",h,actual[h-1],_json(path)["measurement"],branch["actual"][h-1]["truth"],path)
            for arm,mode in (("A","AR"),("B","AR"),("B","TF")):
                if mode=="TF" and h not in (2,4):continue
                row=lookup[anchor,candidate,mode,arm,h];raw,path=_saved_raw(row,root)
                add(arm+" "+mode,h,raw,row["measurement"],row["truth"],path)
        label=f"{anchor}/{candidate}/{SEED}"
        pages=_draw_pages(frames,branch,figures,label)
        manifest["cases"].append(dict(**selected_case,physical_source=str(physical_path),physical_sha256=_digest(physical_path),
            contact_by_H4=any(s.get("contacts",0)>0 for s in branch["actual"]),figures=pages,frames=summaries))
        save()
    manifest.update(status="complete",figure_count=sum(len(c["figures"]) for c in manifest["cases"]),
                    total_figure_bytes=sum(p["bytes"] for c in manifest["cases"] for p in c["figures"]))
    save()
    return manifest
