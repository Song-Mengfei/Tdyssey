"""Present recorded R3A contact timelines without model or physics execution.

The original representative MP4s/strips are linked, never rewritten. Actual
post-solve flags and controller labels are read from compact recorded arrays.
This presentation is a companion to the separate full-library numerical audit.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PHASES = ("approach", "push", "withdraw", "recontact", "static")
MODES = ("hold", "free_route", "contact_push_intent")
PHASE_COLORS = ("#cbd5e1", "#b6e2c0", "#facb9b", "#e6c3ef", "#d7d7d7")
FOCUS_ID = "R3A-seen_val-B-014-a0"
FRAME_INDICES = (0, 12, 24, 36, 48)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(path, parent):
    return Path(os.path.relpath(Path(path).resolve(), Path(parent).resolve())).as_posix()


def recorded_timeline(path, metadata):
    """Read only saved states; frame j follows control j-1, never control j."""
    from tdyssey.envs.official import official_shape

    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in (
            "states", "control_time_seconds", "contact_flags", "control_phase",
            "control_mode", "control_substep_offsets", "substep_status")}
    states = arrays["states"]
    times = arrays["control_time_seconds"]
    offsets = arrays["control_substep_offsets"].astype(np.int64)
    flags = arrays["contact_flags"]
    if (states.shape != (49, 15) or flags.shape != (480, 3)
            or offsets.shape != (49,) or not np.array_equal(offsets, np.arange(49) * 10)
            or not np.all(arrays["substep_status"] == 1)
            or not np.allclose(times, np.arange(49) / 10, rtol=0, atol=1e-12)):
        raise ValueError("Presentation requires a complete 48-control recorded trajectory")
    post = (flags[:, 0] & 2) != 0
    counts = post.reshape(48, 10).sum(axis=1)
    endpoint = post[offsets[1:] - 1]
    cog = np.asarray(official_shape(metadata["letter"]).centroid)
    c, s = np.cos(states[:, 4]), np.sin(states[:, 4])
    centroid = states[:, 2:4] + np.column_stack((c*cog[0]-s*cog[1], s*cog[0]+c*cog[1]))
    translation = np.linalg.norm(np.diff(centroid, axis=0), axis=1) * 96 / 512
    rotation = np.diff(states[:, 4]) * 180 / np.pi
    controls = []
    for i in range(48):
        phase, mode = int(arrays["control_phase"][i]), int(arrays["control_mode"][i])
        controls.append(dict(index=i, start_seconds=float(times[i]), end_seconds=float(times[i+1]),
            planned_phase=PHASES[phase], control_mode=MODES[mode], postsolve_contact_substeps=int(counts[i]),
            postsolve_contact_at_endpoint=bool(endpoint[i]), translation_image96_px=float(translation[i]),
            signed_rotation_deg=float(rotation[i]), nearstatic=bool(translation[i] < .1 and abs(rotation[i]) < .1),
            obvious_motion=bool(translation[i] >= 2 or abs(rotation[i]) >= 5)))
    frames = []
    for j in FRAME_INDICES:
        prior = controls[j-1] if j else None
        frames.append(dict(frame_index=j, timestamp_seconds=float(times[j]),
            preceding_control=None if j == 0 else j-1,
            actual_postsolve_contact=None if prior is None else prior["postsolve_contact_at_endpoint"],
            postsolve_contact_substeps=None if prior is None else prior["postsolve_contact_substeps"],
            planned_phase=None if prior is None else prior["planned_phase"],
            control_mode=None if prior is None else prior["control_mode"]))
    recontact_indices = [item["index"] for item in controls if item["planned_phase"] == "recontact"]
    recontact = None
    if recontact_indices:
        start, end = recontact_indices[0] * 10, (recontact_indices[-1]+1) * 10
        observed = np.flatnonzero(post[start:])
        first = None if not len(observed) else int(start+observed[0])
        recontact = dict(planned_start_seconds=float(start/100), planned_end_seconds=float(end/100),
            actual_postsolve_contact_substeps_in_planned_phase=int(post[start:end].sum()),
            first_actual_postsolve_global_substep=first,
            first_actual_contact_interval_start_seconds=None if first is None else float(first/100),
            recontact_interval_start_delay_seconds=None if first is None else float((first-start)/100),
            first_actual_postsolve_state_time_seconds=None if first is None else float((first+1)/100))
    return dict(trajectory_id=metadata["trajectory_id"], letter=metadata["letter"], split=metadata["split"],
        collection_scenario=metadata["scenario"], controls=controls, displayed_frames=frames,
        recontact=recontact,
        postsolve_agent_object_substeps=int(post.sum()), postsolve_agent_object_seconds=float(post.sum()/100))


def timeline_svg(row):
    width, left, cell = 1040, 178, 17
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="178" viewBox="0 0 {width} 178">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#17202b">']
    for title, y in (("Planned phase", 39), ("Controller mode", 67),
                     ("Actual postsolve AO", 99), ("Observed motion", 130)):
        parts.append(f'<text x="4" y="{y}">{title}</text>')
    for control in row["controls"]:
        i = control["index"]; x = left + i*cell
        phase = PHASES.index(control["planned_phase"])
        mode = MODES.index(control["control_mode"])
        count = control["postsolve_contact_substeps"]
        desc = html.escape(json.dumps(control))
        parts.append(f'<g><title>{desc}</title><rect x="{x}" y="25" width="16" height="20" fill="{PHASE_COLORS[phase]}"/>')
        parts.append(f'<rect x="{x}" y="53" width="16" height="20" fill="{("#ddd", "#a8c4ee", "#4773ad")[mode]}"/>')
        parts.append(f'<rect x="{x}" y="82" width="16" height="22" fill="#f2f2f2"/>')
        parts.append(f'<rect x="{x}" y="{104-count*2.2}" width="16" height="{count*2.2}" fill="#176d4b"/>')
        color = "#bdbdbd" if control["nearstatic"] else ("#df7424" if control["obvious_motion"] else "#f2ce73")
        parts.append(f'<rect x="{x}" y="116" width="16" height="20" fill="{color}"/></g>')
    for i in (0, 12, 24, 36, 48):
        x = left+i*cell
        parts.append(f'<line x1="{x}" x2="{x}" y1="18" y2="141" stroke="#89909b" stroke-dasharray="2 3"/><text x="{x-8}" y="157">{i/10:.1f}s</text>')
    parts.append('</g></svg>')
    return "".join(parts)


def frame_table(row):
    columns = []
    for frame in row["displayed_frames"]:
        if frame["preceding_control"] is None:
            text = '<b>t=0.0s, frame 0</b><br>Initial state<br>Postsolve: N/A<br>Previous phase/mode: N/A'
        else:
            contact = "yes" if frame["actual_postsolve_contact"] else "no"
            text = (f'<b>t={frame["timestamp_seconds"]:.1f}s, frame {frame["frame_index"]}</b>'
                f'<br>Postsolve at t: {contact}<br>Previous control {frame["preceding_control"]}: '
                f'{frame["postsolve_contact_substeps"]}/10 contact substeps<br>'
                f'Plan: {frame["planned_phase"]}<br>Mode: {frame["control_mode"]}')
        columns.append(f'<td>{text}</td>')
    return '<table class="frames"><tr>'+"".join(columns)+'</tr></table>'


def annotated_focus_video(original, row, output):
    """Annotate a lossy original preview outside its unchanged decoded image."""
    import cv2

    capture = cv2.VideoCapture(str(original))
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != 49 or abs(capture.get(cv2.CAP_PROP_FPS)-10) > 1e-8:
        raise ValueError("Original preview timeline differs")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 10., (960, 560))
    if not writer.isOpened():
        raise RuntimeError("Existing OpenCV MP4 encoder unavailable")
    try:
        for j in range(49):
            good, image = capture.read()
            if not good or image.shape != (384, 384, 3):
                raise ValueError("Original preview decode is incomplete")
            canvas = np.full((560, 960, 3), 250, np.uint8)
            canvas[16:400, 16:400] = image
            control = row["controls"][j-1] if j else None
            lines = [row["trajectory_id"], "Collection scenario: release_recontact",
                "Old display role: B:translation", f"Recorded state {j}, t={j/10:.2f}s",
                "Actual AO contact: N/A (initial)" if not control else
                f'Actual postsolve AO at t: {"YES" if control["postsolve_contact_at_endpoint"] else "NO"}',
                "Previous interval: N/A" if not control else
                f'Control {j-1}: {control["postsolve_contact_substeps"]}/10 contact substeps',
                "Plan/mode: N/A" if not control else f'Plan: {control["planned_phase"]}',
                "" if not control else f'Mode: {control["control_mode"]}',
                f'Planned recontact: {row["recontact"]["planned_start_seconds"]:.2f}-{row["recontact"]["planned_end_seconds"]:.2f}s',
                'Contact event: substep 391, [3.91,3.92]s',
                'Main delay: 1.31s to interval start',
                'Saved post-state: 3.92s (delay 1.32s)',
                "No contact during planned recontact.", "Control 31-37: planned push, free_route.",
                "Control 38: push intent, still no contact."]
            for k, line in enumerate(lines):
                cv2.putText(canvas, line, (420, 29+k*25), cv2.FONT_HERSHEY_SIMPLEX, .54, (40, 40, 40), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Postsolve AO contact substeps per 0.1s interval (0-10)", (16, 429), cv2.FONT_HERSHEY_SIMPLEX, .55, (35, 35, 35), 1, cv2.LINE_AA)
            for i, item in enumerate(row["controls"]):
                x = 17+i*19
                cv2.rectangle(canvas, (x, 446), (x+17, 516), (228, 228, 228), -1)
                n = item["postsolve_contact_substeps"]
                if n:
                    cv2.rectangle(canvas, (x, 516-n*7), (x+17, 516), (75, 123, 35), -1)
                if i == j-1:
                    cv2.rectangle(canvas, (x-1, 444), (x+18, 518), (20, 20, 230), 2)
            for i in (0, 12, 24, 36, 48):
                cv2.putText(canvas, f"{i/10:.1f}s", (max(0, 17+i*19-8), 541), cv2.FONT_HERSHEY_SIMPLEX, .47, (35, 35, 35), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        capture.release(); writer.release()
    decoded = cv2.VideoCapture(str(output))
    count, fps = int(decoded.get(cv2.CAP_PROP_FRAME_COUNT)), decoded.get(cv2.CAP_PROP_FPS)
    qa_frames = []
    for frame_index in (36, 40):
        decoded.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        good, image = decoded.read()
        if not good:
            raise ValueError("Annotated preview QA decode failed")
        qa = output.with_name(f'B014-contact-regression-{frame_index/10:.1f}s-QA.png')
        cv2.imwrite(str(qa), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        qa_frames.append(dict(frame_index=frame_index, file=qa.name, bytes=qa.stat().st_size, sha256=sha(qa)))
    decoded.release()
    if count != 49 or abs(fps-10) > 1e-8:
        raise ValueError("Annotated preview frame/fps mismatch")
    return dict(file=output.name, sha256=sha(output), bytes=output.stat().st_size,
        frames=count, fps=fps, qa_frames=qa_frames, source_preview=str(original.resolve()),
        source_preview_sha256=sha(original), source_type="lossy original MP4 preview, decoded image copied; annotations outside image")


def supplemental_selection(core_dir, entries):
    """Select measured TRAIN examples, never use intended scenario as a score."""
    core_dir = Path(core_dir)
    core_meta = read(core_dir/"trajectories.json")
    trajectories = core_meta["trajectories"]
    if len(trajectories) != len(entries) or any(
            item["trajectory_index"] != i or item["trajectory_id"] != entries[i]["trajectory_id"]
            or item["source_npz_sha256"] != entries[i]["sha256"]
            for i, item in enumerate(trajectories)):
        raise ValueError("Core and source manifest trajectory identity/order differ")
    with np.load(core_dir/"controls.npz", allow_pickle=False) as archive:
        core = {key: archive[key] for key in ("trajectory_index", "control", "contact_substeps",
            "terminal_contact", "phase", "mode", "translation_image96_px", "signed_rotation_deg",
            "object_wall_substeps", "nonzero_impulse_counts", "motion_class")}
    n = len(entries)
    if (not np.array_equal(core["trajectory_index"], np.repeat(np.arange(n), 48))
            or not np.array_equal(core["control"], np.tile(np.arange(48), n))):
        raise ValueError("Expected canonical core control rows, trajectory_index*48+control")
    contact = core["contact_substeps"].reshape(n, 48)
    motion = core["motion_class"].reshape(n, 48)
    wall = core["object_wall_substeps"].reshape(n, 48)
    translation = core["translation_image96_px"].reshape(n, 48)
    rotation = core["signed_rotation_deg"].reshape(n, 48)
    impulse = core["nonzero_impulse_counts"][:, 1].reshape(n, 48) > 0
    train = np.asarray([i for i, item in enumerate(trajectories) if item["split"] == "train"])
    rng = np.random.default_rng(5478)
    selected = {}

    def add(index, role, metric, value, supported=True):
        record = selected.setdefault(int(index), dict(trajectory_id=entries[int(index)]["trajectory_id"],
            trajectory_index=int(index), roles=[], measurements=[]))
        record["roles"].append(role)
        record["measurements"].append(dict(role=role, metric=metric, value=float(value), supported=bool(supported)))

    for letter in "TLOCXB":
        pool = [int(i) for i in train if entries[int(i)]["letter"] == letter]
        index = int(rng.choice(pool))
        add(index, f"fixed_seed_random_{letter}", "seed (uniform within TRAIN letter)", 5478)
    sums = contact.sum(axis=1)
    near = (motion == 0).sum(axis=1)
    least = min(train, key=lambda i: (int(sums[i]), -int(near[i]), int(i)))
    add(least, "least_actual_contact", "postsolve AO substeps / 480", sums[least])
    most_near = min(train, key=lambda i: (-int(near[i]), int(sums[i]), int(i)))
    add(most_near, "most_nearstatic", "nearstatic controls / 48", near[most_near])
    recovered = [int(i) for i in train if trajectories[int(i)].get("recontact_latency_seconds") is not None]
    if recovered:
        index = max(recovered, key=lambda i: (trajectories[i]["recontact_latency_seconds"], -i))
        add(index, "slowest_observed_recontact", "delay to first actual contact interval start seconds",
            trajectories[index]["recontact_latency_seconds"])
    censored = [int(i) for i in train if trajectories[int(i)].get("recontact_censored") is True]
    if censored:
        index = max(censored, key=lambda i: (trajectories[i]["recontact_censor_seconds"], -i))
        add(index, "unrecovered_recontact", "right-censored observation duration seconds; not a recovery latency",
            trajectories[index]["recontact_censor_seconds"])
    stalls = ((wall > 0) & (motion == 0)).sum(axis=1)
    index = max(train, key=lambda i: (int(stalls[i]), -int(i)))
    add(index, "wall_stall", "object-wall postsolve AND nearstatic controls", stalls[index], stalls[index] > 0)
    actual = (contact > 0) & impulse
    for role, values, threshold in (
            ("observed_contact_translation", translation, 2.),
            ("observed_contact_positive_rotation", rotation, 5.),
            ("observed_contact_negative_rotation", -rotation, 5.)):
        scores = np.where(actual, values, 0.).max(axis=1)
        index = max(train, key=lambda i: (float(scores[i]), -int(i)))
        unit = "image96px" if role.endswith("translation") else "degrees"
        add(index, role, "max within control with postsolve AO and valid impulse norm>1e-8; "+unit,
            scores[index], scores[index] >= threshold)
    return list(selected.values()), core, dict(
        controls_sha256=sha(core_dir/"controls.npz"), trajectories_sha256=sha(core_dir/"trajectories.json"),
        seed=5478, selection_pool="all 768 TRAIN trajectories; no seen-val selection", unique_examples=len(selected),
        caution="Biased measurement examples, not prevalence estimates. Contact+movement co-occurrence does not prove causality.")


def recorded_strip(source, entry, renderer, output):
    import cv2
    with np.load(source/entry["file"], allow_pickle=False) as archive:
        states = archive["states"]
        hashes = archive["rgb_frame_sha256"]
    frames, verified = [], []
    for frame_index in FRAME_INDICES:
        chw = renderer.render(entry["letter"], states[frame_index])
        actual = hashlib.sha256(chw.tobytes()).digest()
        expected = bytes(hashes[frame_index])
        if actual != expected:
            raise ValueError("Supplemental saved-state RGB differs from original collection frame hash")
        verified.append(dict(frame_index=frame_index, timestamp_seconds=frame_index/10,
            original_collection_rgb_sha256=expected.hex(), saved_state_render_rgb_sha256=actual.hex()))
        frame = np.moveaxis(chw, 0, -1)
        frames.append(cv2.resize(frame, (192, 192), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(output), cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return dict(file=output.name, bytes=output.stat().st_size, sha256=sha(output),
        pixel_exact_saved_original_frames=5, verified_frames=verified,
        resize="96 to192 nearest; original RGB only; no annotation inside image")


def core_check(row, entry_index, core):
    sl = slice(entry_index*48, (entry_index+1)*48)
    pairs = (("contact_substeps", "postsolve_contact_substeps"), ("terminal_contact", "postsolve_contact_at_endpoint"),
        ("translation_image96_px", "translation_image96_px"), ("signed_rotation_deg", "signed_rotation_deg"))
    for core_key, row_key in pairs:
        if not np.allclose(core[core_key][sl], [c[row_key] for c in row["controls"]], rtol=0, atol=1e-9):
            raise ValueError(f"Presentation/core mismatch: {row['trajectory_id']} {core_key}")
    if (not np.array_equal(core["phase"][sl], [PHASES.index(c["planned_phase"]) for c in row["controls"]])
            or not np.array_equal(core["mode"][sl], [MODES.index(c["control_mode"]) for c in row["controls"]])):
        raise ValueError("Presentation/core controller mismatch")


def supplemental_cards(choices, existing, entries, source, source_root, out, core):
    from tdyssey.native_rgb_r3a_data import RecordedStateRenderer
    renderer = RecordedStateRenderer(source_root, maximum_frame_cache_bytes=0)
    cards, new_rows, links = [], [], []
    try:
        for choice in choices:
            entry = entries[choice["trajectory_index"]]
            trajectory_id = entry["trajectory_id"]
            labels = html.escape(", ".join(choice["roles"]))
            measures = "; ".join(html.escape(f'{m["metric"]}: {m["value"]:.6g}'+(' [below role threshold]' if not m["supported"] else ''))
                for m in choice["measurements"])
            links.append(f'<li><a href="#{trajectory_id}">{trajectory_id}</a>: {labels}. {measures}</li>')
            if trajectory_id in existing:
                continue
            metadata = read(source/entry["metadata_file"])
            if sha(source/entry["file"]) != entry["sha256"]:
                raise ValueError("Supplemental source hash differs")
            row = recorded_timeline(source/entry["file"], metadata)
            core_check(row, choice["trajectory_index"], core)
            row.update(display_roles=choice["roles"], source_file=entry["file"], source_sha256=entry["sha256"])
            strip = recorded_strip(source, entry, renderer, out/(trajectory_id+"-measured.png"))
            row["new_strip"] = strip
            svg = trajectory_id+"-timeline.svg"
            (out/svg).write_text(timeline_svg(row), encoding="utf-8")
            row["timeline_svg"] = svg
            cards.append(f'<section id="{trajectory_id}"><h2>{trajectory_id}</h2><p><b>Collection scenario:</b> '
                f'{html.escape(row["collection_scenario"])}; <b>Measured display role:</b> {labels}; TRAIN.</p>'
                f'<p>{measures}</p><img class="strip" src="{strip["file"]}">{frame_table(row)}<img class="timeline" src="{svg}"></section>')
            new_rows.append(row)
    finally:
        renderer.close()
    return '<h2>Measured TRAIN examples, seed 5478 and explicit extremes</h2><p>Selections use actual core measurements, not intended scenario. Duplicate trajectory choices merge roles. Unrecovered recontact is right-censored, not assigned a recovery time.</p><ul>'+''.join(links)+'</ul>'+''.join(cards), new_rows, renderer.stats


def make_figures(core_dir, statistics_dir, output):
    """Plot frozen independent denominators and actual per-index duty."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    core_dir, statistics_dir, output = Path(core_dir), Path(statistics_dir), Path(output)
    summary = read(statistics_dir/"main_summary.json")
    with np.load(core_dir/"controls.npz", allow_pickle=False) as archive:
        split, control, contact, motion = (archive[k] for k in ("split", "control", "contact_substeps", "motion_class"))
    with np.load(statistics_dir/"input-exposure.npz", allow_pickle=False) as archive:
        selected_rows = archive["selected_target_control_rows"]
    if len(np.unique(selected_rows)) != 23984 or np.any(selected_rows < 0) or np.any(selected_rows >= len(split)):
        raise ValueError("Expected 23984 unique selected target global control rows")
    selected = np.zeros(len(split), bool); selected[selected_rows] = True
    if np.any(split[selected] != 0):
        raise ValueError("Selected TRAIN targets include seen-val controls")
    masks = [split == 0, (split == 0) & (control >= 3), selected, split == 1]
    keys = ["A_TRAIN_all", "B_TRAIN_legal_H1", "C_TRAIN_selected", "VAL_all"]
    names = ["All TRAIN time", "Legal H1 TRAIN targets", "Selected TRAIN targets", "Seen-val time"]
    classes = ["no_contact_nearstatic", "no_contact_small", "no_contact_obvious",
        "contact_nearstatic", "contact_small", "contact_obvious"]
    colors = ["#cbd5e1", "#94a3b8", "#475569", "#fecaca", "#fbbf24", "#16a34a"]
    counts = []
    for mask, key in zip(masks, keys):
        actual = [int(np.count_nonzero(mask & ((contact > 0) == bool(c // 3)) & (motion == c % 3))) for c in range(6)]
        frozen = summary["denominators"][key]["six_mutually_exclusive_classes"]
        if actual != [frozen[category]["count"] for category in classes] or sum(actual) != int(mask.sum()):
            raise ValueError("Plot denominator/classes differ from numerical summary: "+key)
        counts.append(actual)
    counts = np.asarray(counts)
    percent = counts/counts.sum(axis=1, keepdims=True)*100
    with plt.rc_context({"svg.fonttype": "none", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(11, 5.4))
        left = np.zeros(4)
        for i, (category, color) in enumerate(zip(classes, colors)):
            ax.barh(np.arange(4), percent[:, i], left=left, color=color, label=category.replace('_', ' '), height=.64)
            for j in range(4):
                if percent[j, i] >= 4:
                    ax.text(left[j]+percent[j, i]/2, j, f'{percent[j,i]:.1f}%', ha="center", va="center",
                        color="white" if i == 2 else "#17202b", fontsize=9)
            left += percent[:, i]
        ax.set_yticks(np.arange(4), [f'{name}\nN={int(n):,} distinct 0.1s intervals' for name,n in zip(names,counts.sum(axis=1))])
        ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Percent within each independent denominator")
        ax.set_title("Actual agent–object contact × recorded object motion")
        ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(.5, -.17), frameon=False, fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.text(.02, .025, "Contact: any actual 100Hz postsolve AO flag within interval. Nearstatic: translation<0.1 image96px AND |rotation|<0.1°.\n"
            "Obvious: translation≥2 image96px OR |rotation|≥5°. Small: remaining intervals. Rows overlap but have separate denominators; selected is not new acquired time.", fontsize=8)
        fig.subplots_adjust(left=.29, right=.98, top=.88, bottom=.31)
        fig.savefig(output/"contact-motion-independent-denominators.svg")
        fig.savefig(output/"contact-motion-independent-denominators-QA.png", dpi=100)
        plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, sharey=True)
        curves = {}
        for axis, code, label, n in zip(axes, (0, 1), ("TRAIN", "Seen-val"), (768, 144)):
            values = {"AO substep duty": [], "Obvious + any AO contact": [],
                "Nearstatic, all": [], "Nearstatic + any AO contact": []}
            for index in range(48):
                mask = (split == code) & (control == index)
                if int(mask.sum()) != n:
                    raise ValueError("Per-index independent trajectory denominator differs")
                values["AO substep duty"].append(float(contact[mask].mean()*10))
                values["Obvious + any AO contact"].append(float(np.mean((motion[mask] == 2) & (contact[mask] > 0))*100))
                values["Nearstatic, all"].append(float(np.mean(motion[mask] == 0)*100))
                values["Nearstatic + any AO contact"].append(float(np.mean((motion[mask] == 0) & (contact[mask] > 0))*100))
            for (name, value), color, style in zip(values.items(), ("#2563eb", "#16a34a", "#555", "#dc2626"), ("-", "-", "--", ":")):
                axis.plot(range(48), value, label=name, color=color, linestyle=style, linewidth=1.8)
            axis.set_title(f'{label}: {n} independent trajectories per control index', loc="left", fontsize=11)
            axis.set_ylim(0, 100); axis.set_ylabel("Percent"); axis.grid(alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
            curves[label] = values
        axes[1].set_xlabel("Recorded control index i, interval [0.1i, 0.1(i+1)] seconds")
        axes[1].set_xticks([0, 3, 12, 18, 22, 26, 30, 36, 42, 47]); axes[1].set_xlim(0, 47)
        axes[0].legend(ncols=2, fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(.01, 1.38))
        fig.text(.02, .015, "AO duty uses 10×N postsolve substeps/index. Other lines use N recorded intervals/index. Contact uses any postsolve AO flag; motion thresholds match the stacked bars.", fontsize=8)
        fig.subplots_adjust(left=.08, right=.98, top=.82, bottom=.10, hspace=.40)
        fig.savefig(output/"actual-contact-motion-by-control-index.svg"); plt.close(fig)
    result = dict(kind="contact-presentation-figures.v1", denominators=keys, class_order=classes,
        counts=counts.tolist(), per_index_curves_percent=curves,
        files=[dict(file=name, bytes=(output/name).stat().st_size, sha256=sha(output/name)) for name in (
            "contact-motion-independent-denominators.svg", "actual-contact-motion-by-control-index.svg")])
    (output/"figure-provenance.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def overview_qa(choices, rows, output):
    """Small review sheet: full denominators plus three measured examples."""
    import cv2
    output = Path(output)
    plot = cv2.imread(str(output/"contact-motion-independent-denominators-QA.png"))
    if plot is None:
        raise ValueError("Overview needs the completed independent-denominator plot")
    by_id = {row["trajectory_id"]: row for row in rows}
    roles = ("slowest_observed_recontact", "most_nearstatic", "observed_contact_positive_rotation")
    picked = [next(choice for choice in choices if role in choice["roles"]) for role in roles]
    height = 50+plot.shape[0]+len(picked)*300
    canvas = np.full((height, 1120, 3), 255, np.uint8)
    cv2.putText(canvas, "R3A contact audit: independent denominators and recorded TRAIN examples", (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX, .63, (25, 25, 25), 1, cv2.LINE_AA)
    canvas[42:42+plot.shape[0], 10:10+plot.shape[1]] = plot
    y = 50+plot.shape[0]
    for role, choice in zip(roles, picked):
        row = by_id[choice["trajectory_id"]]
        path = output/row["new_strip"]["file"] if "new_strip" in row else output/row["original_strip"]
        strip = cv2.imread(str(path))
        if strip is None or strip.shape != (192, 960, 3):
            raise ValueError("Overview expects the unchanged 5-frame strip")
        measure = next(m for m in choice["measurements"] if m["role"] == role)
        title = f'{role}: {choice["trajectory_id"]}, measured value={measure["value"]:.4g}'
        cv2.putText(canvas, title, (20, y+22), cv2.FONT_HERSHEY_SIMPLEX, .54, (30, 30, 30), 1, cv2.LINE_AA)
        note = 'Frames at 0.0, 1.2, 2.4, 3.6, 4.8s; collection scenario: '+row["collection_scenario"]
        if role == "slowest_observed_recontact":
            r = row["recontact"]
            note += f'; recontact starts {r["planned_start_seconds"]:.2f}s, actual interval starts {r["first_actual_contact_interval_start_seconds"]:.2f}s'
        cv2.putText(canvas, note, (20, y+45), cv2.FONT_HERSHEY_SIMPLEX, .43, (65, 65, 65), 1, cv2.LINE_AA)
        canvas[y+53:y+245, 80:1040] = strip
        for control in row["controls"]:
            i, count = control["index"], control["postsolve_contact_substeps"]
            x = 80+i*20
            cv2.rectangle(canvas, (x, y+252), (x+18, y+281), (235, 235, 235), -1)
            if count:
                cv2.rectangle(canvas, (x, y+281-count*3), (x+18, y+281), (75, 123, 35), -1)
        cv2.putText(canvas, "Actual postsolve AO substeps /10 in each recorded control (all 48 intervals)", (80, y+296),
            cv2.FONT_HERSHEY_SIMPLEX, .40, (50, 50, 50), 1, cv2.LINE_AA)
        y += 300
    path = output/"contact-audit-overview-QA.png"
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
        selected_roles=list(roles), RGB="Original saved-frame strips copied unchanged; chart/labels outside RGB")


def build(args):
    manifest_path = args.manifest.resolve(); manifest = read(manifest_path)
    source = (manifest_path.parent/manifest.get("source_root", ".")).resolve()
    provenance_path = args.original_provenance.resolve(); provenance = read(provenance_path)
    originals = provenance_path.parent
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if out == originals or out.is_relative_to(originals):
        raise ValueError("Supplemental presentation must not write into original media directory")
    entries = {entry["trajectory_id"]: entry for entry in manifest["entries"]}
    entry_indices = {entry["trajectory_id"]: i for i, entry in enumerate(manifest["entries"])}
    choices, core, selection_info = [], None, None
    if args.core_dir is not None:
        if args.source_root is None:
            raise ValueError("Measured supplemental saved-frame rendering requires --source-root")
        choices, core, selection_info = supplemental_selection(args.core_dir, manifest["entries"])
    before = {str(originals/record[key]): sha(originals/record[key])
        for record in provenance["records"] for key in ("file", "strip")}
    rows, cards = [], []
    for record in provenance["records"]:
        entry = entries[record["trajectory_id"]]
        path = source/entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Recorded source hash differs")
        metadata = read(source/entry["metadata_file"])
        row = recorded_timeline(path, metadata)
        if core is not None:
            core_check(row, entry_indices[row["trajectory_id"]], core)
        row.update(display_roles=record["roles"], source_file=entry["file"], source_sha256=entry["sha256"],
            original_video=relative(originals/record["file"], out), original_strip=relative(originals/record["strip"], out))
        svg = row["trajectory_id"]+"-timeline.svg"
        (out/svg).write_text(timeline_svg(row), encoding="utf-8")
        row["timeline_svg"] = svg
        focus = row["trajectory_id"] == FOCUS_ID
        special = ''
        if focus:
            recon = row["recontact"]
            if (recon["actual_postsolve_contact_substeps_in_planned_phase"] != 0
                    or recon["first_actual_postsolve_global_substep"] != 391
                    or any(c["control_mode"] != "free_route" for c in row["controls"][31:38])
                    or row["controls"][38]["control_mode"] != "contact_push_intent"):
                raise ValueError("The recorded B014 contact regression changed")
            special = (f'<p class="warning">B014 contact regression: planned recontact '
                f'{recon["planned_start_seconds"]:.2f}–{recon["planned_end_seconds"]:.2f}s '
                'has no actual postsolve agent–object contact. Controls 31–37 are planned push '
                'but still use free_route; control 38 has contact_push_intent but no actual contact. '
                'First event is postsolve substep391, interval [3.91,3.92]s: main interval-start '
                'delay is 1.31s; saved post-state is 3.92s (observed endpoint delay 1.32s). '
                'The interval start is a discrete proxy, not an exact collision instant.</p>')
            if args.annotated_focus_video:
                row["annotated_preview"] = annotated_focus_video(originals/record["file"], row, out/"B014-contact-regression.mp4")
                special += '<video controls preload="metadata" width="960" src="B014-contact-regression.mp4"></video>'
        title = html.escape(row["trajectory_id"])
        roles = html.escape(", ".join(row["display_roles"]))
        cards.append(f'<section id="{title}"><h2>{title}</h2><p><b>Collection scenario:</b> {html.escape(row["collection_scenario"])}; <b>Display selection role:</b> {roles}; <b>Split:</b> {row["split"]}.</p>{special}'
            f'<video controls preload="metadata" width="384" src="{row["original_video"]}"></video>'
            f'<p>Original strip, unchanged. The labels below refer to its five saved frames.</p>'
            f'<img class="strip" src="{row["original_strip"]}">{frame_table(row)}'
            f'<img class="timeline" src="{svg}"></section>')
        rows.append(row)
    if before != {path: sha(path) for path in before}:
        raise RuntimeError("Original media changed during presentation generation")
    supplemental_html, supplemental_rows, renderer_stats = '', [], None
    if choices:
        supplemental_html, supplemental_rows, renderer_stats = supplemental_cards(choices, set(entries)&{r["trajectory_id"] for r in rows},
            manifest["entries"], source, args.source_root, out, core)
    core_link = '' if args.core_summary is None else f'<p><a href="{html.escape(relative(args.core_summary, out))}">Full-library numerical contact audit</a>. Original 35 representative examples are retained below; their selection is not a random full-library estimate.</p>'
    figures = None
    figures_html = ''
    if args.core_dir is not None and args.core_summary is not None and args.core_summary.exists():
        figures = make_figures(args.core_dir, args.core_summary.parent, out)
        figures_html = '<section><h2>Full-library measured proportions</h2><img class="timeline" src="contact-motion-independent-denominators.svg"><img class="timeline" src="actual-contact-motion-by-control-index.svg"></section>'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>R3A actual contact timelines</title><style>body{font:15px system-ui;color:#17202b;background:#e9edf2;margin:24px;max-width:1100px}section{background:white;padding:18px;margin:24px 0;border-radius:6px}h1{font-size:25px}h2{font-size:18px}p{line-height:1.5}.strip,.timeline{width:100%;height:auto}.frames{width:100%;table-layout:fixed;font-size:11px;border-collapse:collapse}.frames td{padding:8px 3px;border:1px solid #d8dee7;vertical-align:top;overflow-wrap:anywhere}.warning{background:#fff0cf;padding:12px}video{max-width:100%}.legend{background:white;padding:15px}</style><h1>R3A actual contact and controller timelines</h1><p>All videos and strips below are the original 35 viewing copies, unchanged. Collection scenario is the planned data-generation scenario; display role is a later maximum/random/least-motion selection. Neither proves continuous contact.</p><p>Five strip frames are 0.0, 1.2, 2.4, 3.6 and 4.8s. A frame at t&gt;0 follows the preceding 0.1s control: postsolve-at-t is its final 100Hz flag; contact count describes all 10 postsolve substeps in that preceding interval. Initial t=0 has no preceding postsolve. Planned phase and controller mode describe that same preceding control.</p><div class="legend"><b>Timeline:</b> planned phase: gray approach, green push, orange withdraw, purple recontact, dark gray static. Controller: gray hold, light blue free_route, dark blue contact_push_intent. Actual contact: green height is 0–10 postsolve substeps. Observed motion: gray nearstatic (&lt;0.1 image96px AND &lt;0.1°), orange obvious (&ge;2px OR &ge;5°), yellow remaining. Hover a cell for exact recorded values. Movement uses geometric area centroid; geometry and controller labels are privileged audit metadata, never model features.</div>'''
    (out/"index.html").write_text(page+core_link+figures_html+supplemental_html+'<h2>Original 35 representative examples, unchanged media</h2>'+''.join(cards)+'</html>', encoding="utf-8")
    result = dict(kind="R3A-contact-presentation.v1", records=rows, original_media_hashes=before,
        original_media_unchanged=True, physics_steps=0, physics_controls=0, model_loads=0, model_calls=0,
        manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
        original_provenance=str(provenance_path), original_provenance_sha256=sha(provenance_path),
        core_summary=None if args.core_summary is None else str(args.core_summary.resolve()),
        supplemental_selection=selection_info, supplemental_choices=choices, supplemental_records=supplemental_rows,
        renderer_stats=renderer_stats, original_35_core_cross_checked=core is not None,
        figures=None if figures is None else figures["files"],
        timestamp_alignment="frame j is state[j] at j/10; previous control j-1 and final postsolve substep 10*j-1; initial N/A")
    if figures is not None:
        result["overview_qa"] = overview_qa(choices, rows+supplemental_rows, out)
    (out/"presentation-provenance.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return dict(records=len(rows), original_files_preserved=len(before),
        output_bytes=sum(path.stat().st_size for path in out.rglob('*') if path.is_file()),
        physics_steps=0, model_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-provenance", type=Path, required=True)
    parser.add_argument("--core-summary", type=Path)
    parser.add_argument("--core-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotated-focus-video", action="store_true")
    print(json.dumps(build(parser.parse_args())))


if __name__ == "__main__":
    main()
