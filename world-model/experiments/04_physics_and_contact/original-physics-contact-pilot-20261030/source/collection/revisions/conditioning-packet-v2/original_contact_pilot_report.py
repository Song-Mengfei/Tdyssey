"""Saved-label-only motion/contact audit and real-time RGB report.

No environment construction, physics replay, learned inference or training.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from .envs.official import official_shape
from .native_rgb_env import shape_geometry
from .original_contact_pilot import LETTERS, RULES, load_attempt, write_json, file_sha

CLASSES = ('contact_moving', 'contact_almost_static', 'object_wall_only',
           'post_release_motion', 'empty_search', 'noncontact_static',
           'other_free_motion', 'invalid')


def centers(letter, states):
    states = np.asarray(states, float)
    local = np.asarray(official_shape(letter).centroid)
    c, s = np.cos(states[:, 4]), np.sin(states[:, 4])
    return states[:, 2:4] + np.column_stack((c*local[0]-s*local[1], s*local[0]+c*local[1]))


def wall_gap(letter, states):
    vertices = np.asarray(shape_geometry(letter).exterior.coords[:-1], float)
    states = np.asarray(states, float)
    c, s = np.cos(states[:, 4]), np.sin(states[:, 4])
    x = states[:, 2, None] + c[:, None]*vertices[:, 0] - s[:, None]*vertices[:, 1]
    y = states[:, 3, None] + s[:, None]*vertices[:, 0] + c[:, None]*vertices[:, 1]
    return np.min(np.column_stack((x.min(axis=1)-7, 504-x.max(axis=1),
                                  y.min(axis=1)-7, 504-y.max(axis=1))), axis=1)


def quantiles(values):
    values = np.asarray(values, float)
    good = values[np.isfinite(values)]
    result = dict(N=len(values), measurable=len(good), unknown=len(values)-len(good))
    result.update({key: float(value) for key, value in zip(
        ('P10', 'P50', 'P90', 'P95', 'max'),
        np.quantile(good, [.1, .5, .9, .95, 1.]))} if len(good) else
        {key: None for key in ('P10', 'P50', 'P90', 'P95', 'max')})
    return result


def motion_category(translation_image_px, rotation_deg):
    translation = translation_image_px >= 2
    rotation = abs(rotation_deg) >= 5
    return ('both' if translation and rotation else 'translation_only' if translation
            else 'rotation_only' if rotation else 'neither')


def _contact_kinds(row):
    return {event['kind'] for event in row.get('contacts', [])
            if event.get('phase') == 'post_solve'}


def _actual_arc(letter, point):
    from shapely.geometry import Point
    geometry = shape_geometry(letter)
    rings = [geometry.exterior, *geometry.interiors]
    p = Point(*point)
    distances = [ring.distance(p) for ring in rings]
    index = int(np.argmin(distances))
    if distances[index] > 1.:
        return None
    return dict(ring=index, arc_bin=int(rings[index].project(p)//8),
                boundary_error_world_px=float(distances[index]))


def _micro_summary(letter, info):
    rows = [row for row in info.get('substeps', []) if row.get('executed')]
    result = dict(seconds=0., AO_seconds=0., OW_seconds=0., AW_seconds=0.,
                  contact_moving_seconds=0., contact_static_seconds=0.,
                  wall_only_seconds=0., noncontact_seconds=0.,
                  boundary_seconds=0., path_world_px=0., angular_path_deg=0.,
                  max_penetration=[0., 0., 0.], arc_hits=[], observed_substeps=len(rows))
    if not rows:
        return result
    pre = np.asarray([row.get('pre_state', row.get('integrator_pre_state')) for row in rows], float)
    post = np.asarray([row['post_state'] for row in rows], float)
    dt = np.asarray([row['dt'] for row in rows], float)
    distance = np.linalg.norm(centers(letter, post)-centers(letter, pre), axis=1)
    angle = np.abs(post[:, 4]-pre[:, 4])*180/math.pi
    gaps = np.minimum(wall_gap(letter, pre), wall_gap(letter, post))
    result['path_world_px'] = float(distance.sum())
    result['angular_path_deg'] = float(angle.sum())
    for index, (row, seconds) in enumerate(zip(rows, dt)):
        kinds = _contact_kinds(row)
        ao, ow, aw = ('agent_object' in kinds, 'object_wall' in kinds, 'agent_wall' in kinds)
        moving = distance[index]*96/512 >= .1*seconds/.1 or angle[index] >= .1*seconds/.1
        result['seconds'] += float(seconds)
        result['AO_seconds'] += float(seconds)*ao
        result['OW_seconds'] += float(seconds)*ow
        result['AW_seconds'] += float(seconds)*aw
        result['contact_moving_seconds'] += float(seconds)*(ao and moving)
        result['contact_static_seconds'] += float(seconds)*(ao and not moving)
        result['wall_only_seconds'] += float(seconds)*(ow and not ao)
        result['noncontact_seconds'] += float(seconds)*(not ao and not ow)
        result['boundary_seconds'] += float(seconds)*(gaps[index] <= 24.)
        penetration = row.get('penetrations', [float('nan')]*3)
        for column, value in enumerate(penetration):
            if np.isfinite(value):
                result['max_penetration'][column] = max(result['max_penetration'][column], float(value))
        # One observed arc contributes dt once per microstep, not once per
        # contact point or duplicated rectangle arbiter.
        observed = set()
        for event in row.get('contacts', []):
            if event.get('phase') != 'post_solve' or event.get('kind') != 'agent_object':
                continue
            object_first = event['shapes_order'][0] == 'object'
            for point in event.get('points', []):
                world = point.get('object_world') or point.get('point_a' if object_first else 'point_b')
                if world is None:
                    continue
                local = point.get('object_local')
                if local is None:
                    delta = np.asarray(world)-post[index, 2:4]
                    c, s = math.cos(post[index, 4]), math.sin(post[index, 4])
                    local = [c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1]]
                arc = _actual_arc(letter, local)
                if arc is not None:
                    observed.add((arc['ring'], arc['arc_bin']))
                else:
                    observed.add((-1, -1))
        if ao and not observed:
            observed.add((-1, -1))
        result['arc_hits'].extend(dict(ring=ring, arc_bin=arc, seconds=float(seconds))
                                  for ring, arc in observed)
    return result


def trajectory_rows(metadata, data):
    slot = metadata['slot']
    letter = slot['letter']
    states = data['states']
    n = len(data['requested_actions'])
    if len(states) != n+1 or data['rgb'].shape != (n+1, 3, 96, 96):
        raise ValueError('Saved state/action/RGB timeline lengths differ')
    center = centers(letter, states)
    angles = states[:, 4]*180/math.pi  # Engine angles are continuous, not modulo90/180.
    controls, arcs = [], []
    for index, (info, decision) in enumerate(zip(data['telemetry'], data['decisions'])):
        micro = _micro_summary(letter, info)
        translation = float(np.linalg.norm(center[index+1]-center[index]))
        rotation = float(angles[index+1]-angles[index])
        moving = translation*96/512 >= .1 or abs(rotation) >= .1
        ao, ow = micro['AO_seconds'] > 0, micro['OW_seconds'] > 0
        pusher_move = float(np.linalg.norm(states[index+1, :2]-states[index, :2]))*96/512
        valid = bool(data['control_valid'][index])
        recent_ao = any(row['AO_seconds'] > 0 for row in controls[-4:])
        category = ('invalid' if not valid else 'contact_moving' if ao and moving
                    else 'contact_almost_static' if ao else 'object_wall_only' if ow
                    else 'post_release_motion' if recent_ao and moving
                    else 'empty_search' if pusher_move >= .1
                    else 'other_free_motion' if moving else 'noncontact_static')
        observed_stage = ('invalid' if not valid else
                          'first_contact' if ao and (not controls or controls[-1]['AO_seconds'] == 0) else
                          'sustained_contact' if ao else
                          'post_release' if recent_ao else
                          'object_wall_without_pusher' if ow else
                          'free_search' if pusher_move >= .1 else 'free_static')
        intervention = info.get('intervention_time_ratio', info.get('constraint_intervention_time_ratio'))
        if intervention is None:
            executed = [row for row in info.get('substeps', []) if row.get('executed')]
            constrained_seconds = sum(float(row.get('dt', 0.)) for row in executed
                                      if row.get('intervened') or row.get('velocity_constrained')
                                      or np.linalg.norm(np.asarray(row.get('unconstrained_velocity', [0., 0.]))-
                                                        np.asarray(row.get('executed_velocity', [0., 0.]))) > 1e-8)
            intervention = constrained_seconds/micro['seconds'] if micro['seconds'] else None
        row = dict(trajectory_id=slot['slot_id'], family_id=slot['family_id'], letter=letter,
                   split=slot['split'], phase_group=slot['phase'], control=index,
                   planned_phase=decision['phase'], valid=valid, category=category,
                   observed_contact_stage=observed_stage,
                   translation_world_px=translation, translation_image_px=translation*96/512,
                   signed_rotation_deg=rotation, rotation_deg=abs(rotation),
                   motion_category=motion_category(translation*96/512, rotation),
                   boundary=micro['boundary_seconds'] > 0,
                   intervention_time_ratio=intervention, **{k: v for k, v in micro.items() if k != 'arc_hits'})
        controls.append(row)
        arcs.extend(dict(trajectory_id=slot['slot_id'], letter=letter, control=index,
                         planned_arc=decision['selected_arc_id'], **hit) for hit in micro['arc_hits'])
    windows = []
    image_hashes = [hashlib.sha256(frame.tobytes()).digest() for frame in data['rgb']]
    for horizon in (1, 4):
        for start in range(n-horizon+1):
            part = controls[start:start+horizon]
            translation = float(np.linalg.norm(center[start+horizon]-center[start]))
            rotation = float(angles[start+horizon]-angles[start])
            valid = all(row['valid'] for row in part) and abs(sum(row['seconds'] for row in part)-horizon*.1) < 1e-7
            available_history = start >= 3
            unique = None
            if available_history:
                payload = b''.join(image_hashes[start-3:start+1])+np.asarray(data['requested_actions'][start:start+horizon], np.float64).tobytes()
                unique = hashlib.sha256(payload).hexdigest()
            candidate = (slot['phase'] in ('formal', 'supplement') and metadata['complete'] and valid and available_history)
            windows.append(dict(trajectory_id=slot['slot_id'], family_id=slot['family_id'],
                letter=letter, split=slot['split'], phase_group=slot['phase'], start_control=start,
                horizon=horizon, seconds=sum(row['seconds'] for row in part), valid=valid,
                candidate=candidate, history4_available=available_history, input_sha256=unique,
                planned_phase=part[0]['planned_phase'] if len({r['planned_phase'] for r in part}) == 1 else 'phase_boundary',
                observed_contact_stage=part[0]['observed_contact_stage'] if len({r['observed_contact_stage'] for r in part}) == 1 else 'contact_stage_boundary',
                boundary=any(row['boundary'] for row in part),
                AO=any(row['AO_seconds'] > 0 for row in part),
                translation_world_px=translation, translation_image_px=translation*96/512,
                rotation_deg=abs(rotation), signed_rotation_deg=rotation,
                path_world_px=sum(row['path_world_px'] for row in part),
                path_image_px=sum(row['path_world_px'] for row in part)*96/512,
                angular_path_deg=sum(row['angular_path_deg'] for row in part),
                motion_category=motion_category(translation*96/512, rotation)))
    return controls, windows, arcs


def _bin_counts(values, edges):
    edges = [float('inf') if edge is None else float(edge) for edge in edges]
    counts, _ = np.histogram(np.asarray(values, float), bins=edges)
    return counts.tolist()


def group_motion(rows, **labels):
    result = dict(**labels, N=len(rows), independent_trajectories=len({r['trajectory_id'] for r in rows}),
                  independent_families=len({r['family_id'] for r in rows}),
                  exact_unique_model_inputs=len({r['input_sha256'] for r in rows if r['input_sha256'] is not None}),
                  motion_categories=dict(Counter(r['motion_category'] for r in rows)))
    for metric in ('translation_world_px', 'translation_image_px', 'rotation_deg', 'path_world_px', 'path_image_px', 'angular_path_deg'):
        result[metric] = quantiles([row[metric] for row in rows])
    result['translation_bins_image_px'] = _bin_counts([r['translation_image_px'] for r in rows], RULES['translation_bins_image_px'])
    result['rotation_bins_deg'] = _bin_counts([r['rotation_deg'] for r in rows], RULES['rotation_bins_deg'])
    return result


def contact_table(rows, *, denominator, candidate_ids=None):
    if candidate_ids is not None:
        rows = [row for row in rows if row['trajectory_id'] in candidate_ids]
    counts = Counter(row['category'] for row in rows)
    all_seconds = sum(row['seconds'] for row in rows)
    valid_seconds = sum(row['seconds'] for row in rows if row['valid'])
    result = dict(denominator=denominator, actual_attempted_controls=len(rows),
                  actual_physical_seconds=all_seconds, valid_physical_seconds=valid_seconds,
                  category_control_counts={key: counts[key] for key in CLASSES},
                  category_control_fractions={key: counts[key]/len(rows) if rows else None for key in CLASSES})
    for key in ('AO_seconds', 'OW_seconds', 'AW_seconds', 'contact_moving_seconds',
                'contact_static_seconds', 'wall_only_seconds', 'noncontact_seconds', 'boundary_seconds'):
        value = sum(row[key] for row in rows)
        valid_value = sum(row[key] for row in rows if row['valid'])
        result[key] = value
        result[key+'_fraction_all_actual_time'] = value/all_seconds if all_seconds else None
        result[key+'_fraction_valid_time'] = valid_value/valid_seconds if valid_seconds else None
    result['invalid_controls'] = counts['invalid']
    result['contact_moving_control_fraction_all_attempts'] = counts['contact_moving']/len(rows) if rows else None
    result['useful_static_negative_control_fraction_all_attempts'] = ((counts['contact_almost_static']+counts['noncontact_static'])/len(rows) if rows else None)
    result['useful_static_negative_definition'] = 'actual contact-almost-static plus noncontact-static; empty search kept separate'
    result['max_penetrations_world_px_OW_AO_AW'] = [max((row['max_penetration'][i] for row in rows), default=None) for i in range(3)]
    result['intervention_time_ratio'] = (sum(row['seconds']*row['intervention_time_ratio'] for row in rows
                                           if row['intervention_time_ratio'] is not None)/all_seconds if all_seconds else None)
    return result


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _motion_plots(destination, windows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for letter in LETTERS:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for ri, horizon in enumerate((1, 4)):
            rows = [r for r in windows if r['letter'] == letter and r['candidate'] and r['horizon'] == horizon]
            for ci, (metric, edges, labels) in enumerate((
                ('translation_image_px', RULES['translation_bins_image_px'], ['<0.5', '0.5-2', '2-5', '>=5']),
                ('rotation_deg', RULES['rotation_bins_deg'], ['<1', '1-5', '5-15', '>=15']))):
                counts = _bin_counts([r[metric] for r in rows], edges)
                axes[ri, ci].bar(labels, counts, color='#477b91')
                axes[ri, ci].set_title(f'{letter} H{horizon}={horizon/10:.1f}s {metric}; N={len(rows)}')
                axes[ri, ci].set_ylabel('All candidate windows (not only moving)')
                axes[ri, ci].tick_params(axis='x', labelsize=9)
        fig.savefig(destination/f'motion-{letter}.png', dpi=120)
        plt.close(fig)


def _arc_plots(destination, arcs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from .native_rgb_data import contour_segments
    summaries = []
    for letter in LETTERS:
        rows = [row for row in arcs if row['letter'] == letter]
        hits = Counter()
        for row in rows:
            hits[(row['ring'], row['arc_bin'])] += row['seconds']
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        shape = shape_geometry(letter)
        rings = [shape.exterior, *shape.interiors]
        for ring_index, ring in enumerate(rings):
            pieces, values = [], []
            for lo in np.arange(0, ring.length, 8.):
                p, q = ring.interpolate(lo), ring.interpolate(min(lo+8., ring.length))
                pieces.append([(p.x, p.y), (q.x, q.y)])
                values.append(hits[(ring_index, int(lo//8))])
            collection = LineCollection(pieces, array=np.asarray(values), cmap='viridis', linewidths=5)
            if ring_index:
                collection.set_linestyle('dashed')
            ax.add_collection(collection)
        ax.autoscale()
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'{letter}: actual contact seconds by outer8-world-px arc\nDashed closed holes: not exterior targets')
        fig.savefig(destination/f'arcs-{letter}.png', dpi=120)
        plt.close(fig)
        exterior = [segment for segment in contour_segments(letter, 8) if segment['ring'] == 0]
        summaries.append(dict(letter=letter, exterior_arc_count=len(exterior),
            observed_exterior_arcs=sum(value > 0 for (ring, _), value in hits.items() if ring == 0),
            observed_closed_hole_arcs=sum(value > 0 for (ring, _), value in hits.items() if ring > 0),
            unknown_projection_seconds=hits[(-1, -1)],
            closed_holes='exterior-inaccessible; never requested',
            arc_contact_seconds=[dict(ring=ring, arc_bin=arc, seconds=value) for (ring, arc), value in sorted(hits.items())]))
    return summaries


def _video(destination, metadata, data, controls):
    """Actual saved RGB, nearest display only;10fps at10Hz, no physics replay."""
    import cv2
    import imageio_ffmpeg
    destination.parent.mkdir(parents=True, exist_ok=True)
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    size = (640, 592)
    process = subprocess.Popen([executable, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{size[0]}x{size[1]}',
        '-r', '10', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '24',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)], stdin=subprocess.PIPE)
    slot = metadata['slot']
    positions = centers(slot['letter'], data['states'])
    for index, image in enumerate(data['rgb']):
        canvas = np.full((size[1], size[0], 3), 248, np.uint8)
        picture = cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_RGB2BGR)
        canvas[104:584, 80:560] = cv2.resize(picture, (480, 480), interpolation=cv2.INTER_NEAREST)
        if index:
            h1 = controls[index-1]
            start = max(0, index-4)
            h4 = np.linalg.norm(positions[index]-positions[start])*96/512
            angle4 = abs(data['states'][index, 4]-data['states'][start, 4])*180/math.pi
            lines = [f'{slot["letter"]} | {slot["slot_id"]}',
                     f't={sum(row["seconds"] for row in controls[:index]):.2f}s | H1 {h1["translation_image_px"]:.2f}px / {h1["rotation_deg"]:.2f}deg | {h1["category"]}',
                     f'H{index-start} {h4:.2f}px / {angle4:.2f}deg | phase={h1["planned_phase"]} | valid={h1["valid"]}']
        else:
            lines = [f'{slot["letter"]} | {slot["slot_id"]}',
                     't=0.00s | independent actual initial construction',
                     'Saved actual RGB,10fps/10Hz; no interpolation, replay or speed-up']
        for row, line in enumerate(lines):
            cv2.putText(canvas, line, (10, 24+28*row), cv2.FONT_HERSHEY_SIMPLEX,
                        .43 if row == 0 else .48, (35, 45, 55), 1, cv2.LINE_AA)
        process.stdin.write(canvas.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoder failed')
    capture = cv2.VideoCapture(str(destination))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    ok, _ = capture.read()
    capture.release()
    if not ok or frames != len(data['rgb']) or abs(fps-10) > 1e-6:
        raise RuntimeError('Saved real-time video decode check failed')
    return dict(path=str(destination), sha256=file_sha(destination), bytes=destination.stat().st_size,
                frames=frames, fps=fps, simulated_seconds=sum(row['seconds'] for row in controls),
                video_seconds=frames/fps, final_frame_hold_seconds=.1,
                statement='Each saved actual RGB at10fps/10Hz. Last observation has one frame hold; no interpolation or accelerated motion.')


def build_report(output, *, videos=False):
    output = Path(output).resolve()
    report = output/'report'
    report.mkdir(parents=True, exist_ok=True)
    controls, windows, arcs, attempts = [], [], [], []
    by_id = {}
    for path in sorted((output/'trajectories').glob('*/metadata.json')):
        metadata = json.loads(path.read_text(encoding='utf-8'))
        if metadata['raw'] is None:
            attempts.append(dict(trajectory_id=metadata['slot']['slot_id'], letter=metadata['slot']['letter'],
                                 complete=False, raw=None, failure=metadata['failure']))
            continue
        raw_path = path.parent/metadata['raw']['file']
        if file_sha(raw_path) != metadata['raw']['sha256']:
            raise ValueError('Saved raw file changed: '+str(raw_path))
        data = load_attempt(raw_path)
        rows, window_rows, arc_rows = trajectory_rows(metadata, data)
        controls.extend(rows)
        windows.extend(window_rows)
        arcs.extend(arc_rows)
        slot = metadata['slot']
        entry = dict(trajectory_id=slot['slot_id'], family_id=slot['family_id'],
                     letter=slot['letter'], split=slot['split'], phase=slot['phase'],
                     initial_space=slot['initial_space'], scenario=slot['scenario'],
                     complete=metadata['complete'], actual_controls=len(rows),
                     valid_controls=sum(r['valid'] for r in rows),
                     max_translation_image_px=max((r['translation_image_px'] for r in rows), default=0),
                     max_rotation_deg=max((r['rotation_deg'] for r in rows), default=0),
                     boundary_seconds=sum(r['boundary_seconds'] for r in rows),
                     contact_moving_seconds=sum(r['contact_moving_seconds'] for r in rows),
                     raw=str(raw_path), raw_sha256=metadata['raw']['sha256'],
                     metadata=str(path), failure=metadata['failure'])
        attempts.append(entry)
        by_id[slot['slot_id']] = (path, rows)
    candidate_ids = {row['trajectory_id'] for row in windows if row['candidate']}
    grouped = []
    # The full attempted-window table precedes valid/candidate filtering.
    for cohort in ('all_attempted_windows', 'valid_physical_windows', 'candidate_model_windows'):
        base = [row for row in windows if cohort == 'all_attempted_windows'
                or cohort == 'valid_physical_windows' and row['valid']
                or cohort == 'candidate_model_windows' and row['candidate']]
        for letter in ('all', *LETTERS):
            for horizon in (1, 4):
                rows = [row for row in base if row['horizon'] == horizon and (letter == 'all' or row['letter'] == letter)]
                grouped.append(group_motion(rows, cohort=cohort, letter=letter, horizon=horizon))
                if letter != 'all':
                    for phase in sorted({row['planned_phase'] for row in rows}):
                        grouped.append(group_motion([row for row in rows if row['planned_phase'] == phase],
                                                    cohort=cohort, letter=letter, horizon=horizon, planned_phase=phase))
                    for boundary in (False, True):
                        grouped.append(group_motion([row for row in rows if row['boundary'] == boundary],
                                                    cohort=cohort, letter=letter, horizon=horizon, boundary=boundary))
                    for contact_stage in sorted({row['observed_contact_stage'] for row in rows}):
                        grouped.append(group_motion([row for row in rows if row['observed_contact_stage'] == contact_stage],
                                                    cohort=cohort, letter=letter, horizon=horizon,
                                                    observed_contact_stage=contact_stage))
    contacts = [contact_table(controls, denominator='all_actual_attempts'),
                contact_table(controls, denominator='complete_candidate_trajectories', candidate_ids=candidate_ids)]
    for phase_group in sorted({row['phase_group'] for row in controls}):
        contacts.append(dict(phase_group=phase_group, **contact_table(
            [row for row in controls if row['phase_group'] == phase_group], denominator='all_actual_attempts_in_phase')))
        for letter in LETTERS:
            contacts.append(dict(phase_group=phase_group, letter=letter, **contact_table(
                [row for row in controls if row['phase_group'] == phase_group and row['letter'] == letter],
                denominator='all_actual_attempts_in_phase')))
            for horizon in (1, 4):
                for cohort in ('all_attempted_windows', 'valid_physical_windows', 'candidate_model_windows'):
                    rows = [row for row in windows if row['phase_group'] == phase_group and row['letter'] == letter
                            and row['horizon'] == horizon and (cohort == 'all_attempted_windows'
                            or cohort == 'valid_physical_windows' and row['valid']
                            or cohort == 'candidate_model_windows' and row['candidate'])]
                    grouped.append(group_motion(rows, cohort=cohort, phase_group=phase_group, letter=letter, horizon=horizon))
    for letter in LETTERS:
        letter_rows = [row for row in controls if row['letter'] == letter]
        contacts.append(dict(letter=letter, **contact_table(letter_rows, denominator='all_actual_attempts')))
        contacts.append(dict(letter=letter, **contact_table(letter_rows, denominator='complete_candidate_trajectories', candidate_ids=candidate_ids)))
    _motion_plots(report, windows)
    arc_summary = _arc_plots(report, arcs)
    video_rows = []
    if videos:
        selection = []
        for letter in LETTERS:
            choices = [row for row in attempts if row['letter'] == letter and row.get('raw')]
            if not choices:
                continue
            complete = [row for row in choices if row['complete']]
            if complete:
                translation = max(complete, key=lambda row: row['max_translation_image_px'])
                rotation = max(complete, key=lambda row: row['max_rotation_deg'])
                boundary = max(complete, key=lambda row: row['boundary_seconds'])
                weak = min(complete, key=lambda row: row['max_translation_image_px'])
                random = sorted(complete, key=lambda row: hashlib.sha256(('fixed-video-seed-2026103019:'+row['trajectory_id']).encode()).digest())[0]
                selection.extend((tag, row) for tag, row in [('clear_translation', translation), ('clear_rotation', rotation),
                    ('boundary', boundary), ('weak_motion', weak), ('fixed_seed_random_complete', random)])
            rejected = [row for row in choices if not row['complete']]
            if rejected:
                selection.append(('rejected_first', rejected[0]))
        generated = {}
        for tag, entry in selection:
            tid = entry['trajectory_id']
            if tid not in generated:
                path, trajectory_controls = by_id[tid]
                metadata = json.loads(path.read_text(encoding='utf-8'))
                data = load_attempt(path.parent/'raw.npz')
                generated[tid] = _video(report/'videos'/(tid+'.mp4'), metadata, data, trajectory_controls)
            video_rows.append(dict(selection=tag, trajectory_id=tid, letter=entry['letter'], **generated[tid]))
        if sum(row['bytes'] for row in generated.values()) > RULES['video_cap_bytes']:
            raise RuntimeError('Actual report videos exceeded64MiB; preserve files and stop')
    all_candidate = [row for row in windows if row['candidate']]
    weak_letters = []
    for letter in LETTERS:
        h1 = [row for row in all_candidate if row['letter'] == letter and row['horizon'] == 1]
        weak = sum(row['translation_image_px'] < .5 for row in h1)/len(h1) if h1 else None
        rotation_trajectories = {row['trajectory_id'] for row in h1 if row['rotation_deg'] >= 5.}
        weak_letters.append(dict(letter=letter, H1_N=len(h1), translation_under_half_image_px_fraction=weak,
                                 H1_obvious_rotation_trajectories=len(rotation_trajectories),
                                 supplement_trigger=weak is None or weak > .5 or not rotation_trajectories))
    summary = dict(kind='original-contact-pilot-saved-label-report.v1',
                   model_calls=0, optimizer_updates=0, physics_replay_steps=0,
                   attempts=attempts, attempt_count=len(attempts),
                   complete_trajectories=sum(row['complete'] for row in attempts),
                   actual_controls=len(controls), valid_controls=sum(row['valid'] for row in controls),
                   actual_microsteps=sum(row['observed_substeps'] for row in controls),
                   actual_physics_seconds=sum(row['seconds'] for row in controls),
                   motion_groups=grouped, contact_denominators=contacts,
                   actual_arc_coverage=arc_summary, videos=video_rows,
                   weak_motion_audit=weak_letters, rules=RULES,
                   metric_contract=dict(world_to_image='96/512', H1_seconds=.1, H4_seconds=.4,
                       positions='exact polygon area centroid from saved continuous state, not body origin or RGB estimate',
                       net_rotation='absolute difference of continuous engine angles, never symmetry reduced',
                       path='sum of actual microstep centroid distances / absolute angle increments; not endpoint chord',
                       candidate='complete valid formal/supplement trajectory, four actual history frames available',
                       unknowns='retained, not passing; early/invalid raw preserved; no filled frames',
                       micro_motion_floor='control0.1imagepx or0.1degree scaled linearly with actual dt; numerical audit threshold, not visual significance',
                       contact='post-solve actual pair presence; arbiter/point counts never replace physical time'))
    write_json(report/'summary.json', summary)
    csv_write(report/'control-metrics.csv', controls)
    csv_write(report/'window-metrics.csv', windows)
    csv_write(report/'motion-summary.csv', grouped)
    csv_write(report/'contact-denominators.csv', contacts)
    csv_write(report/'actual-contact-arcs.csv', arcs)
    csv_write(report/'trajectory-index.csv', attempts)
    def table(rows, columns):
        return '<table><tr>'+''.join('<th>'+html.escape(key)+'</th>' for key in columns)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(row.get(key)))+'</td>' for key in columns)+'</tr>' for row in rows)+'</table>'
    panels = []
    for letter in LETTERS:
        rows = [row for row in grouped if row['cohort'] == 'candidate_model_windows' and row['letter'] == letter
                and 'planned_phase' not in row and 'boundary' not in row and 'phase_group' not in row
                and 'observed_contact_stage' not in row]
        panels.append(f'<section><h2>{letter}</h2><img src="motion-{letter}.png"><img src="arcs-{letter}.png">'+
                      table([dict(H=r['horizon'], N=r['N'], trajectories=r['independent_trajectories'],
                           translation_P50=r['translation_image_px']['P50'], translation_P90=r['translation_image_px']['P90'],
                           rotation_P50=r['rotation_deg']['P50'], rotation_P90=r['rotation_deg']['P90'],
                           categories=r['motion_categories']) for r in rows],
                           ['H', 'N', 'trajectories', 'translation_P50', 'translation_P90', 'rotation_P50', 'rotation_P90', 'categories'])+
                      ''.join(f'<h3>{html.escape(v["selection"])}: {html.escape(v["trajectory_id"])}</h3>'
                              f'<video controls preload="metadata" src="videos/{html.escape(Path(v["path"]).name)}"></video>'
                              for v in video_rows if v['letter'] == letter)+'</section>')
    headline = dict(attempts=len(attempts), complete=sum(row['complete'] for row in attempts),
                    actual_controls=len(controls), valid_controls=sum(row['valid'] for row in controls),
                    physics_seconds=summary['actual_physics_seconds'], optimizer_updates=0)
    content = '<!doctype html><meta charset="utf-8"><title>Original contact pilot: saved evidence</title><style>'\
        'body{font:16px system-ui;max-width:1400px;margin:32px auto;color:#20313d}section{border-top:1px solid #ccc;margin-top:32px}'\
        'table{border-collapse:collapse;margin:20px 0;font-size:13px}td,th{border:1px solid #ccd;padding:6px;max-width:260px;overflow-wrap:anywhere}'\
        'img{max-width:49%;vertical-align:top}video{width:640px;max-width:100%}p{line-height:1.6}</style>'\
        '<h1>Original kinematic / damping0 contact-pilot data audit</h1>'\
        '<p>Candidate data only. No training, model inference, future search or physics replay in this report. '\
        'Raw labels/images remain intact. Calibration, primary and supplement attempts are not silently mixed. '\
        'Contact proportions include every actual attempted control/time; candidate filtering is a separate denominator.</p>'\
        + table([headline], list(headline)) + '<p>World-to-image scale96/512; H1=0.1s, H4=0.4s. '\
        'Net centroid/continuous-angle change and cumulative microstep path are separate. '
    content += 'Videos are actual saved frames at10fps, not accelerated; final observation has a0.1s hold. '
    content += 'Fixed-seed complete videos, weak/rejected cases and selected motion exemplars are labelled separately. '
    content += 'Dashed hole contours are not exterior sampling targets. More details and all denominators: '
    content += '<a href="summary.json">summary.json</a>, <a href="window-metrics.csv">windows</a>, '
    content += '<a href="contact-denominators.csv">contacts</a>, <a href="trajectory-index.csv">all trajectories</a>.</p>'
    content += '<h2>All attempts and candidate-trajectory contact denominators</h2>'+table(contacts,
        ['phase_group', 'letter', 'denominator', 'actual_attempted_controls', 'actual_physical_seconds', 'valid_physical_seconds',
         'category_control_counts', 'AO_seconds_fraction_all_actual_time',
         'contact_moving_seconds_fraction_all_actual_time', 'boundary_seconds_fraction_all_actual_time',
         'invalid_controls', 'max_penetrations_world_px_OW_AO_AW', 'intervention_time_ratio'])
    content += '<h2>Weak-motion audit; finite supplement trigger is not a pass label</h2>'+table(weak_letters,
        ['letter', 'H1_N', 'translation_under_half_image_px_fraction', 'H1_obvious_rotation_trajectories', 'supplement_trigger'])
    content += ''.join(panels)
    (report/'index.html').write_text(content, encoding='utf-8')
    text = '# Original contact pilot: saved data report\n\n'+json.dumps(headline, indent=2)+'\n\n'
    text += '[Interactive report](report/index.html), [all saved metrics](report/summary.json).\n\n'
    text += 'Candidate data only, training0. All attempts, valid physical time and candidate windows have separate denominators. '
    text += 'Physics修复/清理交付由负责人根报告链接；本文件只读真实保存图像、状态与接触，不重放物理。\n'
    (output/'REPORT.md').write_text(text, encoding='utf-8')
    return headline
