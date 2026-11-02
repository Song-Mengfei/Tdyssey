"""Saved-label-only motion/contact audit and real-time RGB report.

No environment construction, physics replay, learned inference or training.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from .envs.official import official_shape
from .native_rgb_env import shape_geometry
from .original_contact_pilot import LETTERS, RULES, load_attempt, file_sha

CLASSES = ('contact_moving', 'contact_almost_static', 'object_wall_only',
           'post_release_motion', 'empty_search', 'noncontact_static',
           'other_free_motion', 'invalid')


def centers(letter, states):
    states = np.asarray(states, float)
    local = np.asarray(official_shape(letter).centroid)
    c, s = np.cos(states[:, 4]), np.sin(states[:, 4])
    return states[:, 2:4] + np.column_stack((c*local[0]-s*local[1], s*local[0]+c*local[1]))


def wall_gap(letter, states):
    vertices = np.asarray(_local_geometry(letter).exterior.coords[:-1], float)
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


def conditioning_packet_sha256(image_hashes, requested_actions, start, horizon):
    """Identity of a raw rollout-conditioning packet, not network input tensors.

    Four actual history RGBs, their three connecting requested actions, and the
    next H requested actions form the external packet. Later autoregressive
    predictions/model inputs are not part of this saved-data identity. Actions
    remain original world-coordinate float64 targets, before model normalization.
    """
    if start < 3 or horizon not in (1, 4):
        raise ValueError('A packet requires four actual history frames and H1/H4')
    actions = np.asarray(requested_actions, dtype='<f8')
    if start+horizon > len(actions) or start >= len(image_hashes):
        raise ValueError('Conditioning packet lies outside saved action/history timeline')
    history = image_hashes[start-3:start+1]
    if len(history) != 4:
        raise ValueError('Four actual history images required')
    header = b'original-contact-pilot-raw-conditioning-packet.v2\0'+bytes([horizon])
    payload = header+b''.join(history)+actions[start-3:start+horizon].tobytes()
    return hashlib.sha256(payload).hexdigest()


def _contact_kinds(row):
    return {event['kind'] for event in row.get('contacts', [])
            if event.get('phase') == 'post_solve'}


@lru_cache(maxsize=6)
def _local_geometry(letter):
    if letter not in LETTERS:
        raise ValueError('Only the six released collection letters are exposed')
    return shape_geometry(letter)


@lru_cache(maxsize=6)
def _local_rings(letter):
    geometry = _local_geometry(letter)
    return (geometry.exterior, *geometry.interiors)


def _actual_arc(letter, point):
    from shapely.geometry import Point
    rings = _local_rings(letter)
    p = Point(*point)
    distances = [ring.distance(p) for ring in rings]
    index = int(np.argmin(distances))
    if distances[index] > 1.:
        return None
    return dict(ring=index, arc_bin=int(rings[index].project(p)//8),
                boundary_error_world_px=float(distances[index]))


def aggregate_actual_arc_hits(hits):
    """Compact per-control arc time; full microstep evidence remains in raw.

    The same arc/microstep is counted once even when multiple contact points or
    component arbiters report it. Different arcs remain nonexclusive in time.
    """
    grouped = defaultdict(dict)
    for hit in hits:
        key = (int(hit['ring']), int(hit['arc_bin']))
        micro = int(hit['microstep_index'])
        seconds = float(hit['seconds'])
        previous = grouped[key].get(micro)
        if previous is not None and previous != seconds:
            raise ValueError('Duplicate arc/microstep has inconsistent actual dt')
        grouped[key][micro] = seconds
    result = []
    for (ring, arc), samples in sorted(grouped.items()):
        indices = sorted(samples)
        result.append(dict(ring=ring, arc_bin=arc,
            seconds=sum(samples[index] for index in indices),
            microstep_count=len(indices), microstep_first=indices[0], microstep_last=indices[-1],
            microstep_indices_sha256=hashlib.sha256(np.asarray(indices, dtype='<u4').tobytes()).hexdigest()))
    return result


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
        result['arc_hits'].extend(dict(ring=ring, arc_bin=arc, seconds=float(seconds), microstep_index=index)
                                  for ring, arc in observed)
    result['arc_hits'] = aggregate_actual_arc_hits(result['arc_hits'])
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
                   sampler_mode=decision.get('mode'), selected_arc_id=decision.get('selected_arc_id'),
                   target_refresh_reason=decision.get('target_refresh_reason'),
                   target_refresh_count=decision.get('target_refresh_count'),
                   observed_contact_stage=observed_stage,
                   translation_world_px=translation, translation_image_px=translation*96/512,
                   signed_rotation_deg=rotation, rotation_deg=abs(rotation),
                   motion_category=motion_category(translation*96/512, rotation),
                   boundary=micro['boundary_seconds'] > 0,
                   intervention_time_ratio=intervention, **{k: v for k, v in micro.items() if k != 'arc_hits'})
        controls.append(row)
        arcs.extend(dict(trajectory_id=slot['slot_id'], family_id=slot['family_id'],
                         phase_group=slot['phase'], split=slot['split'],
                         trajectory_complete=bool(metadata['complete']), control_valid=valid,
                         letter=letter, control=index,
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
                unique = conditioning_packet_sha256(image_hashes, data['requested_actions'], start, horizon)
            candidate = (slot['phase'] in ('formal', 'supplement') and metadata['complete'] and valid and available_history)
            windows.append(dict(trajectory_id=slot['slot_id'], family_id=slot['family_id'],
                letter=letter, split=slot['split'], phase_group=slot['phase'], start_control=start,
                horizon=horizon, seconds=sum(row['seconds'] for row in part), valid=valid,
                candidate=candidate, history4_available=available_history, conditioning_packet_sha256=unique,
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
                  exact_unique_conditioning_packets=len({r['conditioning_packet_sha256'] for r in rows
                                                         if r['conditioning_packet_sha256'] is not None}),
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


def _write_output(path, payload, resource_reservation=None):
    if resource_reservation is not None:
        resource_reservation.write_bytes(path, payload)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(payload)


def _figure_output(figure, path, resource_reservation=None):
    memory = io.BytesIO()
    figure.savefig(memory, format='png', dpi=120)
    _write_output(path, memory.getvalue(), resource_reservation)


def csv_write(path, rows, resource_reservation=None):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    memory = io.StringIO(newline='')
    writer = csv.DictWriter(memory, fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    _write_output(path, memory.getvalue().encode('utf-8'), resource_reservation)


def _motion_plots(destination, windows, resource_reservation=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for cohort, title in (('formal-candidates', 'Formal primary candidate packets only'),
                          ('all-valid-stages', 'All stages valid physical windows (includes calibration)')):
        for letter in LETTERS:
            fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
            fig.suptitle(title)
            for ri, horizon in enumerate((1, 4)):
                rows = [r for r in windows if r['letter'] == letter and r['horizon'] == horizon
                        and (r['candidate'] and r['phase_group'] == 'formal' if cohort == 'formal-candidates' else r['valid'])]
                for ci, (metric, edges, labels) in enumerate((
                    ('translation_image_px', RULES['translation_bins_image_px'], ['<0.5', '0.5-2', '2-5', '>=5']),
                    ('rotation_deg', RULES['rotation_bins_deg'], ['<1', '1-5', '5-15', '>=15']))):
                    counts = _bin_counts([r[metric] for r in rows], edges)
                    axes[ri, ci].bar(labels, counts, color='#477b91')
                    axes[ri, ci].set_title(f'{letter} H{horizon}={horizon/10:.1f}s {metric}; N={len(rows)}')
                    axes[ri, ci].set_ylabel('Every window in stated cohort, not only moving')
                    axes[ri, ci].tick_params(axis='x', labelsize=9)
            _figure_output(fig, destination/f'motion-{cohort}-{letter}.png', resource_reservation)
            plt.close(fig)


def arc_cohort_rows(rows, cohort):
    """Keep actual primary candidate coverage distinct from calibration/extras."""
    if cohort == 'all_stages_all_attempts':
        return list(rows)
    if cohort != 'formal_primary_candidates':
        raise ValueError('Unknown arc coverage cohort')
    return [row for row in rows if row.get('phase_group', row.get('phase')) == 'formal'
            and row['split'] in ('train_candidate', 'seen_validation')
            and row.get('trajectory_complete', row.get('complete', False))
            and row.get('control_valid', row.get('valid', True))]


def _arc_plots(destination, arcs, attempts, controls, resource_reservation=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from .native_rgb_data import contour_segments
    summaries = []
    for cohort, file_label, title in (
            ('formal_primary_candidates', 'formal-candidates', 'Formal primary complete valid trajectories only'),
            ('all_stages_all_attempts', 'all-stages', 'All stages/all attempts, including calibration and invalids')):
        cohort_arcs = arc_cohort_rows(arcs, cohort)
        cohort_attempts = arc_cohort_rows(attempts, cohort)
        candidate_ids = {row['trajectory_id'] for row in cohort_attempts}
        cohort_controls = [row for row in controls if row['trajectory_id'] in candidate_ids
                           and (cohort == 'all_stages_all_attempts' or row['valid'])]
        for letter in LETTERS:
            rows = [row for row in cohort_arcs if row['letter'] == letter]
            trajectories = [row for row in cohort_attempts if row['letter'] == letter]
            physical_controls = [row for row in cohort_controls if row['letter'] == letter]
            ao_seconds = sum(row['AO_seconds'] for row in physical_controls)
            hits = Counter()
            micro_counts = Counter()
            for row in rows:
                hits[(row['ring'], row['arc_bin'])] += row['seconds']
                micro_counts[(row['ring'], row['arc_bin'])] += row['microstep_count']
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            shape = _local_geometry(letter)
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
            ax.set_title(f'{letter}: {title}\nN={len(trajectories)} trajectories / AO={ao_seconds:.3f}s\nActual8-world-px arc seconds; dashed closed holes not exterior targets', fontsize=9)
            file_name = f'arcs-{file_label}-{letter}.png'
            _figure_output(fig, destination/file_name, resource_reservation)
            plt.close(fig)
            exterior = [segment for segment in contour_segments(letter, 8) if segment['ring'] == 0]
            summaries.append(dict(letter=letter, cohort=cohort, figure=file_name,
                denominator=dict(trajectory_attempts=len(trajectories), actual_controls=len(physical_controls),
                                 actual_physical_seconds=sum(row['seconds'] for row in physical_controls),
                                 actual_agent_object_contact_seconds=ao_seconds,
                                 phases=dict(Counter(row.get('phase_group', row.get('phase')) for row in trajectories)),
                                 splits=dict(Counter(row['split'] for row in trajectories))),
                exterior_arc_count=len(exterior),
                observed_exterior_arcs=sum(value > 0 for (ring, _), value in hits.items() if ring == 0),
                observed_closed_hole_arcs=sum(value > 0 for (ring, _), value in hits.items() if ring > 0),
                unknown_projection_seconds=hits[(-1, -1)],
                arc_seconds_are_nonexclusive='Each actual arc is counted at most once/microstep; simultaneous arcs overlap in time, so arc seconds are not a contact-time fraction',
                closed_holes='exterior-inaccessible; never requested',
                arc_contact_seconds=[dict(ring=ring, arc_bin=arc, seconds=value,
                                          microstep_count=micro_counts[(ring, arc)]) for (ring, arc), value in sorted(hits.items())]))
    return summaries


def _video(destination, metadata, data, controls, resource_reservation=None):
    """Actual saved RGB, nearest display only;10fps at10Hz, no physics replay."""
    import cv2
    import imageio_ffmpeg
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = (640, 592)
    previous_size = destination.stat().st_size if destination.exists() else 0
    existing_video_bytes = sum(path.stat().st_size for path in destination.parent.glob('*.mp4'))
    available_video_bytes = RULES['video_cap_bytes']-existing_video_bytes+previous_size
    # FFmpeg's file limit is packet-granular. Reserve an extra raw-frame-sized
    # margin, and require the fully decoded frame count afterward. A truncated
    # bounded file is retained as failure evidence, never reported as a video.
    packet_margin = size[0]*size[1]*3+1024**2
    available_report_bytes = (resource_reservation.remaining_bytes if resource_reservation is not None else
                              128*1024**2)
    encoded_limit = min(8*1024**2, available_video_bytes-packet_margin,
                        available_report_bytes-packet_margin)
    if encoded_limit < 64*1024:
        raise RuntimeError('Insufficient bounded video/report budget before encoding')
    reserved_peak = encoded_limit+packet_margin
    if resource_reservation is not None:
        resource_reservation.ensure_capacity(reserved_peak)
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen([executable, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{size[0]}x{size[1]}',
        '-r', '10', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '24',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-fs', str(encoded_limit), str(destination)],
        stdin=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    slot = metadata['slot']
    positions = centers(slot['letter'], data['states'])
    encoder_error = None
    try:
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
    except Exception as exc:
        encoder_error = exc
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        return_code = process.wait()
        if resource_reservation is not None and destination.exists():
            resource_reservation.commit_external_write(destination, previous_size=previous_size)
    if encoder_error is not None:
        raise RuntimeError('Bounded video encoder failed; partial output retained') from encoder_error
    if return_code != 0:
        raise RuntimeError('Video encoder failed')
    if destination.stat().st_size > reserved_peak or sum(path.stat().st_size for path in destination.parent.glob('*.mp4')) > RULES['video_cap_bytes']:
        raise RuntimeError('Bounded encoder exceeded its reserved video allocation; output retained')
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


def build_report(output, *, videos=False, resource_reservation=None):
    output = Path(output).resolve()
    report = output/'report'
    report.mkdir(parents=True, exist_ok=True)
    initial_resource_snapshot = resource_reservation.snapshot() if resource_reservation is not None else None
    controls, windows, arcs, attempts = [], [], [], []
    by_id = {}
    for path in sorted((output/'trajectories').glob('*/metadata.json')):
        metadata = json.loads(path.read_text(encoding='utf-8'))
        if metadata['raw'] is None:
            slot = metadata['slot']
            attempts.append(dict(trajectory_id=slot['slot_id'], family_id=slot['family_id'],
                                 letter=slot['letter'], phase=slot['phase'], split=slot['split'],
                                 initial_space=slot['initial_space'], scenario=slot['scenario'],
                                 actual_controls=0, valid_controls=0,
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
    _motion_plots(report, windows, resource_reservation)
    arc_summary = _arc_plots(report, arcs, attempts, controls, resource_reservation)
    video_rows = []
    if videos:
        selection = []
        for letter in LETTERS:
            choices = [row for row in attempts if row['letter'] == letter and row.get('raw')]
            if not choices:
                continue
            has_formal = any(row['phase'] == 'formal' for row in choices)
            example_cohort = 'formal_primary_only' if has_formal else 'calibration_only_no_formal_data_yet'
            complete = [row for row in choices if row['complete'] and
                        (row['phase'] == 'formal' if has_formal else row['phase'].startswith('calibration'))]
            if complete:
                translation = max(complete, key=lambda row: row['max_translation_image_px'])
                rotation = max(complete, key=lambda row: row['max_rotation_deg'])
                boundary = max(complete, key=lambda row: row['boundary_seconds'])
                weak = min(complete, key=lambda row: row['max_translation_image_px'])
                random = sorted(complete, key=lambda row: hashlib.sha256(('fixed-video-seed-2026103019:'+row['trajectory_id']).encode()).digest())[0]
                selection.extend((example_cohort+':'+tag, row) for tag, row in [('clear_translation', translation), ('clear_rotation', rotation),
                    ('boundary', boundary), ('weak_motion', weak), ('fixed_seed_random_complete', random)])
            rejected = [row for row in choices if not row['complete']]
            if rejected:
                selection.append(('all_stages_rejected_first:'+rejected[0]['phase'], rejected[0]))
        generated = {}
        for tag, entry in selection:
            tid = entry['trajectory_id']
            if tid not in generated:
                path, trajectory_controls = by_id[tid]
                metadata = json.loads(path.read_text(encoding='utf-8'))
                data = load_attempt(path.parent/'raw.npz')
                generated[tid] = _video(report/'videos'/(tid+'.mp4'), metadata, data, trajectory_controls, resource_reservation)
            video_rows.append(dict(selection=tag, trajectory_id=tid, letter=entry['letter'],
                                   phase_group=entry['phase'], split=entry['split'], **generated[tid]))
        if sum(row['bytes'] for row in generated.values()) > RULES['video_cap_bytes']:
            raise RuntimeError('Actual report videos exceeded64MiB; preserve files and stop')
    all_candidate = [row for row in windows if row['candidate']]
    weak_letters = []
    for letter in LETTERS:
        letter_windows = [row for row in windows if row['letter'] == letter]
        audit_phase = ('formal' if any(row['phase_group'] == 'formal' for row in letter_windows)
                       else 'calibration-v2' if any(row['phase_group'] == 'calibration-v2' for row in letter_windows)
                       else 'calibration-v1')
        h1 = [row for row in letter_windows if row['horizon'] == 1 and row['valid']
              and row['phase_group'] == audit_phase and (row['candidate'] if audit_phase == 'formal' else True)]
        weak = sum(row['translation_image_px'] < .5 for row in h1)/len(h1) if h1 else None
        rotation_trajectories = {row['trajectory_id'] for row in h1 if row['rotation_deg'] >= 5.}
        weak_condition = weak is None or weak > .5 or not rotation_trajectories
        weak_letters.append(dict(letter=letter, audit_phase=audit_phase, H1_N=len(h1), translation_under_half_image_px_fraction=weak,
                                 H1_obvious_rotation_trajectories=len(rotation_trajectories),
                                 weak_motion_condition=weak_condition,
                                 supplement_trigger=weak_condition if audit_phase == 'formal' else None,
                                 supplement_release_authorized=False))
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
    summary['metric_contract']['conditioning_packet_identity'] = (
        'v2 SHA: four actual historyRGB frame digests plus past3requested actions and nextH requested actions; '
        'world-coordinate float64 before normalization. This is an external rollout packet, not complete '
        'network input tensors or later generated autoregressive histories. No futureGT RGB/state in this hash.')
    summary['metric_contract']['actual_arc_export'] = (
        'One aggregate row per trajectory/control/phase/split/complete/valid/actual arc. '
        'Actual dt sum and distinct microstep count retained, with local first/last index and ordered-index SHA. '
        'All individual microstep contacts/points/dt remain losslessly in raw; formatter never runs physics.')
    summary['report_formatter_source_sha256'] = file_sha(__file__)
    summary['resource_write_policy'] = dict(guarded=resource_reservation is not None,
        initial_snapshot=initial_resource_snapshot,
        resource_policy_sha256=resource_reservation.policy_sha256 if resource_reservation is not None else None,
        contract='Preallocated128MiB report outputs; exact-byte write-before guards, bounded per-video encoding; videos remain64MiB total')
    _write_output(report/'summary.json', (json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8'), resource_reservation)
    csv_write(report/'control-metrics.csv', controls, resource_reservation)
    csv_write(report/'window-metrics.csv', windows, resource_reservation)
    csv_write(report/'motion-summary.csv', grouped, resource_reservation)
    csv_write(report/'contact-denominators.csv', contacts, resource_reservation)
    csv_write(report/'actual-contact-arcs.csv', arcs, resource_reservation)
    csv_write(report/'trajectory-index.csv', attempts, resource_reservation)
    def table(rows, columns):
        return '<table><tr>'+''.join('<th>'+html.escape(key)+'</th>' for key in columns)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(row.get(key)))+'</td>' for key in columns)+'</tr>' for row in rows)+'</table>'
    panels = []
    for letter in LETTERS:
        rows = []
        for row in grouped:
            if row['letter'] != letter or any(key in row for key in ('planned_phase', 'boundary', 'observed_contact_stage')):
                continue
            if row['cohort'] == 'valid_physical_windows' and 'phase_group' not in row:
                rows.append(dict(row, displayed_cohort='All stages valid physical windows, including calibration'))
            elif row['cohort'] == 'candidate_model_windows' and row.get('phase_group') == 'formal':
                rows.append(dict(row, displayed_cohort='Formal primary candidate packets only'))
        panels.append(f'<section><h2>{letter}</h2>'
                      f'<h3>All-stage valid physical motion (includes calibration)</h3><img src="motion-all-valid-stages-{letter}.png">'
                      f'<h3>Formal primary candidate motion</h3><img src="motion-formal-candidates-{letter}.png">'
                      f'<h3>Formal primary complete valid trajectory coverage</h3><img src="arcs-formal-candidates-{letter}.png">'
                      f'<h3>All-stage/all-attempt coverage (includes calibration/supplement/invalid)</h3><img src="arcs-all-stages-{letter}.png">'+
                      table([dict(cohort=r['displayed_cohort'], H=r['horizon'], N=r['N'], trajectories=r['independent_trajectories'],
                           translation_P50=r['translation_image_px']['P50'], translation_P90=r['translation_image_px']['P90'],
                           rotation_P50=r['rotation_deg']['P50'], rotation_P90=r['rotation_deg']['P90'],
                           categories=r['motion_categories']) for r in rows],
                           ['cohort', 'H', 'N', 'trajectories', 'translation_P50', 'translation_P90', 'rotation_P50', 'rotation_P90', 'categories'])+
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
        ['letter', 'audit_phase', 'H1_N', 'translation_under_half_image_px_fraction', 'H1_obvious_rotation_trajectories', 'weak_motion_condition', 'supplement_trigger'])
    content += ''.join(panels)
    _write_output(report/'index.html', content.encode('utf-8'), resource_reservation)
    text = '# Original contact pilot: saved data report\n\n'+json.dumps(headline, indent=2)+'\n\n'
    text += '[Interactive report](report/index.html), [all saved metrics](report/summary.json).\n\n'
    text += 'Candidate data only, training0. All attempts, valid physical time and candidate windows have separate denominators. '
    text += 'Physics修复/清理交付由负责人根报告链接；本文件只读真实保存图像、状态与接触，不重放物理。\n'
    _write_output(output/'REPORT.md', text.encode('utf-8'), resource_reservation)
    return headline
