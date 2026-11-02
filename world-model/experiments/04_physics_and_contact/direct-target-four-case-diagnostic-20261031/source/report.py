"""Saved-only report for the four approved original-physics diagnostics.

No simulator/model imports or construction. Native telemetry is observed truth;
return below slop during contact is distinct from zero after separation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import zlib

import numpy as np

CAP = 16 * 1024**2
SHARED_CAP = 3 * 1024**3
ATOMIC_RESERVE = 256 * 1024**2
MEDIA_RESERVE = 2 * 1024**2
IDS = ('DT-O-01-0-b0', 'DT-T-00-0-b0', 'DT-L-05-0-b0', 'DT-C-06-1-b0')
PAIR_INDEX = {'OW': 0, 'AO': 1, 'AW': 2}
KINDS = {'AO': 'agent_object', 'OW': 'object_wall', 'AW': 'agent_wall'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


def bounded_write(root, path, data):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Report output escaped the approved new directory')
    data = bytes(data)
    # Atomic peak includes the old file, new temporary file and all other output.
    if tree_bytes(root)+len(data) > CAP:
        raise RuntimeError('Four-case16MiB output cap unavailable; no deletion')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.report-tmp')
    with tmp.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def storage_preflight(root, config):
    roots_path = config.get('storage_roots', config.get('storage_roots_json'))
    roots = [root]
    if roots_path:
        value = read(roots_path)
        roots += value.get('roots', []) if isinstance(value, dict) else value
    resolved = sorted({Path(p).resolve() for p in roots}, key=lambda p: len(str(p)))
    primary = []
    for path in resolved:
        if not any(path == old or path.is_relative_to(old) for old in primary):
            primary.append(path)
    total, files, missing = 0, 0, []
    for base in primary:
        if not base.exists():
            missing.append(str(base)); continue
        if base.is_file():
            total += base.stat().st_size; files += 1; continue
        for parent, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = [name for name in dirs if not (Path(parent)/name).is_symlink()
                       and not (getattr((Path(parent)/name).stat(), 'st_file_attributes', 0) & 1024)]
            for name in names:
                path = Path(parent)/name
                if not path.is_symlink():
                    total += path.stat().st_size; files += 1
    free = shutil.disk_usage(root).free
    if missing or total+ATOMIC_RESERVE+MEDIA_RESERVE > SHARED_CAP or free < ATOMIC_RESERVE+MEDIA_RESERVE:
        raise RuntimeError('Shared3GiB/256MiB atomic report capacity unavailable')
    if tree_bytes(root)+MEDIA_RESERVE > CAP:
        raise RuntimeError('Four-case media allocation unavailable')
    return dict(shared_bytes=total, shared_files=files, missing_roots=missing,
                disk_free_bytes=free, shared_cap_bytes=SHARED_CAP,
                atomic_reserve_bytes=ATOMIC_RESERVE, media_peak_reserve_bytes=MEDIA_RESERVE)


def load_raw(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    payload = zlib.decompress(data.pop('telemetry_zlib').tobytes())
    if 'telemetry_sha256' in data and hashlib.sha256(payload).digest() != data.pop('telemetry_sha256').tobytes():
        raise ValueError('Saved native telemetry checksum mismatch')
    data.update(json.loads(payload))
    n = len(data['requested_actions'])
    if data['states'].shape != (n+1, 15) or data['rgb'].shape != (n+1, 3, 96, 96):
        raise ValueError('State/RGB/action alignment mismatch')
    if not np.array_equal(data['requested_actions'], data['passed_actions']):
        raise ValueError('Requested target differs from actual environment input')
    if len(data['telemetry']) != n or len(data['control_time_seconds']) != n+1:
        raise ValueError('Saved timeline does not align with native controls')
    return data


def native_info(events, pair, phase=None):
    selected = [e for e in events if e['kind'] == KINDS[pair]
                and (phase is None or e['phase'] == phase)]
    distances = [float(p['distance']) for e in selected for p in e.get('points', [])]
    positive = [e for e in selected if e['phase'] == 'post_solve'
                and np.linalg.norm(e.get('total_impulse', [0., 0.])) > 0.]
    return dict(contact=bool(distances), minimum_distance=min(distances) if distances else None,
                depth=max(0., -min(distances)) if distances else None,
                positive_impulse=bool(positive),
                impulse_norm=sum(float(np.linalg.norm(e['total_impulse'])) for e in positive))


def normal_velocity(events, pair, pre, post, physical_cog):
    if pair not in ('AO', 'OW'):
        return None, None
    event = next((e for e in events if e['kind'] == KINDS[pair] and e.get('points')
                  and e['phase'] == 'post_solve'), None)
    if event is None:
        event = next((e for e in events if e['kind'] == KINDS[pair] and e.get('points')), None)
    if event is None:
        return None, None
    oi = event['shapes_order'].index('object')
    normal = np.asarray(event['normal'], float)*(1. if oi == 0 else -1.)
    point = np.asarray(event['points'][0]['point_a' if oi == 0 else 'point_b'], float)
    c, s = math.cos(post[4]), math.sin(post[4])
    cog = post[2:4]+np.asarray((c*physical_cog[0]-s*physical_cog[1], s*physical_cog[0]+c*physical_cog[1]))
    arm = point-cog
    def evaluate(state):
        object_v = state[7:9]+state[9]*np.asarray((-arm[1], arm[0]))
        other_v = state[5:7] if pair == 'AO' else np.zeros(2)
        return float((other_v-object_v)@normal)
    return evaluate(pre), evaluate(post)


def local_object_geometry(identity):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    return unary_union([Polygon(s['geometry']['vertices']) for s in identity['physics']['shapes']
                        if s['role'] == 'object'])


def signed_gap(local_geometry, state, pair, pusher_radius=15.):
    from shapely import affinity
    from shapely.geometry import Point
    world = affinity.translate(affinity.rotate(local_geometry, state[4], origin=(0, 0), use_radians=True), *state[2:4])
    if pair == 'OW':
        x0, y0, x1, y1 = world.bounds
        return float(min(x0-7., y0-7., 504.-x1, 504.-y1))
    point = Point(*state[:2])
    distance = -world.boundary.distance(point) if world.contains(point) else world.distance(point)
    return float(distance-pusher_radius)


def micro_rows(data, identity, primary_pair):
    geometry = local_object_geometry(identity)
    cog = np.asarray(identity['physics']['object']['center_of_gravity'], float)
    rows, elapsed = [], 0.
    for control, info in enumerate(data['telemetry']):
        target = data['requested_actions'][control]
        for micro, sub in enumerate(info.get('substeps', [])):
            if not sub.get('executed'):
                continue
            dt = float(sub['dt']); elapsed += dt
            if abs(dt-.01) > 1e-12:
                raise ValueError('Non-original substep duration in saved diagnostic')
            pre, post = np.asarray(sub['pre_state'], float), np.asarray(sub['post_state'], float)
            events = sub.get('contacts', [])
            row = dict(control=control, substep=micro, time_seconds=elapsed, dt=dt,
                       target_error_pre=float(np.linalg.norm(target-pre[:2])),
                       target_error_post=float(np.linalg.norm(target-post[:2])),
                       pusher_travel=float(np.linalg.norm(post[:2]-pre[:2])),
                       hard_stop_reasons=sub.get('diagnostic_hard_reasons', sub.get('hard_stop_reasons', [])),
                       pusher_region=sub.get('pusher_region'))
            for pair, index in PAIR_INDEX.items():
                ni = native_info(events, pair)
                row[pair+'_depth'] = float(sub['penetrations'][index])
                row[pair+'_native_contact'] = ni['contact']
                row[pair+'_native_distance'] = ni['minimum_distance']
                row[pair+'_native_depth'] = ni['depth']
                row[pair+'_positive_impulse'] = ni['positive_impulse']
                row[pair+'_impulse_norm'] = ni['impulse_norm']
                for phase in ('pre_solve', 'post_solve'):
                    row[pair+'_'+phase+'_distance'] = native_info(events, pair, phase)['minimum_distance']
            before, after = normal_velocity(events, primary_pair, pre, post, cog)
            row.update(normal_v_detection_reconstructed=before, normal_v_post_solve=after,
                       primary_signed_geometric_gap=signed_gap(geometry, post, primary_pair))
            rows.append(row)
    if abs(elapsed-float(data['control_time_seconds'][-1])) > 1e-8:
        raise ValueError('Native executed time does not equal saved frame timeline')
    return rows


def episode_metrics(rows, pair, old_trigger_micro, slop):
    observed = rows[old_trigger_micro:] if old_trigger_micro < len(rows) else []
    if not observed:
        return dict(observed_after_old_stop=False, interpretation='No preserved continuation available')
    field = pair+'_depth'
    above = [r[field] > 1.+1e-8 for r in observed]
    episodes, current = [], None
    for index, flag in enumerate(above+[False]):
        if flag and current is None:
            current = index
        elif not flag and current is not None:
            part = observed[current:index]
            episodes.append(dict(start_time=part[0]['time_seconds'], end_time=part[-1]['time_seconds'],
                                 observed_seconds=sum(x['dt'] for x in part),
                                 peak=max(x[field] for x in part), continues_at_end=index == len(observed)))
            current = None
    recovered = next((i for i in range(1, len(observed)) if observed[i][field] <= slop+1e-8), None)
    return_row = observed[recovered] if recovered is not None else None
    continuous_contact = (all(r[pair+'_native_contact'] for r in observed[:recovered+1])
                          if recovered is not None else None)
    separated = [i for i, r in enumerate(observed) if r['primary_signed_geometric_gap'] > 1e-8]
    separated_before_return = (any(i < recovered for i in separated) if recovered is not None else None)
    separated_by_return = (any(i <= recovered for i in separated) if recovered is not None else None)
    if return_row is None:
        kind = 'no_return_to_slop_observed_in_budget'
    elif return_row[pair+'_native_contact']:
        kind = ('return_to_slop_during_continuous_native_contact' if continuous_contact
                else 'return_during_contact_after_manifold_gap_or_unknown')
    elif return_row['primary_signed_geometric_gap'] > 1e-8:
        kind = 'return_after_geometric_separation'
    else:
        kind = 'return_without_native_manifold_touching_or_unknown'
    reexceed = [] if recovered is None else [r for r in observed[recovered+1:] if r[field] > 1.+1e-8]
    native_errors = [abs(r[field]-r[pair+'_native_depth']) for r in observed if r[pair+'_native_depth'] is not None]
    return dict(old_stop_point_observed=True, observed_after_old_stop=len(observed)>1,
                old_stop_time_seconds=observed[0]['time_seconds'],
                observed_continuation_seconds=sum(r['dt'] for r in observed[1:]),
                peak_depth=max(r[field] for r in observed), final_depth=observed[-1][field],
                final_native_distance=observed[-1][pair+'_native_distance'],
                final_native_contact=observed[-1][pair+'_native_contact'],
                final_positive_impulse=observed[-1][pair+'_positive_impulse'],
                final_signed_geometric_gap=observed[-1]['primary_signed_geometric_gap'],
                native_depth_max_abs_error=max(native_errors) if native_errors else None,
                positive_impulse_seconds=sum(r['dt'] for r in observed if r[pair+'_positive_impulse']),
                positive_impulse_norm_sum=sum(r[pair+'_impulse_norm'] for r in observed),
                strictly_followup_positive_impulse_seconds=sum(r['dt'] for r in observed[1:] if r[pair+'_positive_impulse']),
                strictly_followup_positive_impulse_norm_sum=sum(r[pair+'_impulse_norm'] for r in observed[1:]),
                above_1_microsteps=sum(above), above_1_seconds=sum(r['dt'] for r, flag in zip(observed, above) if flag),
                strictly_followup_above_1_microsteps=sum(above[1:]),
                strictly_followup_above_1_seconds=sum(r['dt'] for r, flag in zip(observed[1:], above[1:]) if flag),
                above_1_episodes=episodes, longest_above_1_observed_seconds=max((x['observed_seconds'] for x in episodes), default=0.),
                return_kind=kind, first_return_time_seconds=return_row['time_seconds'] if return_row else None,
                continuous_native_contact_until_return=continuous_contact,
                any_geometric_separation_before_return=separated_before_return,
                geometric_separation_at_or_before_return=separated_by_return,
                first_geometric_separation_time_seconds=observed[separated[0]]['time_seconds'] if separated else None,
                first_return_depth=return_row[field] if return_row else None,
                first_return_native_contact=return_row[pair+'_native_contact'] if return_row else None,
                first_return_positive_impulse=return_row[pair+'_positive_impulse'] if return_row else None,
                first_return_signed_gap=return_row['primary_signed_geometric_gap'] if return_row else None,
                later_reexceed_1_microsteps=len(reexceed),
                later_reexceed_1_peak=max((r[field] for r in reexceed), default=None),
                metric_interval_definition='Peak/above1episodes/positive_impulse_seconds include the reproduced old cutpoint interval. strictly_followup fields exclude that first interval. before_return is strict; at_or_before_return additionally includes the return sample.',
                interpretation='Finite diagnostic observation only; no production validity or training admission claim')


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return stream.getvalue().encode('utf-8-sig')


def plot_bytes(rows, pair, old_time, title, root):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(root)/'report/.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = [r['time_seconds'] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for p, color in (('AO', 'tab:blue'), ('OW', 'tab:purple'), ('AW', 'tab:green')):
        axes[0].plot(t, [r[p+'_depth'] for r in rows], label='custom '+p, color=color)
    axes[0].plot(t, [r[pair+'_native_depth'] if r[pair+'_native_depth'] is not None else np.nan for r in rows],
                 '.', color='tab:orange', label='native '+pair)
    axes[0].axhline(1., color='red', ls=':', label='legacy would-stop=1')
    axes[0].axhline(.1, color='gray', ls='--', label='original slop ~.1')
    axes[0].set_ylabel('depth, world units'); axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(t, [r[pair+'_impulse_norm'] for r in rows], label=pair+' positive impulse norm')
    axes[1].plot(t, [r['target_error_post'] for r in rows], label='target error post-step')
    axes[1].set_ylabel('impulse / world units'); axes[1].legend(fontsize=8)
    for key, label in (('normal_v_detection_reconstructed', 'detection/pre_solve relative normal v (reconstructed)'),
                       ('normal_v_post_solve', 'post_solve relative normal v')):
        axes[2].plot(t, [r[key] if r[key] is not None else np.nan for r in rows], label=label)
    axes[2].axhline(0., color='gray', lw=.6); axes[2].set_ylabel('world units/s'); axes[2].legend(fontsize=7)
    axes[3].plot(t, [r['primary_signed_geometric_gap'] for r in rows], label=pair+' signed geometric gap')
    axes[3].plot(t, [r['pusher_travel'] for r in rows], label='actual pusher travel/.01s')
    axes[3].axhline(0., color='gray', lw=.6); axes[3].set_ylabel('world units'); axes[3].legend(fontsize=8)
    axes[3].set_xlabel('actual cumulative simulation seconds (not CPU time)')
    for ax in axes:
        ax.axvline(old_time, color='red', ls='--', lw=.8); ax.grid(alpha=.2)
    fig.suptitle(title+' | saved truth only; no replay/model')
    stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=120); plt.close(fig)
    return stream.getvalue()


def board_bytes(data, case_id):
    import cv2
    n = len(data['rgb'])
    indices = sorted(set([0, min(3, n-1), min(4, n-1), (n-1)//2, n-1]))
    width = 200
    canvas = np.full((250, width*len(indices), 3), 255, np.uint8)
    for j, i in enumerate(indices):
        picture = np.transpose(data['rgb'][i], (1, 2, 0))
        picture = cv2.resize(picture, (192, 192), interpolation=cv2.INTER_NEAREST)
        canvas[42:234, j*width+4:j*width+196] = picture
        cv2.putText(canvas, f'frame{i} t={data["control_time_seconds"][i]:.2f}s',
                    (j*width+4, 34), cv2.FONT_HERSHEY_SIMPLEX, .38, (20, 20, 20), 1)
    cv2.putText(canvas, case_id+' | SAVED TRUTH, NOT PREDICTION', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .36, (20, 20, 20), 1)
    ok, png = cv2.imencode('.png', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('Truth board PNG encoding failed')
    return png.tobytes()


def save_video(root, path, case_id, data):
    import cv2
    import imageio_ffmpeg
    if tree_bytes(root)+MEDIA_RESERVE > CAP:
        raise RuntimeError('Video peak allocation unavailable')
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '384x424', '-r', '10',
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(path)]
    kwargs = {}
    if os.name == 'nt':
        si = subprocess.STARTUPINFO(); si.dwFlags |= subprocess.STARTF_USESHOWWINDOW; si.wShowWindow = 0
        kwargs.update(startupinfo=si, creationflags=subprocess.CREATE_NO_WINDOW)
    frames = []
    for index, frame in enumerate(data['rgb']):
        image = np.full((424, 384, 3), 255, np.uint8)
        image[40:424] = cv2.resize(np.transpose(frame, (1, 2, 0)), (384, 384), interpolation=cv2.INTER_NEAREST)
        cv2.putText(image, case_id+' | SAVED TRUTH', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .36, (20, 20, 20), 1)
        cv2.putText(image, f'frame {index}, actual t={data["control_time_seconds"][index]:.2f}s, 10fps',
                    (5, 31), cv2.FONT_HERSHEY_SIMPLEX, .36, (20, 20, 20), 1)
        frames.append(image.tobytes())
    result = subprocess.run(command, input=b''.join(frames), stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    if result.returncode or not path.is_file():
        raise RuntimeError('Saved truth video encoding failed: '+result.stderr.decode(errors='replace')[-1000:])
    if path.stat().st_size > MEDIA_RESERVE or tree_bytes(root) > CAP:
        raise RuntimeError('Video exceeded its allocation')
    capture = cv2.VideoCapture(str(path)); count = 0; fps = capture.get(cv2.CAP_PROP_FPS)
    while capture.read()[0]:
        count += 1
    capture.release()
    if count != len(data['rgb']) or abs(fps-10.) > .001:
        raise RuntimeError('Actual decoded video timing/frame count mismatch')
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, fps=fps, decoded_frames=count,
                executed_simulation_seconds=float(data['control_time_seconds'][-1]),
                timing='Every available real saved frame once at10fps; initial/end frame adds.1s display time; invalid partial timestamp remains actual.')


def generate_report(root):
    root = Path(root).resolve(); report = root/'report'
    if (report/'summary.json').exists():
        raise FileExistsError('Four-case saved report already exists; no automatic regeneration')
    config = read(root/'runtime-config.json'); approval = read(root/'APPROVAL.json')
    storage = storage_preflight(root, config)
    ledger = read(root/'ledger.json') if (root/'ledger.json').exists() else {}
    completion = read(root/'completion.json') if (root/'completion.json').exists() else {}
    ledger_attempts = {a.get('attempt_id', a.get('id')): a.get('cost', a) for a in ledger.get('attempts', [])}
    source = Path(config.get('old_root', config.get('source_data_root', root.parents[1]/'artifacts/direct-target-acceptance-20261031')))
    old_plan = read(source/'plan.json'); plans = {a['attempt_id']: a for a in old_plan['attempts']}
    intended = approval['cases']; rows, inputs = [], []
    for item in intended:
        case_id = item.get('id', item.get('attempt_id'))
        if case_id not in IDS:
            raise ValueError('Unapproved fifth case in report input')
        path, meta_path = root/'results'/f'{case_id}.npz', root/'results'/f'{case_id}.json'
        old_meta_path = source/'collection/attempts'/f'{case_id}.json'
        old = read(old_meta_path); trigger = old['first_invalid']; pair = 'OW' if case_id.startswith('DT-C-') else 'AO'
        claim = ledger_attempts.get(case_id, {})
        entry = dict(attempt_id=case_id, letter=plans[case_id]['letter'], primary_pair=pair,
                     requested_control_cap=item['control_cap'], requested_substep_cap=item['regular_substep_cap'],
                     attempted=meta_path.exists() or bool(claim), raw_saved=path.exists(), source_metadata_sha256=sha(old_meta_path),
                     source_raw_sha256=old['raw']['sha256'], old_trigger_control=trigger['control'],
                     old_trigger_substep=trigger['substep'], training_eligible=False)
        if not meta_path.exists() or not path.exists():
            entry.update(status='raw_or_metadata_unavailable' if claim else 'unstarted',
                         actual_controls=claim.get('actual_controls', 0),
                         actual_substeps=claim.get('actual_substeps', 0),
                         initialization_integrator_calls=claim.get('initialization_integrator_calls'),
                         hard_stop=claim.get('hard_stop'), failure=claim.get('error', claim.get('failure')),
                         interpretation='Ledger actual calls retained; no unavailable reconstructed motion/RGB outcome claimed')
            rows.append(entry); continue
        meta = read(meta_path); data = load_raw(path); n = len(data['requested_actions'])
        if not np.array_equal(data['requested_actions'], np.asarray(plans[case_id]['actions'][:n], float)):
            raise ValueError('Diagnostic actions differ from original frozen source array')
        expected_raw = meta.get('raw', {})
        if expected_raw.get('sha256') and expected_raw['sha256'] != sha(path):
            raise ValueError('Actual diagnostic raw hash differs from receipt')
        identity = meta.get('initial_identity') or old['initial_identity']
        slop = float(identity['physics']['space']['collision_slop'])
        micros = micro_rows(data, identity, pair)
        old_micro = trigger['control']*10+trigger['substep']
        metrics = episode_metrics(micros, pair, old_micro, slop)
        cost = meta.get('cost', {})
        entry.update(status=meta.get('status', cost.get('status', 'unknown')),
                     actual_controls=cost.get('actual_controls', n), actual_substeps=cost.get('actual_substeps', len(micros)),
                     raw_controls=n, raw_substeps=len(micros),
                     unframed_terminal_control=meta.get('unframed_terminal_control', cost.get('unframed_terminal_control')),
                     simulated_seconds=cost.get('simulated_seconds', float(data['control_time_seconds'][-1])),
                     initialization_integrator_calls=cost.get('initialization_integrator_calls'),
                     wall_seconds=cost.get('wall_seconds'), prefix_match=meta.get('prefix_match'),
                     hard_stop=meta.get('hard_stop'), failure=meta.get('failure'), original_slop=slop,
                     raw_sha256=sha(path), requested_equals_passed_equals_source=True, metrics=metrics)
        if entry['actual_controls'] > item['control_cap'] or entry['actual_substeps'] > item['regular_substep_cap']:
            raise ValueError('Actual case exceeded approved physical budget')
        for p in PAIR_INDEX:
            segment = micros[old_micro:] if old_micro < len(micros) else []
            entry[p+'_peak_after_old_stop'] = max((x[p+'_depth'] for x in segment), default=None)
            entry[p+'_final_depth'] = micros[-1][p+'_depth'] if micros else None
        inputs.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
        if micros:
            plot = report/f'{case_id}-process.png'; board = report/f'{case_id}-truth.png'
            bounded_write(root, plot, plot_bytes(micros, pair, micros[min(old_micro, len(micros)-1)]['time_seconds'], case_id, root))
            bounded_write(root, report/f'{case_id}-substeps.csv', csv_bytes(micros))
            bounded_write(root, board, board_bytes(data, case_id))
            entry.update(process_plot=str(plot), truth_board=str(board),
                         video=save_video(root, report/'videos'/f'{case_id}.mp4', case_id, data))
        rows.append(entry)
    summary = dict(kind='four-case-saved-diagnostic-report.v1', run_id=config.get('run_id'),
        runtime_config_sha256=sha(root/'runtime-config.json'), intended_cases=4,
        actual_saved_cases=sum(r['raw_saved'] for r in rows), cases=rows,
        actual_controls=ledger.get('actual_controls', sum(r.get('actual_controls', 0) for r in rows)),
        actual_substeps=ledger.get('actual_substeps', sum(r.get('actual_substeps', 0) for r in rows)),
        raw_observed_controls=sum(r.get('raw_controls', 0) for r in rows),
        raw_observed_substeps=sum(r.get('raw_substeps', 0) for r in rows),
        initialization_integrator_calls=ledger.get('initialization_integrator_calls', sum(r.get('initialization_integrator_calls') or 0 for r in rows)),
        ledger=ledger, completion=completion, storage_preflight=storage,
        source_raw_files=inputs, training_eligible=False, model_calls=0, optimizer_updates=0, report_physics_calls=0,
        return_definition='First strictly later saved sample <=actual collision_slop+1e-8. Continuous native-contact return requires a manifold at every sample from old cutpoint through return. A return-row manifold after a missing manifold is labelled gap/unknown. Absent return manifold plus positive signed gap means after separation; otherwise ambiguous. Intervening geometric separation and later re-exceedance are separately retained.',
        timing_note='Detection/pre_solve normal velocity is reconstructed from PD-updated integrator-pre velocity at post-integration contact geometry; dynamic damping update occurs before impulse solve. It is not an extra observed immediate pre-impulse velocity.',
        limitations=['Four preselected censored examples do not estimate the74-case false-positive rate.',
            'A return during contact, a return after separation and absence of return are separate outcomes.',
            'No hard stop or a short return below slop does not establish production/training validity.',
            'At approved budget end, unobserved later recovery or deterioration remains unknown.',
            'Production guard changes,19/35/51sampling,index activation,collection expansion and training remain unapproved.'])
    if summary['actual_controls'] > 35 or summary['actual_substeps'] > 350:
        raise ValueError('Overall approved budget exceeded')
    bounded_write(root, report/'table.csv', csv_bytes([{k:v for k,v in e.items() if k not in ('video',)} for e in rows]))
    bounded_write(root, report/'summary.json', encode(summary))
    body = markdown(summary)
    bounded_write(root, root/'REPORT.md', body.encode('utf-8'))
    parts = ['<!doctype html><meta charset="utf-8"><title>Four-case diagnostic truth</title>',
             '<style>body{max-width:1100px;margin:24px auto;font:16px sans-serif}img{max-width:100%}video{width:384px}pre{white-space:pre-wrap}</style>',
             '<h1>Four-case original-physics diagnostic: saved truth</h1><pre>'+html.escape(body)+'</pre>']
    for e in rows:
        parts.append('<h2>'+html.escape(e['attempt_id'])+'</h2>')
        if e.get('process_plot'):
            parts += ['<img src="'+Path(e['process_plot']).name+'">', '<img src="'+Path(e['truth_board']).name+'">',
                      '<video controls preload="metadata" src="videos/'+Path(e['video']['path']).name+'"></video>']
    bounded_write(root, report/'index.html', '\n'.join(parts).encode('utf-8'))
    outputs = [p for p in report.rglob('*') if p.is_file() and '.mpl-cache' not in p.parts]
    bounded_write(root, report/'manifest.json', encode(dict(run_id=summary['run_id'], files=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in outputs]+[dict(path=str(root/'REPORT.md'), bytes=(root/'REPORT.md').stat().st_size, sha256=sha(root/'REPORT.md'))], report_physics_calls=0)))
    return summary


def fmt(value):
    return 'unknown' if value is None else f'{value:.5f}' if isinstance(value, (int, float)) else str(value)


def prefix_text(value):
    if not isinstance(value, dict):
        return str(value)
    return ('matched='+str(value.get('matched'))+', rows='+str(value.get('rows_checked'))+
            ', maxerr='+str(value.get('max_abs_error'))+', terminalRGB='+str(value.get('terminal_rgb_matched')))


def markdown(summary):
    lines = ['# Four-case original-physics continuation diagnostic', '',
        f'Run `{summary["run_id"]}`. Saved cases {summary["actual_saved_cases"]}/4; actual controls {summary["actual_controls"]}/35; native.01s substeps {summary["actual_substeps"]}/350; initialization calls {summary["initialization_integrator_calls"]}.',
        'Diagnostic-only / training_eligible=false. Model calls, optimizer updates and saved-report physics calls=0.', '',
        '|Case|Actual controls/substeps|Prefix match receipt|Primary peak/final depth (cutpoint inclusive)|Final native distance/contact|Strict follow-up positive impulse seconds|Return classification|Later >1 microsteps|Hard stop / status|',
        '|---|---:|---|---|---|---:|---|---:|---|']
    for e in summary['cases']:
        m = e.get('metrics', {})
        lines.append('|'+e['attempt_id']+f'|{e.get("actual_controls",0)}/{e.get("actual_substeps",0)}|'+
            prefix_text(e.get('prefix_match', 'unavailable')).replace('|', '/')+'|'+e['primary_pair']+' '+fmt(m.get('peak_depth'))+'/'+fmt(m.get('final_depth'))+'|'+
            fmt(m.get('final_native_distance'))+'/'+str(m.get('final_native_contact'))+'|'+fmt(m.get('strictly_followup_positive_impulse_seconds'))+'|'+
            m.get('return_kind', e.get('interpretation', 'unavailable'))+'|'+str(m.get('later_reexceed_1_microsteps', 'unknown'))+'|'+
            str(e.get('hard_stop') or e.get('failure') or e['status']).replace('|', '/')+'|')
    lines += ['', 'Depth is in512-world coordinates, not96px image pixels. Native signed distances/positive impulse and every preserved executed.01s sample are in per-case CSV and summary.json. Ledger actual calls remain authoritative if a terminal RGB/raw record was unavailable.',
        summary['return_definition'], '', 'Depth peaks and above1episodes include the reproduced old cutpoint interval; strict follow-up time/impulse/above1 fields exclude that old interval. Native contact continuity is checked across the entire interval, not just at the return sample. Neither contact-time return nor separation proves that sustained compression has been universally solved.',
        summary['timing_note'], '', 'Every available real saved frame appears once in10fps media. Timestamps retain invalid final partial time; no interpolation or model prediction.', '', 'Limits:']
    lines += ['- '+x for x in summary['limitations']]
    lines += ['', 'Suggested next decision: review contact persistence/recovery and confirmed geometry violations separately; do not choose a new universal2/5depth limit from four outcomes. Any production observer or sampling/index change requires human approval. Old120records and oldvalidlabels remain unchanged.']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); value = generate_report(args.root)
    print(json.dumps(dict(run_id=value['run_id'], saved_cases=value['actual_saved_cases'], controls=value['actual_controls'], substeps=value['actual_substeps'], report_physics_calls=0)))
