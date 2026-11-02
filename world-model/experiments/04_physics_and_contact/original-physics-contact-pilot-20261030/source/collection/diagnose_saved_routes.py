"""Saved-only sampler audit: no simulator, model, replay or new actions."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tdyssey.original_contact_pilot import load_attempt, write_json, LETTERS
from tdyssey.native_rgb_r2b_sampling import geometry_at, contact_frame, _stage, _departure_route, free_route

OUT = Path(__file__).resolve().parent
rows = list(csv.DictReader((OUT/'report/control-metrics.csv').open(encoding='utf-8', newline='')))
rows = [row for row in rows if row['phase_group'] == 'calibration-v1']
by_id = defaultdict(list)
for row in rows:
    by_id[row['trajectory_id']].append(row)
table = []
for letter in LETTERS:
    controls = [row for row in rows if row['letter'] == letter]
    wall_only = [row for row in controls if row['category'] == 'object_wall_only']
    table.append(dict(letter=letter, controls=len(controls),
        physical_seconds=sum(float(row['seconds']) for row in controls),
        wall_only_controls=len(wall_only),
        wall_only_seconds=sum(float(row['seconds']) for row in wall_only),
        wall_only_sampler_modes=dict(Counter(row['sampler_mode'] for row in wall_only)),
        wall_only_planned_phases=dict(Counter(row['planned_phase'] for row in wall_only)),
        refresh_reasons=dict(Counter(row['target_refresh_reason'] for row in controls if row['target_refresh_reason']))))
events = []
for tid, control_rows in by_id.items():
    refresh_rows = [row for row in control_rows if row['target_refresh_reason']]
    if not refresh_rows:
        continue
    path = OUT/'trajectories'/tid/'raw.npz'
    data = load_attempt(path)
    for row in refresh_rows:
        index = int(row['control'])
        if index == 0:
            continue
        state = data['states'][index]
        old, new = data['decisions'][index-1:index+1]
        letter = row['letter']
        geometry = geometry_at(letter, state)
        new_contact = dict(point_local=new['selected_point_local'], outward_normal_local=new['selected_normal_local'])
        point, normal, _ = contact_frame(state, new_contact)
        staging = _stage(geometry, point, normal)
        direct = None
        bridge = None
        distance = None
        if staging is not None:
            stage, center, _ = staging
            distance = float(np.linalg.norm(state[:2]-center))
            direct = free_route(geometry, state[:2], stage)
            if direct is None and _departure_route(geometry, state[:2], stage):
                direct = [state[:2].tolist(), stage.tolist()]
            old_contact = dict(point_local=old['selected_point_local'], outward_normal_local=old['selected_normal_local'])
            previous_point, previous_normal, _ = contact_frame(state, old_contact)
            for departure_distance in (48., 33., 18.):
                departure = previous_point+previous_normal*departure_distance
                if not _departure_route(geometry, state[:2], departure):
                    continue
                rest = free_route(geometry, departure, stage)
                if rest is not None:
                    bridge = [state[:2].tolist(), *rest]
                    break
        following = control_rows[index:index+4]
        events.append(dict(trajectory_id=tid, letter=letter, control=index,
            reason=new['target_refresh_reason'], actual_mode=new['mode'],
            actual_requested_action=data['requested_actions'][index].tolist(),
            new_contact_center_distance_world_px=distance,
            current_direct_route_to_new_staging_exists=direct is not None,
            old_contact_departure_bridge_geometrically_verified=bridge is not None,
            bridge_path_world=bridge,
            next4_actual_AO_seconds=sum(float(item['AO_seconds']) for item in following),
            next4_actual_OW_only_controls=sum(item['category'] == 'object_wall_only' for item in following),
            direct_push_while_new_contact_far=(new['mode'] == 'contact_push_intent' and distance is not None and distance > 8.),
            raw=str(path), observation='Existing logged future after refresh is reviewed as evidence, not simulated lookahead.'))

result = dict(kind='calibration-v1-saved-route-diagnosis.v1', physics_steps=0, model_calls=0, optimizer_updates=0,
    scope='All18 actual calibration-v1 attempts; observed outcomes plus present saved geometry only',
    per_letter=table, refresh_events=events, total_refreshes=len(events),
    direct_push_while_new_contact_far=sum(event['direct_push_while_new_contact_far'] for event in events),
    bridge_exists_when_direct_does_not=sum(event['old_contact_departure_bridge_geometrically_verified'] and not event['current_direct_route_to_new_staging_exists'] for event in events),
    interpretation='The executed source resets lost/stall to0 when committing a new arc, then its push branch can skip acquisition and issue the far contact target. Existing route() has no retained old-contact departure bridge. This audit identifies where that source behavior occurred; a revised policy outcome has not been tested.',
    proposed_revision='Retain old departure source transactionally; approach/route a newly selected target until actually acquired, then push. Add verified old-normal departure bridge if direct staging route is unavailable. Preserve original phases/actions/physics and finite predeclared cal-v2 scenes.')
write_json(OUT/'route-diagnosis.json', result)
print(json.dumps({key: value for key, value in result.items() if key != 'refresh_events'}))
