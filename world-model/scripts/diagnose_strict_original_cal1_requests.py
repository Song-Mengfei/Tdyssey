"""Saved-only diagnosis of initial-entry versus staging backtracking."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tdyssey.original_contact_pilot import load_attempt, file_sha, write_json, utc
from tdyssey.native_rgb_r2b_sampling import contact_frame


def main():
    output = ROOT/'artifacts/strict-original-contact-pilot-20261031'
    rows = []
    for path in sorted((output/'collection/trajectories').glob('*/metadata.json')):
        meta = json.loads(path.read_text())
        if meta['slot']['phase'] != 'calibration-v1':
            continue
        raw = load_attempt(path.parent/'raw.npz')
        reversals = []
        entry_indices = []
        positive_seconds = 0.
        for index, (decision, telemetry) in enumerate(zip(raw['decisions'], raw['telemetry'])):
            if decision['mode'] == 'contact_entry_intent':
                entry_indices.append(index)
            if index and raw['decisions'][index-1]['mode'] == 'contact_entry_intent' and decision['mode'] == 'free_route':
                state = raw['states'][index]
                contact = dict(point_local=decision['selected_point_local'],
                    outward_normal_local=decision['selected_normal_local'])
                point, normal, _ = contact_frame(state, contact)
                inward_advance = float((raw['requested_actions'][index]-state[:2])@(-normal))
                if inward_advance < -1e-8:
                    reversals.append(dict(control=index, previous_control=index-1,
                        same_selected_arc=decision['selected_arc_id'] == raw['decisions'][index-1]['selected_arc_id'],
                        requested_inward_advance_world_px=inward_advance,
                        prior_state=state.tolist(), action=raw['requested_actions'][index].tolist(),
                        prior_entry_action=raw['requested_actions'][index-1].tolist(),
                        reason=decision.get('target_refresh_reason'),route_kind=decision.get('route_kind')))
            for row in telemetry['substeps']:
                if row.get('executed') and any(event.get('kind') == 'agent_object' and event.get('phase') == 'post_solve'
                        and np.linalg.norm(event.get('total_impulse', [0., 0.])) > 1e-8 for event in row.get('contacts', [])):
                    positive_seconds += float(row['dt'])
        rows.append(dict(slot_id=meta['slot']['slot_id'], letter=meta['slot']['letter'],
            scenario=meta['slot']['scenario'], location=meta['slot']['initial_space'],
            raw_file=str((path.parent/'raw.npz').resolve()), raw_sha256=file_sha(path.parent/'raw.npz'),
            positive_AO_seconds=positive_seconds, complete=meta['complete'],
            modes=dict(Counter(d['mode'] for d in raw['decisions'])), entry_controls=entry_indices,
            entry_immediately_reversed_controls=reversals,
            all_frames_object_state_exact_initial=bool(np.array_equal(raw['states'][:, 2:5],
                np.tile(raw['states'][0, 2:5], (len(raw['states']), 1))))))
    letter_rows = {}
    for letter in 'TLOCXB':
        subset = [r for r in rows if r['letter'] == letter]
        letter_rows[letter] = dict(attempts=len(subset), positive_AO_seconds=sum(r['positive_AO_seconds'] for r in subset),
            exact_static_attempts=sum(r['all_frames_object_state_exact_initial'] for r in subset),
            entry_to_outward_route_reversals=sum(len(r['entry_immediately_reversed_controls']) for r in subset),
            same_arc_reversals=sum(v['same_selected_arc'] for r in subset for v in r['entry_immediately_reversed_controls']))
    result = dict(kind='strict-original-calibration-v1-saved-request-diagnosis.v1',created_at_utc=utc(),
        root_cause='Missing contact-entry latch: after one entry request the slower original PD pusher has left staging by >4worldpx but remains >8worldpx from contactcenter; inherited branch requests a return to staging, creating a repeat loop before contact.',
        letter_summary=letter_rows, attempts=rows, actual_new_physical_controls=0, model_calls=0,
        optimizer_updates=0, physics_changed=False, source_inputs='Actual saved raw arrays and decisions only')
    destination = output/'validation/calibration-v1-request-diagnosis.json'
    write_json(destination, result)
    print(json.dumps(dict(report=str(destination.resolve()), letter_summary=letter_rows), indent=2))


if __name__ == '__main__':
    main()
