"""Record already completed work and actual callback; no model/physics/UI."""
import datetime
import hashlib
import json
import re
from pathlib import Path

d = Path(__file__).resolve().parent
repo = d.parent.parent
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt = json.loads((d / 'final-callback-result.json').read_text(encoding='utf-8'))
assert receipt['sent'] and not receipt['receipt']['isError']
audit = json.loads((d / 'final-readonly-audit.json').read_text(encoding='utf-8'))
browser = json.loads((d / 'browser-validation/browser-check.json').read_text(encoding='utf-8'))
decode = json.loads((d / 'browser-validation/video-decode-check.json').read_text(encoding='utf-8'))
assert browser['passed'] and decode['passed']
missing = []
for target in re.findall(r'\]\(([^)]+)\)', (d / 'REPORT.md').read_text(encoding='utf-8')):
    if '://' not in target and not (d / target).resolve().exists():
        missing.append(target)
assert not missing, missing
receipt['recorded_at_utc'] = now
receipt['message_sha256'] = hashlib.sha256((d / 'final-callback-message.txt').read_bytes()).hexdigest()
(d / 'final-callback-result.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
state_path = repo / 'artifacts/native-rgb-supervised-20261029/stage-state.json'
state = json.loads(state_path.read_text(encoding='utf-8'))
state.update(status='waiting_user_approval', training_authorized=False,
             updated_at_utc=now,
             next_action='Human review only. No further training, collection, model/sampling changes or remote work without explicit human approval.')
state['r3b_playground_diagnostics'] = {
    'status': 'complete_reported_waiting_user_approval',
    'artifact_root': str(d), 'report': str(d / 'REPORT.md'),
    'actual_callback_receipt': str(d / 'final-callback-result.json'),
    'default_physics': 'original-kinematic-damping0',
    'cross_physics_preview': True,
    'checkpoint_sha256': '23cc0f847fe585f894a7528bd5db1ea3c8541228c6de4396c82f11ce373ce5e7',
    'main_sampler_calls': 320, 'denoiser_forwards': 1536,
    'optimizer_updates': 0, 'recording_browser_rounds_passed': 2,
    'validation_service_pid': browser['final_health']['pid'],
    'validation_final_session': browser['final_session'],
    'remote_work': 0, 'training_authorized': False,
    'repository_source_check': 'failed: three existing published evidence JSON size/hash mismatches; no published evidence edited',
}
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
memo = repo / 'artifacts/native-rgb-supervised-20261029/SUPERVISION.md'
with memo.open('a', encoding='utf-8') as handle:
    handle.write('\n\n## Completed original-physics diagnosis/recording; actual callback sent (2026-10-30)\n\n'
        'Human-authorized first step complete: original physics six-letter parameter identity, T/C exact substep parity, '
        'continued O center/wall separation tests,40 paired clips/320 sampler calls/1536 forwards,0 optimizer updates. '
        'Old-trained7499 remains unchanged and labelled cross-physics. First-step rounding/holes and later H4 fragments/'
        'missing pushers confirmed; A corrected history also has exceptions,3/8/16 mixed. Six references same-class only, '
        'not exact old screenshot replay; final correspondence supersedes initial report snapshot without rewriting receipt.\n\n'
        'Actual Chrome two-round recording/reset/download/open/play plus saved WebM CPU decode/raw hashes passed; '
        'event-lock contention and lost video tail fixed minimally, earlier failures preserved. Original full-state '
        'unavailability and new internal solver-cache restore limitation remain. Validation ended PID24644 Ostep0 '
        'no active recording. No auto pushing. Reports/figures/raw/videos/source/cost audit in ' + str(d) + '. '
        'Repository source check separately fails existing custom-L/pretrained-T/palette-T summary JSON published-manifest '
        'hashes; no old evidence modified.\n\n'
        'ACTUAL send_message_to_thread succeeded to parent__LOCAL_CHAT__ (host local), '
        'including original epoch O/C improvements, all new limitations and real local paths. Exact message/real receipt: '
        + str(d / 'final-callback-message.txt') + ' / ' + str(d / 'final-callback-result.json') + '. '
        'STOP waiting_user_approval. Parent may only read/present; no next training/collection/model/sampling/remote release.\n')
result = {
    'completed_at_utc': now, 'status': 'waiting_user_approval', 'optimizer_updates': 0,
    'actual_callback_sent': True, 'callback_receipt': str(d / 'final-callback-result.json'),
    'report_local_links_valid': True, 'browser_passed': True, 'video_decode_passed': True,
    'source_check_passed': False,
    'source_check_failures': ['custom-L-summary.json size/hash mismatch', 'pretrained-T-summary.json size/hash mismatch', 'palette-T-summary.json size/hash mismatch'],
    'training_authorized': False,
}
(d / 'completion.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
