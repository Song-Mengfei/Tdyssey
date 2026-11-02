"""Finite, no-retry CPU collection supervisor; never imports physics or models.

Only launcher.py may start real work after a separately frozen formal release.
--check-only reads identities and prints evidence without spawning or writing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import (CHECKPOINT, KIND, LETTERS, PROTOCOL, actual_python_pe, atomic_json,
                    child_environment, exclusive_json, in_job, read_json, sha, utc,
                    storage_guard, validate_config, wait_launcher_exit)


def own_paths(config, label):
    root = Path(config['collection'])/'runs'/label
    return dict(root=root, ledger=root/'ledger.json', completion=root/'completion.json',
                identity=root/'identity.json', progress=root/'progress.json',
                stop=Path(config['artifact_root'])/'background'/'stop-requests'/(label+'.json'))


def selected_ids(config, by_id, phase, letters):
    frozen = config['formal_slot_ids'] if phase == 'formal' else [
        slot for letter in letters for slot in config['supplement_slot_ids'][letter]]
    return {slot for slot in frozen if by_id[slot]['letter'] in letters}


def truth(value):
    if value in (True, 'True', 'true', '1', 1):
        return True
    if value in (False, 'False', 'false', '0', 0):
        return False
    raise ValueError('Invalid boolean in saved CSV: ' + repr(value))


def weak_motion(csv_path, config, by_id=None):
    """All formal valid actual H1 controls, including first/final controls.

    Candidate windows, calibration and supplements never enter this denominator.
    """
    groups = {letter: [] for letter in LETTERS}
    seen = set()
    valid_per_trajectory = Counter()
    formal_ids = set(config['formal_slot_ids'])
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['phase_group'] != 'formal':
                continue
            if row['trajectory_id'] not in formal_ids:
                raise ValueError('Report includes an unbound formal trajectory')
            if by_id is not None and row['letter'] != by_id[row['trajectory_id']]['letter']:
                raise ValueError('Formal CSV letter differs from frozen scene plan')
            key = (row['trajectory_id'], int(row['control']))
            if not 0 <= key[1] < 64:
                raise ValueError('Formal control index outside original64-control trajectory')
            if key in seen:
                raise ValueError('Duplicate formal H1 control row')
            seen.add(key)
            if not truth(row['valid']):
                continue
            translation, rotation = float(row['translation_image_px']), float(row['rotation_deg'])
            if not math.isfinite(translation) or not math.isfinite(rotation) or translation < 0 or rotation < 0:
                raise ValueError('Invalid saved motion metric')
            groups[row['letter']].append((row['trajectory_id'], translation, rotation))
            valid_per_trajectory[row['trajectory_id']] += 1
    decisions = []
    for letter, rows in groups.items():
        slow = sum(translation < .5 for _, translation, _ in rows)
        rotation_ids = sorted({tid for tid, _, rotation in rows if rotation >= 5})
        translation_trigger = bool(rows) and slow/len(rows) > .5
        rotation_trigger = not rotation_ids
        decisions.append(dict(letter=letter, H1_valid_actual_controls=len(rows),
            H1_translation_below_half_image_px=slow,
            H1_translation_below_half_fraction=slow/len(rows) if rows else None,
            H1_rotation_at_least5deg_trajectory_ids=rotation_ids,
            translation_trigger=translation_trigger, rotation_trigger=rotation_trigger,
            triggered=translation_trigger or rotation_trigger,
            denominator='all formal valid actual H1 controls; no history/candidate filtering'))
    return dict(kind='original-contact-fixed-weak-motion-trigger.v1',
        csv_path=str(Path(csv_path).resolve()), csv_sha256=sha(csv_path),
        phases=['formal'], horizon_seconds=.1, decisions=decisions,
        valid_formal_controls_per_trajectory=dict(valid_per_trajectory),
        triggered_letters=''.join(row['letter'] for row in decisions if row['triggered']))


def merge_ledgers(config, by_id, ledger_paths, *, require_formal=False, supplement_letters=''):
    """Reject duplicate reservations and retain every physical attempt/cost."""
    attempts, seen, sources = [], set(), []
    total = dict(actual_controls=0, actual_substeps=0, actual_physics_seconds=0.)
    valid_controls = 0
    for path in ledger_paths:
        ledger = read_json(path)
        if ledger.get('optimizer_updates', 0) != 0 or ledger.get('model_calls', 0) != 0:
            raise ValueError('Unexpected training/model activity in ledger')
        ledger_controls = 0
        for row in ledger['attempts']:
            slot_id = row['slot_id']
            if slot_id in seen:
                raise ValueError('Duplicate slot reservation across source ledgers: ' + slot_id)
            seen.add(slot_id)
            if slot_id not in by_id:
                raise ValueError('Attempt outside original frozen plan: ' + slot_id)
            slot = by_id[slot_id]
            for key in ('phase', 'split', 'letter'):
                if row[key] != slot[key]:
                    raise ValueError('Attempt scene identity mismatch: ' + slot_id)
            if row.get('plan_sha256') != config['plan']['sha256'] or row.get('physics_protocol_sha256') != PROTOCOL:
                raise ValueError('Attempt plan/protocol identity mismatch: ' + slot_id)
            controls = row['actual_controls']
            if isinstance(controls, bool) or not isinstance(controls, int) or not 0 <= controls <= 64:
                raise ValueError('Invalid attempted-control count: ' + slot_id)
            metadata_path = Path(config['collection'])/'trajectories'/slot_id/'metadata.json'
            metadata = read_json(metadata_path)
            if metadata['slot'] != slot or metadata['reservation'] != row:
                raise ValueError('Metadata/ledger reservation mismatch: ' + slot_id)
            if metadata.get('protocol_sha256') != PROTOCOL or metadata.get('models') != 0 or metadata.get('optimizer_updates') != 0:
                raise ValueError('Invalid metadata physical/model identity: ' + slot_id)
            if metadata.get('contiguous_frames') is not True or metadata.get('hidden_replay_or_filled_frames') is not False or metadata.get('future_training_authorized') is not False:
                raise ValueError('Metadata continuity/no-fill/training restrictions changed: ' + slot_id)
            valid = metadata['actual_valid_controls']
            if not isinstance(valid, int) or isinstance(valid, bool) or not 0 <= valid <= controls:
                raise ValueError('Invalid valid-control count: ' + slot_id)
            if metadata['complete'] != (controls == 64 and valid == 64):
                raise ValueError('Complete flag is not64 valid actual controls: ' + slot_id)
            if metadata['complete'] != (row['status'] == 'completed'):
                raise ValueError('Complete status mismatch: ' + slot_id)
            if metadata['complete'] and metadata.get('failure') is not None:
                raise ValueError('Completed trajectory still reports a physical failure: ' + slot_id)
            raw = metadata.get('raw')
            if controls and (not raw or not (metadata_path.parent/'raw.npz').is_file()):
                raise ValueError('Actual controls missing durable raw evidence: ' + slot_id)
            if raw:
                raw_path = metadata_path.parent/'raw.npz'
                if raw.get('file') != 'raw.npz' or raw.get('bytes') != raw_path.stat().st_size or raw.get('sha256') != sha(raw_path):
                    raise ValueError('Durable raw size/SHA differs from metadata: ' + slot_id)
            ledger_controls += controls
            valid_controls += valid
            attempts.append(row)
        if ledger_controls != ledger['actual_controls']:
            raise ValueError('Ledger control total differs from its actual reservations: ' + str(path))
        for key in total:
            value = ledger[key]
            if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid cost: ' + key)
            if key != 'actual_physics_seconds' and not isinstance(value, int):
                raise ValueError('Integer count required: ' + key)
            total[key] += value
        if ledger['actual_substeps'] > ledger_controls*200 or abs(ledger['actual_physics_seconds']-ledger['actual_substeps']*.0005) > 1e-6:
            raise ValueError('Actual substeps/seconds exceed fixed v6 physical protocol')
        sources.append(dict(path=str(Path(path).resolve()), sha256=sha(path), attempts=len(ledger['attempts']),
                            actual_controls=ledger['actual_controls'], actual_substeps=ledger['actual_substeps'],
                            actual_physics_seconds=ledger['actual_physics_seconds']))
    formal_attempts = [row for row in attempts if row['phase'] == 'formal']
    formal_ids = {row['slot_id'] for row in formal_attempts}
    if require_formal and (formal_ids != set(config['formal_slot_ids']) or any(row['status'] != 'completed' for row in formal_attempts)):
        raise ValueError('Formal360 complete64-valid-control attempts are not present; no supplement release')
    if supplement_letters:
        expected = {slot for letter in supplement_letters for slot in config['supplement_slot_ids'][letter]}
        actual = {row['slot_id'] for row in attempts if row['phase'] == 'supplement'}
        if expected != actual or any(row['status'] != 'completed' for row in attempts if row['phase'] == 'supplement'):
            raise ValueError('Bounded supplement has incomplete/missing attempts')
    return dict(kind='original-contact-pilot-actual-cost.v1', attempts=attempts, **total,
                actual_valid_controls=valid_controls, optimizer_updates=0, model_calls=0,
                merged_sources=sources, merged_at_utc=utc(),
                phase_attempts=dict(Counter(row['phase'] for row in attempts)),
                phase_complete=dict(Counter(row['phase'] for row in attempts if row['status'] == 'completed')))


class Supervisor:
    def __init__(self, config_path, expected_sha, launcher_pid):
        self.config_path, self.expected_sha = Path(config_path), expected_sha
        self.config, self.paths, self.by_id = validate_config(config_path, expected_sha, require_release=True)
        self.background = Path(self.config['artifact_root'])/'background'
        self.launcher_pid = launcher_pid
        self.children = []
        self.ledger_paths = [self.paths['calibration_ledger']]
        self.state = 'preflight'
        self.started = utc()
        self.started_monotonic = time.monotonic()
        self.next_heartbeat = 0
        self.failure = None

    def snapshot(self, *, force=False):
        if not force and time.monotonic() < self.next_heartbeat:
            return
        self.next_heartbeat = time.monotonic()+30
        children = []
        for child in self.children:
            paths = child['paths']
            children.append(dict(label=child['label'], phase=child['phase'], letters=child['letters'],
                pid=child['process'].pid, exit_code=child['process'].poll(),
                progress=read_json(paths['progress']) if paths['progress'].is_file() else None,
                stop_requested=paths['stop'].is_file()))
        record = dict(kind=KIND, run_id=self.config['run_id'], config_sha256=self.expected_sha,
            state=self.state, pid=os.getpid(), updated_at_utc=utc(), started_at_utc=self.started,
            elapsed_wall_seconds=time.monotonic()-self.started_monotonic, children=children,
            failure=self.failure, model_calls=0, optimizer_updates=0, auto_retry=False)
        atomic_json(self.background/'status.json', record)
        atomic_json(self.background/'heartbeat.json', record)

    def revalidate(self):
        validate_config(self.config_path, self.expected_sha, require_release=True)

    def request_stop(self, reason):
        for child in self.children:
            if child['process'].poll() is None and not child['paths']['stop'].exists():
                exclusive_json(child['paths']['stop'], dict(run_id=self.config['run_id'],
                    run_label=child['label'], reason=reason, requested_at_utc=utc(),
                    contract='stop only at next trajectory boundary; no retries; preserve every attempted control'))

    def start_collectors(self, phase, released_letters):
        self.revalidate()
        postprocess=read_json(self.paths['resource_policy'])['postprocess_reserve_bytes']
        atomic_json(self.background/(phase+'-storage-preflight.json'),storage_guard(self.config,extra_bytes=postprocess))
        self.state = 'collecting-'+phase
        for shard in self.config['shards']:
            letters = ''.join(letter for letter in shard['letters'] if letter in released_letters)
            if not letters:
                continue
            label = shard[phase+'_label']
            paths = own_paths(self.config, label)
            if paths['root'].exists() or paths['stop'].exists():
                raise RuntimeError('Run label/stop request already exists; no automatic retry: ' + label)
            expected = selected_ids(self.config, self.by_id, phase, letters)
            paths['root'].mkdir(parents=True, exist_ok=False)
            command = [actual_python_pe(), '-X', 'utf8', '-B', str(self.paths['collector_script']), 'collect',
                '--output', self.config['collection'], '--plan', str(self.paths['plan']), '--phase', phase,
                '--physics-gate', str(self.paths['physics_gate']), '--source-root', self.config['source_root'],
                '--storage-roots-json', str(self.paths['storage_roots']),
                '--policy-revision', str(self.paths[phase+'_policy']), '--resource-policy', str(self.paths['resource_policy']),
                '--letters', letters, '--run-label', label, '--stop-file', str(paths['stop'])]
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
            with (paths['root']/'stdout.log').open('ab') as out, (paths['root']/'stderr.log').open('ab') as err:
                process = subprocess.Popen(command, cwd=self.config['repo'], env=child_environment(self.config),
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, close_fds=True, startupinfo=startup,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
            child = dict(label=label, paths=paths, expected=expected, process=process, phase=phase, letters=letters)
            self.children.append(child)
            job = in_job(int(process._handle))
            atomic_json(paths['root']/'child-process.json', dict(pid=process.pid, run_id=self.config['run_id'],
                run_label=label, command=command, is_process_in_job=job, started_at_utc=utc(),
                interpreter_pe=command[0], venv_launcher=self.config['python_executable']))
            if job:
                raise RuntimeError('Collector unexpectedly belongs to a Windows Job; boundary-stop requested')
        self.snapshot(force=True)

    def inspect_child_failure(self, child):
        paths = child['paths']
        if paths['ledger'].exists():
            ledger = read_json(paths['ledger'])
            invalid = [row for row in ledger['attempts'] if row['status'] not in ('reserved', 'completed')]
            if invalid:
                return 'Substantive physical failure '+child['label']+': '+invalid[0]['slot_id']
        if paths['completion'].exists():
            completion = read_json(paths['completion'])
            if completion.get('status') == 'failed' or completion.get('graceful_stop') or completion.get('completed') is False:
                return 'Collector failed/stopped: '+child['label']
        if child['process'].poll() not in (None, 0):
            return 'Collector CLI exit '+str(child['process'].returncode)+': '+child['label']
        return None

    def wait_children(self):
        while any(child['process'].poll() is None for child in self.children):
            if (self.background/'STOP.json').exists():
                self.failure = self.failure or 'Explicit coordinator boundary-stop request'
            for child in self.children:
                self.failure = self.failure or self.inspect_child_failure(child)
            if self.failure:
                self.state = 'stopping-at-trajectory-boundary'
                self.request_stop(self.failure)
            self.snapshot()
            time.sleep(1)
        for child in self.children:
            self.failure = self.failure or self.inspect_child_failure(child)
        self.snapshot(force=True)
        if self.failure:
            raise RuntimeError(self.failure)

    def verify_phase(self, phase):
        for child in [child for child in self.children if child['phase'] == phase]:
            paths = child['paths']
            if child['process'].returncode != 0 or not all(paths[key].is_file() for key in ('ledger', 'completion', 'identity')):
                raise RuntimeError('Collector CLI exited without complete durable receipts: '+child['label'])
            completion, identity, ledger = map(read_json, (paths['completion'], paths['identity'], paths['ledger']))
            for record in (completion, identity):
                if record.get('run_label') != child['label'] or record.get('phase') != phase:
                    raise ValueError('Shard identity mismatch: '+child['label'])
                if record.get('plan_sha256') != self.config['plan']['sha256']:
                    raise ValueError('Shard plan mismatch: '+child['label'])
                if record.get('physics_protocol_sha256') != PROTOCOL:
                    raise ValueError('Shard protocol mismatch: '+child['label'])
            if completion.get('completed') is not True or completion.get('graceful_stop', False):
                raise ValueError('Shard did not attempt its complete finite slot set')
            for record in (completion, identity):
                if record.get('optimizer_updates', 0) != 0 or record.get('model_calls', record.get('models', 0)) != 0:
                    raise ValueError('Shard unexpectedly called a model or optimizer')
                if record.get('physics_gate_sha256') != self.config['physics_gate']['sha256']:
                    raise ValueError('Shard physics gate identity mismatch')
                if record.get('resource_policy_sha256') != self.config['resource_policy']['sha256']:
                    raise ValueError('Shard resource policy identity mismatch')
                if record.get('policy_revision_sha256') != self.config[phase+'_policy']['sha256']:
                    raise ValueError('Shard strategy file identity mismatch')
            if {row['slot_id'] for row in ledger['attempts']} != child['expected'] or len(ledger['attempts']) != len(child['expected']):
                raise ValueError('Shard attempted slot denominator differs from frozen release')
            if any(row['status'] != 'completed' for row in ledger['attempts']):
                raise ValueError('Shard has a substantive physical rejection; never retry')
            self.ledger_paths.append(paths['ledger'])

    def write_merged(self, *, require_formal, supplement_letters=''):
        ledger = merge_ledgers(self.config, self.by_id, self.ledger_paths,
                               require_formal=require_formal, supplement_letters=supplement_letters)
        atomic_json(Path(self.config['collection'])/'ledger.json', ledger)
        atomic_json(self.background/'merged-ledger-proof.json', dict(
            run_id=self.config['run_id'], ledger_path=str(Path(self.config['collection'])/'ledger.json'),
            ledger_sha256=sha(Path(self.config['collection'])/'ledger.json'),
            phase_attempts=ledger['phase_attempts'], phase_complete=ledger['phase_complete'],
            actual_controls=ledger['actual_controls'], actual_valid_controls=ledger['actual_valid_controls'],
            actual_substeps=ledger['actual_substeps'], actual_physics_seconds=ledger['actual_physics_seconds'],
            sources=ledger['merged_sources'], optimizer_updates=0, model_calls=0))
        return ledger

    def saved_command(self, command, label):
        self.revalidate()
        # A finite video budget is reserved before generation. The saved-report
        # CLI also applies the frozen policy before each expensive output.
        extra=read_json(self.paths['resource_policy'])['postprocess_reserve_bytes'] if label.endswith('report') else 0
        atomic_json(self.background/(label+'-storage-preflight.json'),storage_guard(self.config,extra_bytes=extra))
        self.state = 'saved-only-'+label
        with (self.background/(label+'-stdout.log')).open('ab') as out, (self.background/(label+'-stderr.log')).open('ab') as err:
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
            process = subprocess.Popen([actual_python_pe(), '-X', 'utf8', '-B', *command],
                cwd=self.config['repo'], env=child_environment(self.config), stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, close_fds=True, startupinfo=startup,
                creationflags=subprocess.CREATE_NO_WINDOW)
            while process.poll() is None:
                self.snapshot()
                time.sleep(1)
            if process.returncode:
                raise RuntimeError('Saved-only '+label+' CLI failed: '+str(process.returncode))
        atomic_json(self.background/(label+'-storage-postflight.json'),storage_guard(self.config))

    def manifest(self):
        atomic_json(self.background/'manifest-storage-preflight.json',storage_guard(self.config))
        collection = Path(self.config['collection'])
        required = [self.config_path, self.paths['plan'], self.paths['physics_gate'], self.paths['resource_policy'],
            self.paths['formal_policy'], self.paths['supplement_policy'], collection/'ledger.json',
            collection/'REPORT.md', collection/'report/summary.json',
            collection/'report/control-metrics.csv', self.background/'weak-motion-trigger.json',
            Path(self.config['artifact_root'])/'REPORT.md', Path(self.config['artifact_root'])/'index.html',
            Path(self.config['artifact_root'])/'overview-link-check.json']
        missing=[str(path) for path in required if not Path(path).is_file()]
        if missing:
            raise ValueError('Required final saved evidence missing: '+str(missing))
        # Raw files stay in place. Hash every actual attempted raw/metadata/claim;
        # read-only SHA streaming does not load a model or run physics.
        for row in read_json(collection/'ledger.json')['attempts']:
            directory = collection/'trajectories'/row['slot_id']
            required += [directory/'claim.json', directory/'metadata.json', directory/'raw.npz']
        required += [Path(item['path']) for item in self.config['source_bindings']]
        required += [Path(self.config[key]['path']) for key in ('calibration_ledger','collector_script','overview_script','notification')]
        required += [self.background/name for name in ('worker-startup.json','launcher-receipt.json','merged-ledger-proof.json')]
        required += list((collection/'report').rglob('*'))
        for child in self.children:
            required += list(child['paths']['root'].rglob('*'))
        # All selected videos, their decode receipts, figures and metric CSVs are
        # included in place, not merely the report entry point.
        unique = {str(Path(path).resolve()).casefold():Path(path).resolve() for path in required if Path(path).is_file()}
        entries = [dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)) for path in unique.values()]
        atomic_json(self.background/'final-manifest.json', dict(run_id=self.config['run_id'], generated_at_utc=utc(),
            checkpoint=dict(path=str(self.paths['checkpoint']), sha256=sha(self.paths['checkpoint']), unchanged=True),
            files=entries, model_calls=0, optimizer_updates=0, physics_replay_steps=0))
        atomic_json(self.background/'manifest-storage-postflight.json',storage_guard(self.config))

    def failure_manifest(self):
        paths = [self.config_path, self.background/'FAILED.md', self.background/'failure-evidence.json',
                 self.background/'worker-startup.json', self.background/'launcher-receipt.json', *self.ledger_paths]
        for source in self.ledger_paths:
            try:
                for row in read_json(source)['attempts']:
                    directory=Path(self.config['collection'])/'trajectories'/row['slot_id']
                    paths += [directory/'claim.json',directory/'metadata.json',directory/'raw.npz']
            except Exception:
                pass  # The unparseable original is hashed, not silently repaired.
        for child in self.children:
            paths += list(child['paths']['root'].rglob('*'))
        unique={str(Path(path).resolve()).casefold():Path(path).resolve() for path in paths if Path(path).is_file()}
        atomic_json(self.background/'failure-manifest.json',dict(run_id=self.config['run_id'],
            config_sha256=self.expected_sha, generated_at_utc=utc(), status='failed',
            files=[dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for path in unique.values()],
            model_calls=0,optimizer_updates=0,physics_replay_steps=0))

    def finish(self, success, ledger=None):
        self.state = 'waiting_app_callback' if success else 'failed_waiting_app_callback'
        self.snapshot(force=True)
        record = dict(kind=KIND, run_id=self.config['run_id'], config_sha256=self.expected_sha,
            status='success' if success else 'failed', finished_at_utc=utc(), failure=self.failure,
            totals_unresolved=ledger is None,
            formal_target_attempts=360, formal_target_controls=23040,
            phase_attempts=ledger.get('phase_attempts') if ledger else None,
            phase_complete=ledger.get('phase_complete') if ledger else None,
            actual_controls=ledger.get('actual_controls') if ledger else None,
            actual_valid_controls=ledger.get('actual_valid_controls') if ledger else None,
            actual_substeps=ledger.get('actual_substeps') if ledger else None,
            actual_physics_seconds=ledger.get('actual_physics_seconds') if ledger else None,
            REPORT=str(Path(self.config['artifact_root'])/'REPORT.md') if success else None,
            FAILED=str(self.background/'FAILED.md') if not success else None,
            collection_REPORT=str(Path(self.config['collection'])/'REPORT.md') if success else None,
            manifest=str(self.background/('final-manifest.json' if success else 'failure-manifest.json')),
            cost_ledger_paths=[str(path) for path in self.ledger_paths],
            checkpoint_sha256=sha(self.paths['checkpoint']), checkpoint_expected_sha256=CHECKPOINT,
            no_auto_retry=True, model_calls=0, optimizer_updates=0, training_authorized=False,
            callback=dict(sent=False, performed=False, required_actor='Codex app MCP / official heartbeat',
                          reason='Local worker cannot perform MCP callback while Codex is closed'),
            limitations=['Physical candidate data only; no model improvement was measured.',
                'Historical7499 checkpoint remains trained on old physics.',
                'Completion does not authorize training, more epochs, more collection or server changes.'])
        atomic_json(self.background/'completion.json', record)

    def run(self):
        exclusive_json(self.background/'worker.lock.json', dict(run_id=self.config['run_id'], config_sha256=self.expected_sha,
            pid=os.getpid(), started_at_utc=self.started, no_automatic_recovery=True))
        ledger = None
        try:
            if self.launcher_pid <= 0 or self.launcher_pid == os.getpid():
                raise RuntimeError('Independent launcher parent PID is required')
            job = in_job()
            startup = dict(run_id=self.config['run_id'], config_sha256=self.expected_sha, pid=os.getpid(),
                observed_parent_pid=os.getppid(), launcher_pid=self.launcher_pid,
                is_process_in_job=job, interpreter_pe=actual_python_pe(), sys_executable=sys.executable,
                sys_prefix=sys.prefix, venv_launcher=os.environ.get('__PYVENV_LAUNCHER__'), at_utc=utc())
            if job:
                atomic_json(self.background/'worker-startup.json', startup)
                raise RuntimeError('Worker is in a Windows Job; refusing first control')
            if Path(sys.prefix).resolve() != Path(self.config['python_executable']).parent.parent.resolve():
                raise RuntimeError('Worker did not activate the expected existing venv')
            startup['launcher_exit'] = wait_launcher_exit(self.launcher_pid)
            startup['controls_before_parent_exit'] = 0
            atomic_json(self.background/'worker-startup.json', startup)
            self.revalidate()
            for slot in self.config['formal_slot_ids']:
                if (Path(self.config['collection'])/'trajectories'/slot/'claim.json').exists():
                    raise RuntimeError('Existing formal claim needs owner audit; no resume/retry: '+slot)
            self.start_collectors('formal', LETTERS)
            self.wait_children()
            self.verify_phase('formal')
            ledger = self.write_merged(require_formal=True)
            self.saved_command([str(self.paths['collector_script']), 'report', '--output', self.config['collection'], '--videos',
                '--resource-policy',str(self.paths['resource_policy']),'--storage-roots-json',str(self.paths['storage_roots'])], 'formal-report')
            decision = weak_motion(Path(self.config['collection'])/'report/control-metrics.csv', self.config, self.by_id)
            if sum(row['H1_valid_actual_controls'] for row in decision['decisions']) != sum(
                    row['actual_controls'] for row in ledger['attempts'] if row['phase'] == 'formal'):
                raise ValueError('Formal CSV valid-control denominator disagrees with all360 completed trajectories')
            if decision['valid_formal_controls_per_trajectory'] != {slot:64 for slot in self.config['formal_slot_ids']}:
                raise ValueError('Formal CSV must retain all64 valid actual H1 controls per independent trajectory')
            if any(row['H1_valid_actual_controls'] != 3840 for row in decision['decisions']):
                raise ValueError('Each formal letter must retain60 independent trajectories x64 valid actual H1 controls')
            atomic_json(self.background/'weak-motion-trigger.json', decision)
            triggered = decision['triggered_letters']
            if triggered:
                release = dict(released=True, run_id=self.config['run_id'], config_sha256=self.expected_sha,
                    released_at_utc=utc(), letters=triggered, maximum_attempts_per_letter=12,
                    slot_ids={letter:self.config['supplement_slot_ids'][letter] for letter in triggered},
                    policy=self.config['supplement_policy'], trigger=decision,
                    no_fill_or_retry=True, training_authorized=False)
                exclusive_json(Path(self.config['collection'])/'supplement-release.json', release)
                for letter in triggered:
                    for slot in self.config['supplement_slot_ids'][letter]:
                        if (Path(self.config['collection'])/'trajectories'/slot/'claim.json').exists():
                            raise RuntimeError('Supplement claim already exists; no automatic retry')
                self.start_collectors('supplement', triggered)
                self.wait_children()
                self.verify_phase('supplement')
                ledger = self.write_merged(require_formal=True, supplement_letters=triggered)
                self.saved_command([str(self.paths['collector_script']), 'report', '--output', self.config['collection'], '--videos',
                    '--resource-policy',str(self.paths['resource_policy']),'--storage-roots-json',str(self.paths['storage_roots'])], 'final-report')
            self.saved_command([str(self.paths['overview_script']), '--output', self.config['artifact_root']], 'overview')
            self.revalidate()
            self.manifest()
            self.finish(True, ledger)
            return 0
        except Exception as error:
            self.failure = self.failure or repr(error)
            self.state = 'stopping-at-trajectory-boundary'
            self.request_stop(self.failure)
            # These are only owned collection processes. Never terminate or restart them.
            # Keep a heartbeat while their explicit stop-file reaches a trajectory boundary.
            while any(child['process'].poll() is None for child in self.children):
                self.snapshot()
                time.sleep(1)
            self.ledger_paths = list(dict.fromkeys([self.paths['calibration_ledger']]+[
                child['paths']['ledger'] for child in self.children if child['paths']['ledger'].is_file()]))
            evidence = dict(run_id=self.config['run_id'], config_sha256=self.expected_sha,
                failure=self.failure, traceback=traceback.format_exc(), at_utc=utc(),
                sources=[],
                children=[dict(run_label=child['label'], pid=child['process'].pid,
                               exit_code=child['process'].returncode) for child in self.children],
                model_calls=0, optimizer_updates=0, no_auto_retry=True)
            for path in self.ledger_paths:
                source=dict(path=str(path),sha256=sha(path))
                try:
                    original=read_json(path)
                    source['reported_actual_costs_unverified']=dict(
                        actual_controls=original.get('actual_controls'),actual_substeps=original.get('actual_substeps'),
                        actual_physics_seconds=original.get('actual_physics_seconds'),attempts=len(original.get('attempts',[])))
                except Exception as read_error:
                    source['read_error']=repr(read_error)
                evidence['sources'].append(source)
            atomic_json(self.background/'failure-evidence.json', evidence)
            try:
                ledger = self.write_merged(require_formal=False)
            except Exception as merge_error:
                ledger = None
                evidence['merge_error'] = repr(merge_error)
                evidence['totals_unresolved'] = True
                atomic_json(self.background/'failure-evidence.json', evidence)
            (self.background/'FAILED.md').write_text(
                '# Background collection stopped\n\n'+self.failure+'\n\n'
                'All owned collectors exited after a trajectory-boundary stop request or their own failure. '
                'No retry, model inference, optimizer update or additional release was performed. '
                'Actual costs and failures remain in the referenced immutable shard ledgers/raw data. '
                'A partial formal run is not a model result.\n', encoding='utf-8')
            self.failure_manifest()
            self.finish(False, ledger)
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--launcher-pid', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        config, paths, by_id = validate_config(args.config, args.config_sha, require_release=False)
        print(json.dumps(dict(passed=True, run_id=config['run_id'], config_sha256=args.config_sha,
            released=config.get('released'), planned_formal_attempts=360,
            possible_supplement_attempts=72, original_plan_slots=len(by_id),
            source_bindings=len(config['source_bindings']), actual_physics_steps=0,
            model_calls=0, optimizer_updates=0, processes_started=0, files_written=0), ensure_ascii=False))
        return 0
    if args.launcher_pid is None:
        parser.error('Real work requires the independent launcher; use --check-only for read-only checks')
    return Supervisor(args.config, args.config_sha, args.launcher_pid).run()


if __name__ == '__main__':
    raise SystemExit(main())
