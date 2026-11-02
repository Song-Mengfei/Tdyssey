"""Shared storage reservations for candidate collection and saved-only reports.

No model, simulator, network or data-format changes. An OS lock serializes the
shared snapshot/check/reservation transaction across independent processes.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

SHARED_CAP = 3*1024**3
ATOMIC_RESERVE = 256*1024**2
DEFAULT_ATTEMPT = 16*1024**2
POSTPROCESS_RESERVE = 128*1024**2
ADMIN_WRITE_MARGIN = 1024**2
_THREAD_LOCKS = {}


class ResourceLimitError(RuntimeError):
    """No output/physical step should begin after this failed reservation."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_resource_policy(path):
    if path is None:
        return dict(kind='original-contact-pilot-default-storage-policy.v1',
                    collection_cap_bytes=512*1024**2, shared_cap_bytes=SHARED_CAP,
                    atomic_reserve_bytes=ATOMIC_RESERVE,
                    per_attempt_atomic_reserve_bytes=DEFAULT_ATTEMPT,
                    maximum_concurrent_collectors=2,
                    concurrent_attempt_reserve_bytes=2*DEFAULT_ATTEMPT,
                    postprocess_reserve_bytes=POSTPROCESS_RESERVE,
                    training_authorized=False), None
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if value.get('shared_cap_bytes') != SHARED_CAP or value.get('atomic_reserve_bytes') != ATOMIC_RESERVE:
        raise ValueError('The human shared3GiB/256MiB atomic limits must remain unchanged')
    if value.get('postprocess_reserve_bytes') != POSTPROCESS_RESERVE:
        raise ValueError('Exactly128MiB postprocessing allocation is required')
    if value.get('maximum_concurrent_collectors') != 2:
        raise ValueError('Exactly two maximum concurrent collectors are declared')
    if value.get('per_attempt_atomic_reserve_bytes') != DEFAULT_ATTEMPT or value.get('concurrent_attempt_reserve_bytes') != 2*DEFAULT_ATTEMPT:
        raise ValueError('Two16MiB concurrent attempt allocations must be retained')
    cap = value.get('collection_cap_bytes')
    if type(cap) is not int or not DEFAULT_ATTEMPT < cap < SHARED_CAP-ATOMIC_RESERVE:
        raise ValueError('Invalid per-collection allocation')
    if value.get('training_authorized') is not False:
        raise ValueError('Resource allocation never authorizes training')
    return value, hashlib.sha256(raw).hexdigest()


@contextmanager
def _process_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    local = _THREAD_LOCKS.setdefault(str(path.resolve()).casefold(), threading.Lock())
    with local:
        with path.open('a+b') as handle:
            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(.05)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_json_atomic(path, value):
    temporary = Path(path).with_suffix('.json.tmp')
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
    with temporary.open('wb') as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class ResourceReservation:
    """One bounded writer lease. Stale leases are never silently discarded.

    Collection retains its allocation until the whole trajectory has been saved.
    Postprocessing consumes at most128MiB net growth; each atomic output must
    also fit its remaining peak budget. A failed process leaves a conservative
    lease for owner audit rather than pretending its pending writes disappeared.
    """
    def __init__(self, output, shared_roots, resource_policy_path, owner, *,
                 role='collection', reserve_bytes=None, _snapshot_fn=None, _disk_free_fn=None):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.shared_roots = [Path(path).resolve() for path in shared_roots]
        self.resource_policy_path = (Path(resource_policy_path).resolve() if resource_policy_path is not None else None)
        self.policy, self.policy_sha256 = load_resource_policy(self.resource_policy_path)
        if role not in ('collection', 'postprocess'):
            raise ValueError('Unknown reservation role')
        self.role, self.owner = role, str(owner)
        if not self.owner:
            raise ValueError('A readable owner identity is required')
        self.initial_budget = int(reserve_bytes if reserve_bytes is not None else
                                  DEFAULT_ATTEMPT if role == 'collection' else POSTPROCESS_RESERVE)
        if self.initial_budget <= 0 or (role == 'postprocess' and self.initial_budget != POSTPROCESS_RESERVE):
            raise ValueError('Invalid writer allocation')
        self.reservation_id = uuid.uuid4().hex
        self.lock_path = self.output/'.resource.lock'
        self.ledger_path = self.output/'resource-reservations.json'
        self._snapshot_fn = _snapshot_fn
        self._disk_free_fn = _disk_free_fn
        self.entered = False
        self.last_snapshot = None

    def _ledger(self):
        if not self.ledger_path.exists():
            return dict(kind='original-contact-pilot-shared-writer-reservations.v1',
                        resource_policy_sha256=self.policy_sha256, leases=[])
        value = json.loads(self.ledger_path.read_text(encoding='utf-8'))
        active = [row for row in value['leases'] if row['status'] == 'active']
        if value['resource_policy_sha256'] != self.policy_sha256:
            if active:
                raise ResourceLimitError('Active writer uses another resource policy; no silent lease reuse')
            value.setdefault('historical_resource_policy_sha256', []).append(value['resource_policy_sha256'])
            value['resource_policy_sha256'] = self.policy_sha256
        return value

    def _scan(self, roots):
        if self._snapshot_fn is not None:
            return self._snapshot_fn(roots)
        from .original_contact_pilot import storage_snapshot
        return storage_snapshot(roots)

    def _check(self, ledger):
        active = [row for row in ledger['leases'] if row['status'] == 'active']
        collectors = [row for row in active if row['role'] == 'collection']
        reporters = [row for row in active if row['role'] == 'postprocess']
        if len(collectors) > self.policy['maximum_concurrent_collectors'] or len(reporters) > 1 or collectors and reporters:
            raise ResourceLimitError('Shared writers exceed the declared collection/report concurrency')
        outstanding = sum(row['remaining_bytes'] for row in active)
        # The first collector must leave room for its second independent peer,
        # and every collection boundary preserves the separate report budget.
        collector_floor = self.policy['concurrent_attempt_reserve_bytes'] if collectors else 0
        writer_reserve = max(outstanding, collector_floor)
        postprocess_hold = 0 if reporters else self.policy['postprocess_reserve_bytes']
        collection = self._scan([self.output])
        shared = self._scan([*self.shared_roots, self.output])
        free = (int(self._disk_free_fn()) if self._disk_free_fn is not None else
                shutil.disk_usage(self.output).free)
        reserved = self.policy['atomic_reserve_bytes']+writer_reserve+postprocess_hold
        # The unchanged global hardcap is authoritative before the local cap.
        if shared['bytes']+reserved > self.policy['shared_cap_bytes'] or free < reserved:
            raise ResourceLimitError('Shared3GiB/256MiB atomic/report/writer reservation is unavailable')
        if collection['bytes']+writer_reserve+postprocess_hold > self.policy['collection_cap_bytes']:
            raise ResourceLimitError('Per-collection allocation including pending/report writes is unavailable')
        return dict(collection=collection, shared=shared, disk_free_bytes=free,
                    active_writer_reserve_bytes=writer_reserve,
                    postprocess_hold_bytes=postprocess_hold, reserved_bytes=reserved,
                    resource_policy_sha256=self.policy_sha256)

    def __enter__(self):
        if self.entered:
            raise RuntimeError('Writer reservation cannot be entered twice')
        with _process_lock(self.lock_path):
            ledger = self._ledger()
            preexisting_outputs = (self._scan([self.output/'report', self.output/'REPORT.md'])['bytes']
                                   if self.role == 'postprocess' else 0)
            if preexisting_outputs > self.initial_budget:
                raise ResourceLimitError('Existing report outputs already exceed128MiB; preserve them and stop')
            row = dict(reservation_id=self.reservation_id, owner=self.owner, pid=os.getpid(),
                       role=self.role, status='active', started_at_utc=utc(),
                       allocated_bytes=self.initial_budget,
                       preexisting_managed_output_bytes=preexisting_outputs,
                       remaining_bytes=self.initial_budget-preexisting_outputs,
                       committed_net_growth_bytes=0, writes=0)
            ledger['leases'].append(row)
            self.last_snapshot = self._check(ledger)
            _write_json_atomic(self.ledger_path, ledger)
            self.entered = True
        return self

    def _row(self, ledger):
        row = next((row for row in ledger['leases'] if row['reservation_id'] == self.reservation_id), None)
        if row is None or row['status'] != 'active':
            raise RuntimeError('Writer does not hold an active shared reservation')
        return row

    def ensure_capacity(self, byte_count):
        if not self.entered or type(byte_count) is not int or byte_count < 0:
            raise ValueError('Active reservation and nonnegative exact bytes required')
        with _process_lock(self.lock_path):
            ledger = self._ledger()
            row = self._row(ledger)
            needed = byte_count+(ADMIN_WRITE_MARGIN if self.role == 'collection' else 0)
            expanded = needed > row['remaining_bytes']
            if expanded:
                if self.role == 'postprocess':
                    raise ResourceLimitError('An atomic report output exceeds the remaining128MiB report budget')
                delta = needed-row['remaining_bytes']
                row['remaining_bytes'] += delta
                row['allocated_bytes'] += delta
            # A collection lease already owns its remaining headroom. Do not
            # rescan every checkpoint unless it needs a larger allocation.
            # Report writes deliberately check live shared/free state each time.
            if expanded or self.role == 'postprocess':
                self.last_snapshot = self._check(ledger)
            _write_json_atomic(self.ledger_path, ledger)
        return self.last_snapshot

    def write_bytes(self, path, data):
        """Check known compressed/output bytes before an atomic disk write."""
        path = Path(path).resolve()
        if not path.is_relative_to(self.output):
            raise ValueError('A reserved output must stay inside its collection directory')
        data = bytes(data)
        # The short atomic file write stays inside the shared lock, so another
        # guard never mistakes a visible temp+old file for unreserved new data.
        with _process_lock(self.lock_path):
            ledger = self._ledger()
            row = self._row(ledger)
            needed = len(data)+(ADMIN_WRITE_MARGIN if self.role == 'collection' else 0)
            expanded = needed > row['remaining_bytes']
            if expanded:
                if self.role == 'postprocess':
                    raise ResourceLimitError('An atomic report output exceeds the remaining128MiB report budget')
                delta = needed-row['remaining_bytes']
                row['remaining_bytes'] += delta
                row['allocated_bytes'] += delta
            if expanded or self.role == 'postprocess':
                self.last_snapshot = self._check(ledger)
            previous_size = path.stat().st_size if path.exists() else 0
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name+'.resource-tmp-'+self.reservation_id)
            with temporary.open('xb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            growth = len(data)-previous_size
            row['committed_net_growth_bytes'] += growth
            row['remaining_bytes'] = row['allocated_bytes']-row['preexisting_managed_output_bytes']-row['committed_net_growth_bytes']
            row['writes'] += 1
            _write_json_atomic(self.ledger_path, ledger)

    def commit_external_write(self, path, *, previous_size=0):
        """Account a previously guarded bounded encoder output, after closure."""
        path = Path(path).resolve()
        if not path.is_relative_to(self.output) or not path.is_file():
            raise ValueError('Expected a saved reserved output')
        delta = path.stat().st_size-int(previous_size)
        with _process_lock(self.lock_path):
            ledger = self._ledger()
            row = self._row(ledger)
            row['committed_net_growth_bytes'] += delta
            row['writes'] += 1
            row['remaining_bytes'] = row['allocated_bytes']-row['preexisting_managed_output_bytes']-row['committed_net_growth_bytes']
            if row['remaining_bytes'] < 0:
                raise ResourceLimitError('Output exceeded its preallocated writer budget')
            _write_json_atomic(self.ledger_path, ledger)

    def snapshot(self):
        if not self.entered:
            raise RuntimeError('Reservation is not active')
        with _process_lock(self.lock_path):
            ledger = self._ledger()
            self._row(ledger)
            self.last_snapshot = self._check(ledger)
        return self.last_snapshot

    @property
    def remaining_bytes(self):
        with _process_lock(self.lock_path):
            return int(self._row(self._ledger())['remaining_bytes'])

    def __exit__(self, exc_type, exc, traceback):
        if not self.entered:
            return False
        with _process_lock(self.lock_path):
            ledger = self._ledger()
            row = self._row(ledger)
            row.update(status='released_after_failure' if exc_type else 'released',
                       released_at_utc=utc(), failure=repr(exc) if exc_type else None)
            _write_json_atomic(self.ledger_path, ledger)
            self.entered = False
        return False
