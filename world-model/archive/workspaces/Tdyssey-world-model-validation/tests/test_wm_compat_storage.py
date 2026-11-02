"""Adversarial byte-only storage tests, exclusively in temporary directories."""

import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import tdyssey.wm_compat_storage as storage
from tdyssey.wm_compat_storage import Coordinator, StorageError, create_lock_file, scan_files, unlinked_path


class StrictStorageChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.result,self.telemetry=self.root/'results',self.root/'telemetry'
        self.result.mkdir();self.telemetry.mkdir()
        self.coordinators=[]
        self.addCleanup(self.close_all)
    def close_all(self):
        for coordinator in self.coordinators: coordinator.close()
    def coordinator(self,shared=False,**kwargs):
        if shared:
            lock=self.telemetry/'storage.lock'
            if not lock.exists():create_lock_file(lock)
            kwargs['lock_path']=lock
        value=Coordinator(result_roots=(self.result,),telemetry_roots=(self.telemetry,),**kwargs)
        self.coordinators.append(value)
        return value
    def publish(self,c,path,payload,maximum=None):
        ticket=c.reserve_bytes(path,len(payload) if maximum is None else maximum)
        path.write_bytes(payload);c.remember_file(path);c.commit(ticket,len(payload))
        return ticket
    def test_hot_quota_has_no_scans_or_old_content_hashes(self):
        c=self.coordinator(shared=True)
        with patch.object(storage,'scan_files',side_effect=AssertionError('hot scan')),patch.object(storage,'fingerprint',side_effect=AssertionError('hot hash')):
            lease=c.reserve_bytes(self.result/'a',10);c.commit(lease,7)
            for _ in range(20):
                self.assertEqual(c.usage()['output_bytes'],10)
                c.check_growth(self.result/'b',1)
        self.assertEqual(c.usage()['output_observed_bytes'],7)
    def test_capacity_reserves_before_io_and_failed_maximum_is_not_refunded(self):
        c=self.coordinator(limits={'output_bytes':16,'telemetry_bytes':16})
        lease=c.reserve_bytes(self.result/'first',10);c.commit(lease,5)
        c.reserve_bytes(self.result/'failed',6)
        self.assertEqual(c.usage()['output_bytes'],16)
        self.assertEqual(c.usage()['output_observed_bytes'],5)
        with self.assertRaises(StorageError):c.reserve_bytes(self.result/'overflow',1)
        with self.assertRaises(StorageError):c.commit(lease,5)
    def test_two_instances_share_maximum_and_actual_counters(self):
        a=self.coordinator(shared=True);b=self.coordinator(shared=True)
        ticket=a.reserve_bytes(self.telemetry/'log',13);a.commit(ticket,8)
        self.assertEqual(b.usage()['telemetry_bytes'],4097+13)
        self.assertEqual(b.usage()['telemetry_observed_bytes'],8)
    def test_different_policy_cannot_attach(self):
        self.coordinator(shared=True)
        with self.assertRaises(StorageError):self.coordinator(shared=True,pending_reserve=1)
    def test_slow_owner_io_does_not_own_the_quota_mutex(self):
        a=self.coordinator(shared=True);b=self.coordinator(shared=True)
        ready,release=threading.Event(),threading.Event();errors=[]
        def write():
            try:
                with a.local_transaction('slow-IO'):
                    self.assertFalse(a.accounting_lock_held);ready.set()
                    if not release.wait(2):raise AssertionError('barrier')
            except BaseException as error:errors.append(error)
        thread=threading.Thread(target=write);thread.start()
        try:
            self.assertTrue(ready.wait(1))
            lease=b.reserve_bytes(self.telemetry/'progress',5);b.commit(lease,5)
            self.assertEqual(b.usage()['telemetry_observed_bytes'],5)
        finally:release.set();thread.join(3)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
    def test_prepaid_scratch_covers_publication_overlap(self):
        c=self.coordinator(pending_reserve=2*storage.MIB)
        final,temp=self.result/'final',self.result/'partial'
        lease=c.reserve_bytes(final,16,temporary_bytes=16)
        temp.write_bytes(b'a'*16);final.write_bytes(b'a'*16)
        self.assertEqual(c.usage()['output_bytes'],2*storage.MIB+16)
        with self.assertRaises(StorageError):c.reserve_bytes(self.result/'other',1,temporary_bytes=1)
        temp.unlink();c.remember_file(final);c.commit(lease,16)
        self.assertEqual(c.audit()['output_observed_bytes'],16)
    def test_telemetry_temporary_capacity_released_only_on_success(self):
        c=self.coordinator(limits={'telemetry_bytes':32})
        lease=c.reserve_bytes(self.telemetry/'a',8,temporary_bytes=8)
        self.assertEqual(c.usage()['telemetry_bytes'],16);c.commit(lease,6)
        self.assertEqual(c.usage()['telemetry_bytes'],8)
        c.reserve_bytes(self.telemetry/'failed',8,temporary_bytes=8)
        self.assertEqual(c.usage()['telemetry_bytes'],24)

    def assert_storage_error(self, action):
            with self.assertRaises(StorageError) as caught:
                action()
            self.assertIsInstance(caught.exception.details, dict)
            self.assertTrue(caught.exception.details)
            return caught.exception

    def test_snapshot_contains_content_hash_metadata_and_identity(self):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = b"ordinary immutable bytes"
                path = root/"unknown.tmp"
                path.write_bytes(payload)
                records = scan_files(root, phase="snapshot-check")
                self.assertEqual(len(records), 1)
                record = next(iter(records.values()))
                self.assertEqual(record.size, len(payload))
                self.assertEqual(record.mtime_ns, path.stat().st_mtime_ns)
                self.assertEqual(record.sha256, hashlib.sha256(payload).hexdigest())
                self.assertIsNotNone(record.identity)

    def test_missing_scan_root_is_not_an_empty_snapshot(self):
            with tempfile.TemporaryDirectory() as directory:
                missing = Path(directory)/"missing-root"
                self.assert_storage_error(lambda: scan_files(missing, phase="missing-root"))
                self.assertEqual(scan_files(missing, phase="explicit-initial-root", allow_missing_root=True), {})

    def test_ancestor_symlink_mode_and_reparse_attributes_are_rejected(self):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                original = os.lstat
                for kind in ("symlink", "reparse"):
                    def fake_lstat(path, *args, **kwargs):
                        record = original(path, *args, **kwargs)
                        if Path(path) != root:
                            return record
                        fields = {name: getattr(record, name) for name in dir(record) if name.startswith("st_")}
                        if kind == "symlink":
                            fields["st_mode"] = stat.S_IFLNK|0o777
                        else:
                            fields["st_file_attributes"] = fields.get("st_file_attributes", 0)|0x400
                        return SimpleNamespace(**fields)
                    with self.subTest(kind=kind), patch.object(storage.os, "lstat", side_effect=fake_lstat):
                        self.assert_storage_error(lambda: unlinked_path(root/"future-child"))
                        self.assert_storage_error(lambda: scan_files(root, phase="reject-"+kind))

    def test_enumeration_failure_and_unknown_tmp_disappearance_fail_closed(self):
            class IteratorProxy:
                def __init__(self, iterator, before=None, failure=None):
                    self.iterator, self.before, self.failure = iterator, before, failure
                def __iter__(self): return self
                def __next__(self):
                    if self.failure is not None:
                        raise self.failure
                    entry = next(self.iterator)
                    if self.before:
                        self.before(entry)
                    return entry
                def close(self): self.iterator.close()
                def __enter__(self): return self
                def __exit__(self, *args): self.close()
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root/"unknown.tmp"
                target.write_bytes(b"never ignore this file")
                original = os.scandir
                for error in (PermissionError(13, "injected enumeration denial", str(root)), OSError(5, "injected enumeration failure", str(root))):
                    def denied(path):
                        iterator = original(path)
                        return IteratorProxy(iterator, failure=error) if Path(path) == root else iterator
                    with self.subTest(error=type(error).__name__), patch.object(storage.os, "scandir", side_effect=denied):
                        self.assert_storage_error(lambda: scan_files(root, phase="enumeration-denied"))
                def vanishing(path):
                    iterator = original(path)
                    def before(entry):
                        if Path(entry.path) == target:
                            target.unlink()
                    return IteratorProxy(iterator, before=before) if Path(path) == root else iterator
                with patch.object(storage.os, "scandir", side_effect=vanishing):
                    self.assert_storage_error(lambda: scan_files(root, phase="unknown-tmp-vanished"))

    def test_permission_and_hash_read_errors_remain_storage_errors(self):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root/"payload.bin"
                target.write_bytes(b"data")
                original_open, original_read = os.open, os.read
                for error in (PermissionError(13, "injected read denial", str(target)), OSError(5, "injected read failure", str(target))):
                    target_fds = set()
                    def tracked_open(path, *args, **kwargs):
                        fd = original_open(path, *args, **kwargs)
                        if Path(path) == target:
                            target_fds.add(fd)
                        return fd
                    def denied_read(fd, *args, **kwargs):
                        if fd in target_fds:
                            raise error
                        return original_read(fd, *args, **kwargs)
                    with self.subTest(error=type(error).__name__), patch.object(storage.os, "open", tracked_open), patch.object(storage.os, "read", denied_read):
                        caught = self.assert_storage_error(lambda: scan_files(root, phase="hash-read-denied"))
                        self.assertEqual(caught.details["exception_type"], type(error).__name__)
                        self.assertEqual(caught.details["errno"], error.errno)
                        self.assertTrue(caught.details["phase"].endswith(".read"))

    def test_create_lock_is_exclusive_and_one_byte(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"lock"
                create_lock_file(path)
                payload = path.read_bytes()
                self.assertEqual(len(payload), 1)
                with self.assertRaises((StorageError, FileExistsError)):
                    create_lock_file(path)
                self.assertEqual(path.read_bytes(), payload)
    def test_fixed_slots_prepaid_and_oversize_rejected_at_audit(self):
        status,stop=self.telemetry/'status',self.telemetry/'stop'
        c=self.coordinator(shared=True,atomic_targets={status:16},fixed_slots={stop:16},emergency_headroom=16,telemetry_reserve=16)
        expected=4097+32+16+16+16
        self.assertEqual(c.usage()['telemetry_bytes'],expected)
        c.claim_fixed(status);c.claim_fixed(stop)
        status.write_bytes(b'a'*16);stop.write_bytes(b'b'*16)
        self.assertEqual(c.audit()['telemetry_observed_bytes'],4097+32)
        self.assertEqual(c.usage()['telemetry_bytes'],expected)
        with self.assertRaises(StorageError):c.claim_fixed(self.telemetry/'unknown')
        status.write_bytes(b'a'*17)
        with self.assertRaises(StorageError):c.audit()
    def test_audit_detects_same_size_restored_mtime_mutation_and_missing(self):
        c=self.coordinator();path=self.result/'sealed'
        with patch.object(storage,'_change_token',return_value=1):
            self.publish(c,path,b'first');before=path.stat()
            path.write_bytes(b'other');os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            self.assertEqual(c.usage()['output_bytes'],5)
            with self.assertRaises(StorageError):c.audit()
        path.unlink()
        with self.assertRaises(StorageError):c.audit()
    def test_unregistered_extra_file_is_an_audit_error(self):
        c=self.coordinator();self.result.joinpath('unknown.tmp').write_bytes(b'unregistered')
        self.assertEqual(c.usage()['output_bytes'],0)
        with self.assertRaisesRegex(StorageError,'Physical bytes'):c.audit()
    def test_cross_owner_expected_manifest_is_checked(self):
        writer=self.coordinator(shared=True);auditor=self.coordinator(shared=True)
        path=self.result/'child';self.publish(writer,path,b'first')
        expected={str(path):writer._observed[str(path)]}
        self.assertEqual(auditor.audit(expected=expected)['output_observed_bytes'],5)
        before=path.stat();path.write_bytes(b'other');os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaises(StorageError):auditor.audit(expected=expected)
    def test_append_only_reads_new_payload_and_audits_prefix_at_boundary(self):
        path=self.telemetry/'journal';c=self.coordinator(append_only=(path,))
        with path.open('xb',buffering=0) as stream:
            for payload in (b'first\n',b'second\n'):
                lease=c.reserve_bytes(path,len(payload));c.validate_append(path,stream)
                with patch.object(storage.os,'read',side_effect=AssertionError('old history read')):
                    stream.write(payload);c.record_append(path,payload,stream)
                c.commit(lease,len(payload))
        self.assertEqual(c.audit()['telemetry_observed_bytes'],13)
        before=path.stat();path.write_bytes(b'other\nsecond\n');os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with path.open('ab',buffering=0) as stream,self.assertRaises(StorageError):c.validate_append(path,stream)
        with self.assertRaises(StorageError):c.audit()
    def test_counter_header_and_lock_corruption_fail_closed(self):
        c=self.coordinator(shared=True);original=c.counter_path.read_bytes()
        with c.counter_path.open('r+b',buffering=0) as stream:stream.write(b'BADHEAD!')
        with self.assertRaises(StorageError):c.usage()
        with c.counter_path.open('r+b',buffering=0) as stream:stream.write(original)
        with c.lock_path.open('r+b',buffering=0) as stream:stream.write(b'00')
        with self.assertRaises(StorageError):c.usage()
    def test_cross_process_counter_sharing(self):
        import json
        c=self.coordinator(shared=True)
        child="""
import json,sys
from pathlib import Path
from tdyssey.wm_compat_storage import Coordinator
r=Path(sys.argv[1]);c=Coordinator(lock_path=r/'telemetry/storage.lock',result_roots=(r/'results',),telemetry_roots=(r/'telemetry',))
t=c.reserve_bytes(r/'results/child.bin',9);c.commit(t,5)
print(json.dumps(c.usage()),flush=True);c.close()
"""
        process=subprocess.run([sys.executable,'-S','-B','-c',child,str(self.root)],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=5)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertEqual(json.loads(process.stdout)['output_bytes'],9)
        self.assertEqual(c.usage()['output_observed_bytes'],5)

    def test_duplicate_concurrent_commit_releases_temporary_capacity_once(self):
        c=self.coordinator(shared=True)
        ticket=c.reserve_bytes(self.telemetry/'one',8,temporary_bytes=8)
        ready=threading.Barrier(3);success=[];errors=[]
        def finish():
            ready.wait()
            try:c.commit(ticket,8);success.append(True)
            except StorageError as error:errors.append(error)
        threads=[threading.Thread(target=finish) for _ in range(2)]
        for thread in threads:thread.start()
        ready.wait()
        for thread in threads:thread.join(2)
        self.assertEqual((len(success),len(errors)),(1,1))
        self.assertEqual(c.usage()['telemetry_observed_bytes'],8)
        self.assertEqual(c.usage()['telemetry_bytes'],4097+8)
    def test_closed_coordinator_cannot_reset_or_reuse_quota(self):
        c=self.coordinator(shared=True)
        ticket=c.reserve_bytes(self.result/'one',5);c.commit(ticket,5)
        c.close()
        with self.assertRaises(StorageError):c.usage()
        other=self.coordinator(shared=True)
        self.assertEqual(other.usage()['output_bytes'],5)
    def test_prepaid_control_page_is_checked_before_creation(self):
        with self.assertRaises(StorageError):self.coordinator(shared=True,limits={'telemetry_bytes':16})
        self.assertFalse((self.telemetry/'storage.lock.counters').exists())
    def test_windows_byte_contention_waits_instead_of_reading_locked_byte(self):
        import json
        c=self.coordinator(shared=True)
        child="""
import json,sys
from pathlib import Path
from tdyssey.wm_compat_storage import Coordinator,StorageError
r=Path(sys.argv[1]);c=Coordinator(lock_path=r/'telemetry/storage.lock',result_roots=(r/'results',),telemetry_roots=(r/'telemetry',),timeout=.05)
print('ready',flush=True)
sys.stdin.readline()
try:c.usage()
except StorageError as error:print(json.dumps(error.details),flush=True)
else:raise AssertionError('Held mutex was ignored')
sys.stdin.readline()
print(json.dumps(c.usage()),flush=True);c.close()
"""
        process=subprocess.Popen([sys.executable,'-S','-B','-c',child,str(self.root)],
            cwd=Path(__file__).resolve().parents[1],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(),'ready')
            with c.transaction('test.parent_owns_byte'):
                process.stdin.write('attempt\n');process.stdin.flush()
                blocked=json.loads(process.stdout.readline())
                self.assertEqual(blocked['exception_type'],'TimeoutError')
                self.assertTrue(blocked['phase'].endswith('.lock_acquire'))
            process.stdin.write('released\n');process.stdin.flush()
            self.assertEqual(json.loads(process.stdout.readline())['output_bytes'],0)
            self.assertEqual(process.wait(timeout=5),0,process.stderr.read())
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=5)
            for stream in (process.stdin,process.stdout,process.stderr):stream.close()

if __name__=='__main__':unittest.main()
