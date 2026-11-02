"""Stdlib synthetic byte/IO fixtures; all explicit Win32 file APIs are patched.

Root owns execution. These tests import no Torch/NumPy/SDK, launch no processes,
query no resources, and touch only NEW workspace temporary fixtures.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tdyssey import gpc_evaluation_status as journal

EVIDENCE = Path(__file__).resolve().parents[1]/'artifacts/gpc-checkpoint-evaluation-engineering-20261026/append-unit-fixtures'


def status(**changes):
    value = dict(phase='setup', training_complete=False, model_instances=0, model_loads=0,
        optimizer_updates=0, forecasts=0, euler_steps=0, eval_batches=0, eval_examples=0,
        **dict.fromkeys(journal._FORBIDDEN, 0),
        cuda_allocated_bytes=None, cuda_reserved_bytes=None, cuda_peak_allocated_bytes=None,
        cuda_peak_reserved_bytes=None, cuda_sample_at_unix=None, cuda_sample_at_monotonic=None,
        cuda_sample_state='unavailable')
    value.update(changes)
    return value


class TrackingStream:
    def __init__(self, stream):self.stream = stream;self.reads = []
    def __getattr__(self, name):return getattr(self.stream, name)
    def read(self, count):
        raw = self.stream.read(count);self.reads.append((count, len(raw)));return raw
    def seek(self, *args):raise AssertionError('Reader must never seek/re-read history')
    def __enter__(self):return self
    def __exit__(self, *args):self.close()


class EvaluationStatusTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='status-', dir=EVIDENCE)
        self.addCleanup(self.temp.cleanup);self.root = Path(self.temp.name)
        self.path = self.root/journal.STATUS_NAME;self.pid = os.getpid();self.nonce = 'synthetic-status-nonce'
        self.clock = 0;self.opened = []
        self.original_snapshot = journal._snapshot
        def open_file(path, *, create=False):
            raw = Path(path).open('xb' if create else 'rb', buffering=0)
            result = raw if create else TrackingStream(raw)
            self.opened.append(result)
            return result
        def snapshot(stream):
            value = os.fstat(stream.fileno())
            if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
                raise journal.StatusJournalError('Synthetic nonordinary/single-link check')
            return (value.st_dev, value.st_ino), value.st_size
        self.patch_open = patch.object(journal, '_open_file', side_effect=open_file)
        self.patch_snapshot = patch.object(journal, '_snapshot', side_effect=snapshot)
        self.patch_kernel = patch.object(journal, '_kernel', side_effect=AssertionError('Native API forbidden in synthetic tests'))
        for current in (self.patch_open, self.patch_snapshot, self.patch_kernel):
            current.start();self.addCleanup(current.stop)

    def writer(self, **kwargs):
        result = journal.Writer(self.path, pid=self.pid, nonce=self.nonce,
            clock=lambda:1000.+self.clock, monotonic=lambda:10.+self.clock, **kwargs)
        self.addCleanup(result.close)
        return result

    def reader(self, **kwargs):
        result = journal.Reader(self.path, pid=self.pid, nonce=self.nonce, **kwargs)
        self.addCleanup(result.close)
        return result

    def frame(self, sequence, **changes):
        value = status(actual_worker_pid=self.pid, nonce=self.nonce,
            at_unix=1000.+sequence, at_monotonic=10.+sequence)
        value.update(changes)
        return journal._frame(sequence, json.dumps(value, separators=(',', ':'), allow_nan=False).encode())

    def test_capacity_bound_covers_full_eval_progress_and_heartbeat(self):
        self.assertEqual(journal.MAX_ENCODED_RECORD_BYTES, 1592)
        self.assertEqual(journal.MAX_RECORDS*journal.MAX_ENCODED_RECORD_BYTES, 15920000)
        self.assertLess(4035+3600+128, journal.MAX_RECORDS)
        self.assertLess(journal.MAX_RECORDS*journal.MAX_ENCODED_RECORD_BYTES, journal.MAX_BYTES)
        self.assertLess(journal.MAX_ENCODED_RECORD_BYTES, journal.MAX_RECORD_BYTES)

    def test_incremental_roundtrip_never_rereads_or_refreshes_without_new_record(self):
        writer = self.writer();first = writer.publish(status())
        reader = self.reader(max_read_bytes=17);observed = []
        while reader.sequence < 1:
            value = reader.poll()
            if value is not None:observed.append(value)
        self.assertEqual(observed, [first]);self.assertEqual(reader.latest, first)
        reads = len(reader._stream.reads)
        self.assertIsNone(reader.poll());self.assertEqual(len(reader._stream.reads), reads)
        self.clock = 1;second = writer.publish(status(model_loads=1))
        while reader.sequence < 2:reader.poll()
        self.assertEqual(reader.latest, second);self.assertEqual(reader.partial_bytes, 0)
        self.assertEqual(reader.committed_bytes, writer.committed_bytes)
        self.assertEqual(sum(n for _, n in reader._stream.reads), self.path.stat().st_size)
        self.assertTrue(all(request <= 17 for request, _ in reader._stream.reads))
        altered = reader.latest;altered['phase'] = 'training'
        self.assertEqual(reader.latest, second)

    def test_partial_write_eio_retains_charge_tail_first_error_and_no_freshness(self):
        writer = self.writer();first = writer.publish(status());reader = self.reader()
        self.assertEqual(reader.poll(), first);before = self.path.read_bytes()
        error = OSError(errno.EIO, 'injected after seven bytes');error.winerror = 1117
        def partial(stream, raw):stream.write(raw[:7]);raise error
        with patch.object(journal, '_write_all', side_effect=partial):
            with self.assertRaises(OSError) as caught:writer.publish(status(model_loads=1))
        self.assertIs(caught.exception, error);self.assertIs(writer.first_error, error)
        self.assertEqual((writer.reserved_records, writer.committed_records), (2, 1))
        self.assertGreater(writer.reserved_bytes, writer.committed_bytes)
        retained = self.path.read_bytes();self.assertEqual(retained[:len(before)], before)
        self.assertEqual(len(retained)-len(before), 7)
        self.assertIsNone(reader.poll());self.assertEqual(reader.partial_bytes, 7);self.assertEqual(reader.latest, first)
        with self.assertRaises(journal.StatusJournalError) as again:writer.publish(status())
        self.assertIs(again.exception.__cause__, error);writer.close()
        self.assertEqual(self.path.read_bytes(), retained)

    def test_partial_header_body_and_commit_marker_do_not_update_until_complete(self):
        writer = self.writer();first = writer.publish(status());writer.close()
        reader = self.reader();self.assertEqual(reader.poll(), first)
        second = self.frame(2);points = (7, journal._HEADER.size+5, len(second)-1, len(second))
        prior = 0
        with self.path.open('ab', buffering=0) as stream:
            for end in points:
                stream.write(second[prior:end]);prior = end
                value = reader.poll()
                if end != len(second):
                    self.assertIsNone(value);self.assertEqual(reader.latest, first);self.assertEqual(reader.sequence, 1)
                else:self.assertEqual(value['at_monotonic'], 12.)
        self.assertEqual(reader.sequence, 2);self.assertEqual(reader.partial_bytes, 0)

    def test_complete_corruption_duplicate_gap_and_schema_errors_fail_closed(self):
        first = self.frame(1)
        flipped = bytearray(first);flipped[journal._HEADER.size+2] ^= 1
        variants = dict(checksum=bytes(flipped), marker=first[:-1]+b'X', duplicate=first+first,
            gap=self.frame(2), invalid_counter=self.frame(1, optimizer_updates=1),
            invalid_phase=self.frame(1, phase='training'), identity=self.frame(1, nonce='different'))
        for name, raw in variants.items():
            with self.subTest(name=name):
                directory = self.root/name;directory.mkdir();path = directory/journal.STATUS_NAME;path.write_bytes(raw)
                with journal.Reader(path, pid=self.pid, nonce=self.nonce) as reader:
                    with self.assertRaises(journal.StatusJournalError):reader.poll()
                    first_error = reader.first_error
                    with self.assertRaises(journal.StatusJournalError) as repeated:reader.poll()
                    self.assertIs(repeated.exception.__cause__, first_error)

    def test_bad_middle_frame_cannot_be_hidden_by_new_good_frame(self):
        self.path.write_bytes(self.frame(1)+self.frame(2, optimizer_updates=1)+self.frame(3))
        reader = self.reader()
        with self.assertRaisesRegex(journal.StatusJournalError, 'optimizer_updates'):reader.poll()
        self.assertIsNone(reader.latest);self.assertEqual(reader.sequence, 0)

    def test_additional_validator_runs_for_each_frame_and_rejects_middle(self):
        self.path.write_bytes(self.frame(1)+self.frame(2)+self.frame(3));seen = []
        def validate(value):
            seen.append(value['at_monotonic'])
            return 'synthetic middle rejection' if value['at_monotonic'] == 12 else None
        reader = self.reader(validate=validate)
        with self.assertRaisesRegex(journal.StatusJournalError, 'middle rejection'):reader.poll()
        self.assertEqual(seen, [11., 12.]);self.assertIsNone(reader.latest)

    def test_backward_monotonic_middle_frame_is_not_skipped(self):
        first = self.frame(1);value = status(actual_worker_pid=self.pid, nonce=self.nonce, at_unix=1002., at_monotonic=9.)
        middle = journal._frame(2, json.dumps(value, separators=(',', ':')).encode())
        self.path.write_bytes(first+middle+self.frame(3))
        with self.assertRaisesRegex(journal.StatusJournalError, 'backward'):self.reader().poll()

    def test_record_byte_and_payload_limits_admit_before_writing(self):
        for kind in ('records', 'bytes', 'payload'):
            with self.subTest(kind=kind):
                directory = self.root/kind;directory.mkdir();self.path = directory/journal.STATUS_NAME
                writer = self.writer(**({'max_records':1} if kind == 'records' else {'max_bytes':1} if kind == 'bytes' else {}))
                if kind == 'records':writer.publish(status())
                before = self.path.read_bytes()
                with patch.object(journal, '_write_all', side_effect=AssertionError('Rejected operation wrote bytes')) as write:
                    with self.assertRaises(journal.StatusJournalError):
                        writer.publish(status(extra='x'*1536) if kind == 'payload' else status())
                    write.assert_not_called()
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual(writer.committed_records, 1 if kind == 'records' else 0)
        for kwargs in ({'max_records':journal.MAX_RECORDS+1}, {'max_bytes':journal.MAX_BYTES+1}):
            with self.assertRaises(ValueError):journal.Writer(self.root/journal.STATUS_NAME, pid=self.pid, nonce=self.nonce, **kwargs)
        self.assertFalse((self.root/journal.STATUS_NAME).exists())

    def test_fsync_error_keeps_complete_visible_bytes_but_never_claims_writer_commit(self):
        writer = self.writer();error = OSError(errno.EIO, 'injected fsync failure');error.winerror = 1117
        with patch.object(journal.os, 'fsync', side_effect=error):
            with self.assertRaises(OSError) as caught:writer.publish(status())
        self.assertIs(caught.exception, error);self.assertEqual(error.heartbeat_step, 'status.append.fsync')
        self.assertEqual((writer.reserved_records, writer.committed_records), (1, 0))
        self.assertEqual(writer.committed_bytes, 0);self.assertGreater(self.path.stat().st_size, 0)
        # Visibility is not proof of fsync success. Parent must first check HB error.
        reader = self.reader();self.assertIsNotNone(reader.poll());self.assertIs(writer.first_error, error)

    def test_truncation_and_path_fileid_change_fail_even_without_new_record(self):
        writer = self.writer();writer.publish(status());writer.close();reader = self.reader();reader.poll()
        os.truncate(self.path, 1)
        with self.assertRaisesRegex(journal.StatusJournalError, 'truncated'):reader.poll()
        reader.close();directory = self.root/'identity';directory.mkdir();self.path = directory/journal.STATUS_NAME
        writer = self.writer();writer.publish(status());writer.close();reader = self.reader();reader.poll()
        changed = (reader.file_id[0], reader.file_id[1]+1)
        with patch.object(journal, '_path_snapshot', return_value=(changed, reader.bytes_seen)):
            with self.assertRaisesRegex(journal.StatusJournalError, 'fileID'):reader.poll()

    def test_fixed_name_exclusive_writer_and_no_resolve(self):
        with self.assertRaises(journal.StatusJournalError):journal.Writer(self.root/'other.json', pid=self.pid, nonce=self.nonce)
        with patch.object(Path, 'resolve', side_effect=AssertionError('No resolve/canonicalization allowed')):
            writer = self.writer();writer.publish(status());reader = self.reader();reader.poll()
            with self.assertRaises(FileExistsError):journal.Writer(self.path, pid=self.pid, nonce=self.nonce)
        with self.assertRaises(ValueError):journal.Writer(self.root/'elsewhere'/journal.STATUS_NAME, pid=self.pid+1, nonce=self.nonce)

    def test_reparse_ancestor_and_native_nonordinary_link_flags_reject_synthetically(self):
        with patch.object(journal.os, 'lstat', return_value=SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)):
            with self.assertRaisesRegex(journal.StatusJournalError, 'non-reparse'):journal._path(self.path)
        for attributes, links, file_type in ((0x400, 1, 1), (0x10, 1, 1), (0, 2, 1), (0, 0, 1), (0, 1, 2)):
            with self.subTest(attributes=attributes, links=links, file_type=file_type):
                def information(handle, pointer):
                    value = pointer._obj;value.attributes = attributes;value.links = links
                    value.volume = 7;value.index_low = 19
                    return 1
                fake = SimpleNamespace(GetFileInformationByHandle=information, GetFileType=lambda handle:file_type)
                with patch.object(journal.sys, 'platform', 'win32'), patch.object(journal, '_kernel', return_value=fake), \
                     patch.dict('sys.modules', msvcrt=SimpleNamespace(get_osfhandle=lambda fd:123)):
                    with self.assertRaises(journal.StatusJournalError):self.original_snapshot(SimpleNamespace(fileno=lambda:1))

    def test_ordinary_reader_can_keep_prefix_open_while_writer_appends(self):
        writer = self.writer();writer.publish(status());first = self.path.read_bytes()
        with self.path.open('rb') as held:
            prefix = held.read(7);self.clock = 1;writer.publish(status(model_loads=1))
            self.assertEqual(prefix+held.read(len(first)-7), first)
            self.assertGreater(len(held.read()), 0)
        reader = self.reader();self.assertIsNotNone(reader.poll());self.assertEqual(reader.sequence, 2)

    def test_reader_io_error_preserves_errno_winerror_and_never_retries(self):
        writer = self.writer();writer.publish(status());reader = self.reader()
        original = reader._stream;error = PermissionError(errno.EACCES, 'synthetic ACL read');error.winerror = 5
        class BrokenRead:
            calls = 0
            def __getattr__(self, name):return getattr(original, name)
            def read(self, count):self.calls += 1;raise error
        broken = BrokenRead();reader._stream = broken
        with self.assertRaises(PermissionError) as caught:reader.poll()
        self.assertIs(caught.exception, error);self.assertEqual(error.winerror, 5)
        with self.assertRaises(journal.StatusJournalError) as repeated:reader.poll()
        self.assertIs(repeated.exception.__cause__, error);self.assertEqual(broken.calls, 1)

    def test_primary_partial_write_survives_secondary_close_without_buffer_replay(self):
        writer = self.writer();original = writer._stream
        primary = OSError(errno.EIO, 'first partial write');secondary = OSError(errno.EBADF, 'secondary close')
        class CloseFailure:
            def __getattr__(self, name):return getattr(original, name)
            def close(self):original.close();raise secondary
        writer._stream = CloseFailure()
        def partial(stream, raw):stream.write(raw[:7]);raise primary
        with patch.object(journal, '_write_all', side_effect=partial):
            with self.assertRaises(OSError) as caught:
                with writer:writer.publish(status())
        self.assertIs(caught.exception, primary);self.assertIs(writer.first_error, primary)
        self.assertEqual(self.path.stat().st_size, 7)
        self.assertTrue(any('secondary close' in note for note in primary.__notes__))


if __name__ == '__main__':unittest.main()
