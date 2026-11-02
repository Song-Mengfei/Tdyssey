"""Strict-original data admission and saved RGB windows; no model or simulator.

An externally frozen manifest SHA is required. Renaming an old trajectory or
editing only its protocol label cannot admit it. Candidate admission is separate
from the human authorization required by the training dataset constructor.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT/'artifacts/original-physics-contact-pilot-20261030'
MANIFEST_KIND = 'strict-original-contact-reference.v1'
ATTEMPT_KIND = 'strict-original-contact-attempt.v1'
LOGICAL_SPLITS = {'train': 'train', 'train_candidate': 'train',
                  'seen_val': 'seen_val', 'seen_validation': 'seen_val',
                  'calibration': 'calibration'}


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _protocol():
    # This import constructs no environment and does not import a model.
    from .native_rgb_original_physics import PROTOCOL, PROTOCOL_SHA256
    return PROTOCOL, PROTOCOL_SHA256


def _inside(path, root):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('Data reference escapes its frozen source root')
    return path


def _excluded(path, exclusion_manifest_path=None):
    path = Path(path).resolve()
    roots = [LEGACY_ROOT.resolve()]
    if exclusion_manifest_path is not None:
        excluded = read_json(exclusion_manifest_path)
        if excluded.get('training_eligible') is not False:
            raise ValueError('Exclusion manifest must reject diagnostic data')
        roots += [Path(item).resolve() for item in excluded['excluded_roots']]
    if any(path.is_relative_to(root) for root in roots):
        raise ValueError('Diagnostic/modified-physics source is excluded from new training')


@dataclass(frozen=True)
class VerifiedStrictManifest:
    path: Path
    root: Path
    sha256: str
    value: dict
    sources: dict
    entries: dict
    training_authorized: bool
    exclusion_manifest_path: Path | None


def verify_strict_original_dataset(manifest_path, *, expected_manifest_sha256,
                                   require_training_authorization=True,
                                   exclusion_manifest_path=None):
    """Verify frozen membership, current executed sources and original protocol.

    Set require_training_authorization=False only for saved-data candidate audit.
    It does not authorize a training loader or turn candidate flags into approval.
    """
    path = Path(manifest_path).resolve(strict=True)
    _excluded(path, exclusion_manifest_path)
    if not expected_manifest_sha256 or sha256(path) != expected_manifest_sha256:
        raise ValueError('Externally frozen strict-original manifest SHA differs')
    value = read_json(path)
    protocol, protocol_sha = _protocol()
    if (value.get('kind') != MANIFEST_KIND or value.get('protocol_sha256') != protocol_sha
            or value.get('control_dt') != .1 or value.get('old_physics_data') is not False):
        raise ValueError('Strict original protocol and independent provenance required')
    root = _inside(path.parent/value.get('root', '.'), path.parent)
    _excluded(root, exclusion_manifest_path)
    bindings = value.get('source_bindings', [])
    sources = {}
    for item in bindings:
        source = Path(item['path']).resolve(strict=True)
        if source in sources or sha256(source) != item['sha256']:
            raise ValueError('Executed source binding changed or repeated')
        sources[source] = item['sha256']
    required = [ROOT/'tdyssey/native_rgb_original_physics.py',
                ROOT/'tdyssey/strict_original_contact_pilot.py',
                ROOT/'tdyssey/native_rgb_data_eligibility.py',
                ROOT/'manifests/official_dp.json']
    if any(source.resolve() not in sources for source in required):
        raise ValueError('Original wrapper, collector, loader and pinned source manifest must be bound')
    source_root = Path(value['upstream_source_root']).resolve(strict=True)
    for relative, digest in protocol['source_sha256'].items():
        source = (source_root/relative).resolve(strict=True)
        if sources.get(source) != digest or sha256(source) != digest:
            raise ValueError('Pinned original upstream physics source is not bound unchanged')
    authorized = (value.get('training_eligible') is True
                  and value.get('training_authorized') is True
                  and value.get('pending_user_approval') is False
                  and bool(value.get('human_approval_evidence')))
    if require_training_authorization and not authorized:
        raise ValueError('Candidate data are waiting for explicit human training approval')
    entries = {}
    identities = set()
    for entry in value.get('entries', []):
        raw = _inside(root/entry['file'], root)
        meta = _inside(root/entry['metadata_file'], root)
        _excluded(raw, exclusion_manifest_path)
        if raw in entries or entry['trajectory_id'] in identities:
            raise ValueError('Duplicate trajectory identity or raw file')
        identities.add(entry['trajectory_id'])
        if entry.get('letter') not in tuple('TLOCXB') or entry.get('split') not in LOGICAL_SPLITS:
            raise ValueError('Only six exposed letters and declared splits are admissible')
        if sha256(raw) != entry['sha256'] or sha256(meta) != entry['metadata_sha256']:
            raise ValueError('Frozen raw/metadata hashes differ')
        entries[raw] = entry
    if not entries:
        raise ValueError('A nonempty actual trajectory manifest is required')
    return VerifiedStrictManifest(path, root, expected_manifest_sha256, value,
        sources, entries, authorized,
        None if exclusion_manifest_path is None else Path(exclusion_manifest_path).resolve())


def validate_window(frame_indices, valid_controls, start, stop, *,
                    control_time_seconds, segment_ids):
    """Require every frame/transition in [start, stop] to be legal and adjacent."""
    import numpy as np
    indices = np.asarray(frame_indices)
    valid = np.asarray(valid_controls)
    times = np.asarray(control_time_seconds)
    segments = np.asarray(segment_ids)
    n = len(valid)
    if (valid.dtype != bool or valid.shape != (n,) or indices.dtype.kind not in 'iu'
            or indices.shape != (n+1,) or times.shape != (n+1,)
            or segments.shape != (n+1,) or segments.dtype.kind not in 'iu'
            or type(start) is not int or type(stop) is not int
            or not 0 <= start < stop <= n):
        raise ValueError('Exact saved frame/control timeline is required')
    if (not valid[start:stop].all() or not np.all(np.diff(indices[start:stop+1]) == 1)
            or not np.isfinite(times[start:stop+1]).all()
            or not np.allclose(np.diff(times[start:stop+1]), .1, rtol=0, atol=1e-12)
            or not np.all(segments[start:stop+1] == segments[start])):
        raise ValueError('Window crosses invalid state, reset, deleted frame or timeline gap')
    return True


def guard_trajectory(path, metadata, verified_manifest, *, frame_indices=None,
                     valid_controls=None, control_time_seconds=None, segment_ids=None):
    """Check actual trajectory membership, physics identity and legal mask/ranges."""
    import numpy as np
    verified = verified_manifest
    path = _inside(path, verified.root)
    _excluded(path, verified.exclusion_manifest_path)
    if sha256(verified.path) != verified.sha256:
        raise ValueError('Admission manifest changed after verification')
    if any(sha256(source) != digest for source, digest in verified.sources.items()):
        raise ValueError('Executed source changed after admission')
    entry = verified.entries.get(path)
    if entry is None or sha256(path) != entry['sha256']:
        raise ValueError('Trajectory is not an unchanged frozen member')
    meta_path = _inside(verified.root/entry['metadata_file'], verified.root)
    if sha256(meta_path) != entry['metadata_sha256'] or metadata != read_json(meta_path):
        raise ValueError('Actual frozen trajectory metadata differs')
    protocol, protocol_sha = _protocol()
    identity = metadata.get('initial_identity', {})
    physics = identity.get('physics', {})
    slot = metadata.get('slot', {})
    if (metadata.get('kind') != ATTEMPT_KIND or metadata.get('protocol_sha256') != protocol_sha
            or metadata.get('old_physics_data') is not False
            or identity.get('protocol_sha256') != protocol_sha or identity.get('protocol') != protocol
            or slot.get('slot_id') != entry['trajectory_id']
            or any(slot.get(key) != entry.get(key) for key in ('letter', 'split', 'phase'))):
        raise ValueError('Actual original provenance or trajectory identity differs')
    if (physics.get('controller') != 'original-upstream-kinematic-pd'
            or physics.get('kp') != 100 or physics.get('kv') != 20
            or physics.get('control_dt') != .1 or physics.get('substep_dt') != .01
            or physics.get('substeps_per_control') != 10
            or physics.get('space', {}).get('iterations') != 10
            or physics.get('space', {}).get('damping') != 0
            or physics.get('agent', {}).get('body_type') != 1
            or physics.get('object', {}).get('body_type') != 0):
        raise ValueError('Actual construction is not original100Hz/solver10/kinematic/zero damping')
    for key, name in (('physics_source_sha256', 'native_rgb_original_physics.py'),
                      ('collector_source_sha256', 'strict_original_contact_pilot.py')):
        if metadata.get(key) != verified.sources[(ROOT/'tdyssey'/name).resolve()]:
            raise ValueError('Trajectory executed another physics or collector source')
    if metadata.get('raw', {}).get('sha256') != entry['sha256']:
        raise ValueError('Raw metadata hash differs from frozen membership')
    if valid_controls is not None:
        valid = np.asarray(valid_controls)
        if valid.dtype != bool or valid.ndim != 1:
            raise ValueError('Exact boolean validity mask required')
        ranges = []
        for i, flag in enumerate(valid):
            if flag and (i == 0 or not valid[i-1]):
                ranges.append([i, i+1])
            elif flag:
                ranges[-1][1] = i+1
        if (ranges != metadata.get('valid_contiguous_control_ranges')
                or int(valid.sum()) != metadata.get('actual_valid_controls')):
            raise ValueError('Actual saved mask differs from legal-fragment ranges')
        if metadata.get('contiguous_frames') is not True:
            raise ValueError('This collector must retain contiguous attempted frames')
        # Illegal transitions are retained as evidence; only their legal ranges
        # are eligible for windows. A partial last transition is never repaired.
        for first, last in ranges:
            validate_window(frame_indices, valid, first, last,
                control_time_seconds=control_time_seconds, segment_ids=segment_ids)
    return entry


class StrictOriginalRGBDataset:
    """Future H1 trainer interface; construction rejects unapproved candidates.

    Model items contain four observed RGB frames plus the next target, and the
    four real requested actions. No geometry, labels or privileged states leak.
    """
    def __init__(self, manifest_path, *, expected_manifest_sha256, split='train',
                 exclusion_manifest_path=None):
        if split not in ('train', 'seen_val'):
            raise ValueError('Training loader accepts train or seen_val only')
        self.verified = verify_strict_original_dataset(manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
            exclusion_manifest_path=exclusion_manifest_path)
        self.windows = []
        for path, entry in self.verified.entries.items():
            if LOGICAL_SPLITS[entry['split']] != split:
                continue
            arrays = self._load(path)
            for anchor in range(3, len(arrays['control_valid'])):
                try:
                    validate_window(arrays['frame_indices'], arrays['control_valid'], anchor-3, anchor+1,
                        control_time_seconds=arrays['control_time_seconds'], segment_ids=arrays['segment_ids'])
                except ValueError:
                    continue
                self.windows.append((path, anchor))

    def _load(self, path):
        import numpy as np
        entry = self.verified.entries[path]
        metadata = read_json(self.verified.root/entry['metadata_file'])
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in ('rgb', 'requested_actions', 'control_valid',
                'frame_indices', 'control_time_seconds', 'segment_ids')}
        guard_trajectory(path, metadata, self.verified, frame_indices=arrays['frame_indices'],
            valid_controls=arrays['control_valid'], control_time_seconds=arrays['control_time_seconds'],
            segment_ids=arrays['segment_ids'])
        n = len(arrays['control_valid'])
        rgb, actions = arrays['rgb'], arrays['requested_actions']
        if (rgb.dtype != np.uint8 or rgb.shape != (n+1, 3, 96, 96)
                or actions.dtype != np.float64 or actions.shape != (n, 2)
                or not np.isfinite(actions).all() or (actions < 0).any() or (actions > 511).any()):
            raise ValueError('Exact96RGB and real requested actions are required')
        return arrays

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        import numpy as np
        path, anchor = self.windows[index]
        arrays = self._load(path)
        validate_window(arrays['frame_indices'], arrays['control_valid'], anchor-3, anchor+1,
            control_time_seconds=arrays['control_time_seconds'], segment_ids=arrays['segment_ids'])
        return dict(image=arrays['rgb'][anchor-3:anchor+2].astype(np.float32)/255.,
                    action=arrays['requested_actions'][anchor-3:anchor+1].astype(np.float32)/511.*2.-1.)
