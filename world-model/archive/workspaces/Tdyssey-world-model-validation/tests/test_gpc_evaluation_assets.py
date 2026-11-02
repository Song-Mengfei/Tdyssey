"""Stdlib markers only: no real assets, tensor runtime or source accounts."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tdyssey import gpc_evaluation_assets as assets_module


class TensorMarker:
    def __init__(self, shape=(2,), dtype='torch.float32', finite=True, device='cpu', grad=False):
        self.shape, self.dtype, self.finite = shape, dtype, finite
        self.device = SimpleNamespace(type=device)
        self.layout, self.requires_grad = 'strided', grad

    def numel(self):
        return math.prod(self.shape)

    def clone(self):
        raise AssertionError('A model-only load must not clone rollback weights')


class Poison:
    def __getattribute__(self, name):
        raise AssertionError('Optimizer/RNG contents must not be inspected/restored')


class TorchMarker:
    Tensor, strided = TensorMarker, 'strided'

    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def load(self, path, **kwargs):
        assert kwargs == dict(map_location='cpu', weights_only=True, mmap=True)
        self.calls.append(path)
        return self.payload

    def isfinite(self, tensor):
        return SimpleNamespace(all=lambda: tensor.finite)

    def __getattr__(self, name):
        raise AssertionError('Unexpected Torch/optimizer/RNG operation: '+name)


class EvaluationAssetTests(unittest.TestCase):
    @contextmanager
    def fake_assets(self):
        a = assets_module
        root = a.ROOT/'artifacts/gpc-checkpoint-evaluation-engineering-20261026/marker-only'
        spec = {f'w{i}': dict(shape=[2], dtype='torch.float32') for i in range(298)}
        attempt = dict(binding=dict(config_sha256=a.CONFIG_SHA))
        boundary = dict(spent_reserved_updates=500, spent_train_examples=500)
        with ExitStack() as stack:
            stack.enter_context(patch.object(a, 'SPEC_SHA', a._sha(a._canonical(spec))))
            stack.enter_context(patch.object(a, 'BOUNDARY_SHA', a._sha(a._canonical(boundary))))
            stack.enter_context(patch.dict(a.METADATA, attempt=(a.METADATA['attempt'][0], a._sha(a._canonical(attempt)+b'\n'))))
            value = dict(kind='gpc_eval_assets.v1', repository_root=str(root), sources=a.source_declarations(),
                weights={role: dict(path=str(root/rel), bytes=size, sha256=digest)
                         for role, (rel, size, digest) in a.WEIGHTS.items()},
                metadata={name: dict(path=str(root/rel), bytes=1, sha256=digest)
                          for name, (rel, digest) in a.METADATA.items()},
                source_logical_steps=500, new_account_updates=0, source_attempt=attempt,
                selection_boundary=boundary, selection_sequence=1014, model_spec=spec)
            yield value

    def payload(self, value):
        a = assets_module
        return dict(kind='gpc_adaptation_checkpoint.v1', logical_steps=500,
            attempt=deepcopy(value['source_attempt']), config_identity=deepcopy(value['source_attempt']['binding']),
            budget=deepcopy(value['selection_boundary']), extra_state=dict(next_example=500, training_schedule=a._schedule_identity(500)),
            model={name: TensorMarker() for name in value['model_spec']}, optimizer=Poison(), rng=Poison())

    def test_cheap_prepared_check_has_no_io_and_keeps_source500_account0(self):
        with self.fake_assets() as value, patch.object(Path, 'open', side_effect=AssertionError('Unexpected file I/O')):
            assets_module.check_prepared_assets(value)
            self.assertEqual((value['source_logical_steps'], value['new_account_updates']), (500, 0))
            for mutation in ('new_account_updates', 'source_logical_steps'):
                changed = deepcopy(value)
                changed[mutation] = 1
                with self.assertRaises(ValueError):
                    assets_module.check_prepared_assets(changed)

    def test_cheap_check_rejects_crossed_weights_account_and_spec(self):
        with self.fake_assets() as value:
            variants = []
            changed = deepcopy(value); changed['weights']['frozen']['sha256'] = assets_module.SELECTED_SHA; variants.append(changed)
            changed = deepcopy(value); changed['source_attempt']['binding']['config_sha256'] = '0'*64; variants.append(changed)
            changed = deepcopy(value); changed['model_spec']['w0']['shape'] = [3]; variants.append(changed)
            changed = deepcopy(value); changed['selection_boundary']['spent_train_examples'] = 499; variants.append(changed)
            for changed in variants:
                with self.assertRaises(ValueError):assets_module.check_prepared_assets(changed)

    def test_model_only_selected_drops_container_without_optimizer_rng_or_clone(self):
        with self.fake_assets() as value:
            before = deepcopy(value); payload = self.payload(value); marker = TorchMarker(payload)
            with patch.object(assets_module, '_file_identity', return_value=value['weights']['adapted']):
                state = assets_module.load_model_only_state('adapted', value, torch_module=marker)
            self.assertEqual(len(state), 298)
            self.assertEqual(payload, {})
            self.assertEqual(value, before)
            self.assertEqual(len(marker.calls), 1)

    def test_changed_file_rejected_before_torch_load(self):
        with self.fake_assets() as value:
            marker = TorchMarker(self.payload(value)); identity = dict(value['weights']['adapted'], sha256='0'*64)
            with patch.object(assets_module, '_file_identity', return_value=identity), self.assertRaises(ValueError):
                assets_module.load_model_only_state('adapted', value, torch_module=marker)
            self.assertEqual(marker.calls, [])

    def test_loaded_source_identity_rejected_without_restoration(self):
        with self.fake_assets() as value:
            for name, bad in (('logical_steps', 499), ('attempt', {}), ('config_identity', {}), ('budget', {}), ('extra_state', {})):
                with self.subTest(field=name):
                    payload = self.payload(value); payload[name] = bad; marker = TorchMarker(payload)
                    with patch.object(assets_module, '_file_identity', return_value=value['weights']['adapted']), self.assertRaises(ValueError):
                        assets_module.load_model_only_state('adapted', value, torch_module=marker)
                    self.assertEqual(payload, {})

    def test_keys_shapes_dtypes_finite_device_grad_rejected(self):
        with self.fake_assets() as value:
            bad_tensors = [TensorMarker(shape=(3,)), TensorMarker(dtype='torch.float64'), TensorMarker(finite=False),
                           TensorMarker(device='cuda'), TensorMarker(grad=True), object()]
            for tensor in bad_tensors:
                payload = self.payload(value); payload['model']['w0'] = tensor
                with patch.object(assets_module, '_file_identity', return_value=value['weights']['adapted']), self.assertRaises(ValueError):
                    assets_module.load_model_only_state('adapted', value, torch_module=TorchMarker(payload))
            payload = self.payload(value); del payload['model']['w0']
            with patch.object(assets_module, '_file_identity', return_value=value['weights']['adapted']), self.assertRaises(ValueError):
                assets_module.load_model_only_state('adapted', value, torch_module=TorchMarker(payload))

    def test_frozen_direct_state_uses_same_strict_spec(self):
        with self.fake_assets() as value:
            state = {name: TensorMarker() for name in value['model_spec']}; marker = TorchMarker(state)
            with patch.object(assets_module, '_file_identity', return_value=value['weights']['frozen']):
                result = assets_module.load_model_only_state('frozen', value, torch_module=marker)
            self.assertIs(result, state)

    def test_verifier_rejects_wrong_initial_hash_without_reading_metadata(self):
        with patch.object(assets_module, '_file_identity', return_value=dict(path='fake', bytes=35043916, sha256='0'*64)), \
                patch.object(Path, 'read_bytes', side_effect=AssertionError('Metadata must not be read after wrong weight')), \
                self.assertRaises(ValueError):
            assets_module.verify_evaluation_assets()

    def journal(self, corrupt_index=False):
        a = assets_module; rows = []; previous = '0'*64
        def event(kind, data):
            nonlocal previous
            sequence = len(rows)+1
            row = dict(sequence=sequence, previous_sha256=previous, at_unix=sequence, at_monotonic=sequence, kind=kind, data=data)
            line = a._canonical(row)+b'\n'; previous = a._sha(line); rows.append(line)
            return sequence
        event('created', {})
        entries = [dict(dataset_index=i) for i in range(500)]
        for i in range(500):
            operation = event('reserve_update', dict(batch_size=1, metadata=dict(training_schedule=a._schedule_identity(i),
                next_cursor=i+1, dataset_indices=[i+1 if corrupt_index and i == 249 else i])))
            event('complete_update', dict(operation_id=operation))
        boundary = dict(spent_reserved_updates=500, spent_train_examples=500)
        event('checkpoint_committed', dict(sha256=a.SELECTED_SHA, boundary=boundary))
        event('checkpoint_selected', dict(sha256=a.SELECTED_SHA))
        return b''.join(rows), entries, boundary

    def test_source500_journal_chain_and_exact_schedule_order(self):
        raw, entries, boundary = self.journal()
        result, sequence = assets_module._selection_from_journal(raw, entries)
        self.assertEqual((result, sequence), (boundary, 1003))
        bad, entries, _ = self.journal(corrupt_index=True)
        with self.assertRaises(ValueError):assets_module._selection_from_journal(bad, entries)
        with self.assertRaises(ValueError):assets_module._selection_from_journal(raw.replace(b'"complete_update"', b'"unknown_update"', 1), entries)


if __name__ == '__main__':
    unittest.main()
