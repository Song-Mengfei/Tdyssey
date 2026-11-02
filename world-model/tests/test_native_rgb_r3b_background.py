"""CPU-only recovery boundaries; no experiment/model/renderer is constructed."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r3b_background_under_test',REPO/'scripts/native_rgb_r3b_background.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n',encoding='utf-8')


class RecoveryBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=REPO/'artifacts')
        self.root=Path(self.temp.name);self.stage=self.root/'r3b';self.stage.mkdir()
        self.attempt=self.stage/'attempts/approved-recovery';self.attempt.mkdir(parents=True)
        self.patch=patch.multiple(worker,ROOT=self.root,STAGE=self.stage);self.patch.start()

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def fixture(self):
        (self.root/'tdyssey').mkdir()
        (self.root/'tdyssey/native_rgb_r3a_data.py').write_bytes(b'original-frozen-loader')
        (self.root/'tdyssey/native_rgb_r3b_loader_recovery.py').write_bytes(b'focused-actual-repair')
        previous=self.stage/'background/completion.json'
        write(previous,dict(run_id='failed-run',completed=False,closed=True,formal_progress=dict(stage_attempts=0,stage_successes=0,formal_draws=0)))
        original=self.stage/'evaluation/spec.json';write(original,dict(selection='exact frozen windows and original seeds'))
        gate=dict(passed=True,run_id='real-cpu-diagnostic',root_cause_confirmed=True,
            root_cause='CPU fixture of an evidenced bootstrap boundary',repair='CPU fixture minimal repair',
            data_manifest_sha256=worker.DATA_SHA,loader_source_sha256=worker.sha(self.root/'tdyssey/native_rgb_r3a_data.py'),
            repair_source_sha256=worker.sha(self.root/'tdyssey/native_rgb_r3b_loader_recovery.py'),
            workers=2,prefetch_batches=2,model_loaded=False,cuda_initialized=False,
            launcher_exited_before_progress=True,verified_batches=4,verified_worker_pids=[101,202],
            dataset_contract=dict(physics_source_root=str(self.root/'existing-physics'),frame_cache_bytes=16*1024**2,state_cache_bytes=8*1024**2))
        gate_path=self.attempt/'diagnostics/gate.json';write(gate_path,gate)
        config=dict(stage_root=str(self.attempt),budget_stage_root=str(self.stage),recovery_authorized=True,run_id='new-approved-run',
            previous_failure=dict(path=str(previous),sha256=worker.sha(previous),run_id='failed-run'),
            frozen_validation_spec=dict(path=str(original),sha256=worker.sha(original)),
            loader_diagnostic_gate=dict(path=str(gate_path),sha256=worker.sha(gate_path),run_id=gate['run_id']),
            physics_source_root=str(self.root/'existing-physics'),loader_recovery_source_sha256=gate['repair_source_sha256'])
        return config,gate,gate_path

    def identity(self,config):
        with patch.object(worker,'FROZEN_SPEC_SHA',config['frozen_validation_spec']['sha256']):
            return worker.recovery_identity(config)

    def test_attempt_cannot_escape_or_enlarge_budget(self):
        config=dict(stage_root=str(self.root/'unrelated'),recovery_authorized=True)
        with self.assertRaisesRegex(ValueError,'below R3B/attempts'):worker.stage_paths(config)
        with self.assertRaisesRegex(ValueError,'budget root'):worker.stage_paths(dict(stage_root=str(self.attempt),budget_stage_root=str(self.attempt),recovery_authorized=True))
        with self.assertRaises(PermissionError):worker.stage_paths(dict(stage_root=str(self.attempt)))

    def test_actual_gate_requires_both_workers_progress_and_same_repair(self):
        config,gate,path=self.fixture();self.assertEqual(self.identity(config)['verified_batches'],4)
        for key,value in [('verified_worker_pids',[101,101]),('launcher_exited_before_progress',False),('repair_source_sha256','different'),('root_cause_confirmed',False)]:
            bad=deepcopy(gate);bad[key]=value;write(path,bad);config['loader_diagnostic_gate']['sha256']=worker.sha(path)
            with self.subTest(key=key),self.assertRaises(ValueError):self.identity(config)

    def test_consumed_predecessor_cannot_be_restarted_from_zero(self):
        config,gate,path=self.fixture();old=Path(config['previous_failure']['path']);value=worker.read(old)
        value['formal_progress']['stage_attempts']=1;write(old,value);config['previous_failure']['sha256']=worker.sha(old)
        with self.assertRaisesRegex(ValueError,'consumed prior epoch'):self.identity(config)

    def test_frozen_spec_is_byte_copy_and_reentry_cannot_replace(self):
        config,gate,path=self.fixture();identity=self.identity(config);source=Path(config['frozen_validation_spec']['path']);before=source.read_bytes()
        with patch.object(worker,'FROZEN_SPEC_SHA',worker.sha(source)):
            target=worker.restore_validation_spec(self.attempt,identity)
            self.assertEqual(target.read_bytes(),before);self.assertEqual(source.read_bytes(),before)
            target.write_bytes(b'changed attempt validation')
            with self.assertRaisesRegex(ValueError,'different validation'):worker.restore_validation_spec(self.attempt,identity)
        self.assertEqual(source.read_bytes(),before)

    def test_old_diagnostics_and_attempt_share_live_stage_cap(self):
        (self.stage/'old-failure.bin').write_bytes(b'x'*20)
        (self.attempt/'new.bin').write_bytes(b'y'*10)
        config=dict(stage_root=str(self.attempt),budget_stage_root=str(self.stage),recovery_authorized=True,
            shared_storage_roots=[str(self.stage)],stage_cap_bytes=40)
        initial=worker.storage(config);self.assertEqual(initial['stage_persistent_bytes'],30)
        check=worker.BoundedStorageCheck(config,initial)
        (self.stage/'old-diagnostic.bin').write_bytes(b'z'*20)
        with self.assertRaisesRegex(RuntimeError,'stage/disk shortage'):check(write_reserve_bytes=1)
        with self.assertRaisesRegex(RuntimeError,'storage shortage'):worker.storage(config)

    def test_callback_metadata_does_not_fabricate_formal_progress(self):
        status=worker.Status(self.attempt/'background','config',False,'run')
        status.update('loader_cpu_verification',dict(verified_batches=4,verified_worker_pids=[101,202]))
        self.assertEqual(status.value['formal']['stage_attempts'],0)
        self.assertEqual(status.value['formal']['formal_draws'],0)
        self.assertEqual(worker.read(self.attempt/'background/heartbeat.json')['run_id'],'run')

    def test_success_report_includes_evidenced_repair_prior_cost_and_actual_epoch(self):
        from tdyssey import native_rgb_r3b_report as report
        spec=dict(natural_selection=dict(kind='CPU-only synthetic report fixture'))
        write(self.attempt/'evaluation/spec.json',spec)
        spec_sha=worker.sha(self.attempt/'evaluation/spec.json')
        payloads={phase:dict(evaluation_complete=True,spec_sha256=spec_sha,rows=[])for phase in('t0','epoch1')}
        for cohort,count in(('natural',144),('stress',16),('old4',4)):
            for i in range(count):
                letter='TLOCXB'[i//24]if cohort=='natural'else'OCBX'[i%4]
                for horizon in(1,4):
                    for method in('copy_last','t0','epoch1'):
                        metrics={name:.1 for name in report.METRICS}
                        metrics.update(object_roi_mae={'copy_last':.3,'t0':.4,'epoch1':.2}[method],physical_measurement_status='ok')
                        row=dict(window_id=f'{cohort}:{i}',cohorts=[cohort],letter=letter,horizon=horizon,method=method,
                            shape_status='pass',status='completed',obvious_motion=i%3!=0,nearstatic_aux=i%3==0,metrics=metrics)
                        if method=='copy_last':
                            for payload in payloads.values():payload['rows'].append(row)
                        else:payloads['t0'if method=='t0'else'epoch1']['rows'].append(row)
        for phase,payload in payloads.items():write(self.attempt/f'evaluation/{phase}/metrics.json',payload)
        gate=self.attempt/'diagnostics/gate.json';write(gate,dict(CPU_only_fixture=True))
        recovery=dict(previous_failure=dict(run_id='old-failure'),root_cause='documented fixture bootstrap cause',repair='documented fixture minimal repair',
            loader_diagnostic_gate=dict(path=str(gate),sha256=worker.sha(gate)))
        write(self.attempt/'background/pipeline-state.json',dict(formal_progress=dict(stage_attempts=1499,formal_draws=23984,epoch_complete=True,stage_successes=1499,AMP_skips=0),
            recovery_context=recovery,model_accounting=dict(stages=[dict(observed_seconds=12.,engineering=dict(attempts=1),formal=dict(attempts=1499))])))
        (self.attempt/'run').mkdir();(self.attempt/'run/last.pt').write_bytes(b'CPU-fixture-not-a-checkpoint')
        with patch.object(report,'make_panels',return_value=[]):result=report.build_report(self.attempt)
        text=(self.attempt/'REPORT.md').read_text(encoding='utf-8')
        self.assertIn('documented fixture bootstrap cause',text)
        self.assertIn('documented fixture minimal repair',text)
        self.assertIn('33.047',text);self.assertIn('12.000',text)
        self.assertIn('1,499',text);self.assertIn('23,984',text)
        self.assertEqual(worker.read(self.attempt/'result.json')['recovery_context'],recovery)
        self.assertEqual(result['natural_H4_shape']['denominator'],144)


if __name__=='__main__':unittest.main()
