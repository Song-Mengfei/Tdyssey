"""Stdlib-only frozen identity and lifecycle tests; zero processes/physics/model."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,importlib.util,json,sys,tempfile,unittest

PHASE=Path(__file__).resolve().parents[2]/'background/phase.py'
spec=importlib.util.spec_from_file_location('strict_original_phase_audit',PHASE)
phase=importlib.util.module_from_spec(spec);spec.loader.exec_module(phase)

class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'strict-original-contact-pilot-20261031'
        self.runtime=self.root/'background/calibration-v1';self.runtime.mkdir(parents=True)
        self.repo=Path(self.temp.name)/'repo';(self.repo/'scripts').mkdir(parents=True)
        self.collector=self.repo/'scripts/strict_original_contact_pilot.py';self.collector.write_text('# fixture only\n')
        self.binding=self.root/'binding.txt';self.binding.write_text('frozen')
        self.config=self.runtime/'runtime-config.json'
        for rel in ('tdyssey/strict_original_contact_pilot.py','tdyssey/native_rgb_original_physics.py','tdyssey/native_rgb_data_eligibility.py','manifests/official_dp.json'):
            target=self.repo/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('{}'if target.suffix=='.json'else'# fixture only\n')
        for rel in ('plan.json','policy.json','resources.json','storage-roots.json'):(self.root/rel).write_text('{}')
        (self.root/'policy.json').write_text(json.dumps(dict(policy_execution_version='strict-original-requested-target-policy.v1')))
        self.c=dict(kind='strict-original-cpu-phase.v1',released=True,artifact_root=str(self.root),run_id='SOPC-audit-fixture',training_authorized=False,
            model_calls=0,optimizer_updates=0,auto_retry=False,phase='calibration-v1',physics_settings=dict(physics_hz=100,dt=.01,iterations=10,control_hz=10,kp=100,kv=20,damping=0,pusher='KINEMATIC'),
            python_executable=sys.executable,repo=str(self.repo),runtime=str(self.runtime),run_label='calibration-v1',letters='TLOCXB',expected_attempts=18,
            protocol_sha256='1'*64,collection=str(self.root/'collection'),plan=str(self.root/'plan.json'),policy=str(self.root/'policy.json'),physics_gate=str(self.root/'gate.json'),source_root=str(self.root/'upstream'),
            collector_progress=str(self.root/'collection/runs/calibration-v1/progress.json'),collector_completion=str(self.root/'collection/runs/calibration-v1/completion.json'),stop_file=str(self.root/'collection/runs/calibration-v1/STOP.json'),resource_policy=str(self.root/'resources.json'),storage_roots=str(self.root/'storage-roots.json'))
        Path(self.c['collector_progress']).parent.mkdir(parents=True)
        self.gate=dict(passed=True,strict_original_restored_verified=True,training_authorized=False,phase=self.c['phase'],run_id=self.c['run_id'],protocol_sha256=self.c['protocol_sha256'],plan_sha256=phase.sha(self.c['plan']),policy_sha256=phase.sha(self.c['policy']))
        Path(self.c['physics_gate']).write_text(json.dumps(self.gate))
        required=[PHASE,self.binding,self.collector,*[self.repo/p for p in ('tdyssey/strict_original_contact_pilot.py','tdyssey/native_rgb_original_physics.py','tdyssey/native_rgb_data_eligibility.py','manifests/official_dp.json')],*[Path(self.c[k])for k in ('plan','physics_gate','policy','resource_policy','storage_roots')]]
        self.c['bindings']=[dict(path=str(p),sha256=phase.sha(p))for p in required]
        self.c['collector_command']=phase.collector_command(self.c,'{CONFIG_SHA}')
        self.save()
    def save(self):self.config.write_text(json.dumps(self.c));self.digest=phase.sha(self.config)
    def verify(self):return phase.validate(self.config,self.digest)

class ValidationTests(Fixture):
    def test_canonical_frozen_command_schema(self):
        self.assertEqual(self.verify()[0]['run_id'],self.c['run_id'])
        for rel in ('tdyssey/strict_original_target_policy_v2.py','scripts/strict_original_contact_pilot_v2.py'):
            target=self.repo/rel;target.write_text('# frozen v2 fixture only\n')
            self.c['bindings'].append(dict(path=str(target),sha256=phase.sha(target)))
        self.c.update(sampling_version='strict-original-requested-target-policy.v2',phase='calibration-v2')
        self.gate['phase']='calibration-v2'
        def refresh_policy(version):
            Path(self.c['policy']).write_text(json.dumps(dict(policy_execution_version=version)))
            self.gate['policy_sha256']=phase.sha(self.c['policy'])
            Path(self.c['physics_gate']).write_text(json.dumps(self.gate))
            for row in self.c['bindings']:row['sha256']=phase.sha(row['path'])
            self.c['collector_command']=phase.collector_command(self.c,'{CONFIG_SHA}');self.save()
        refresh_policy('strict-original-requested-target-policy.v2')
        with self.subTest('Exact bound v2 requested-action CLI allowed'):
            verified,_=self.verify()
            self.assertEqual(Path(verified['collector_command'][4]).name,'strict_original_contact_pilot_v2.py')
        refresh_policy('strict-original-requested-target-policy.v1')
        with self.subTest('V2 config with mismatched frozen policy rejected'):
            with self.assertRaisesRegex(ValueError,'planner version differs'):self.verify()
    def test_wrong_runtime_config_hash_rejected(self):
        with self.assertRaises(ValueError):phase.validate(self.config,'0'*64)
    def test_changed_source_binding_rejected(self):
        self.binding.write_text('changed')
        with self.assertRaises(ValueError):self.verify()
    def test_wrong_phase_enum_rejected(self):
        self.c['phase']='training';self.save()
        with self.assertRaises(ValueError):self.verify()
    def test_gpu_scope_rejected(self):
        self.c['model_calls']=1;self.save()
        with self.assertRaises(ValueError):self.verify()
    def test_unreleased_rejected(self):
        self.c['released']=False;self.save()
        with self.assertRaises(ValueError):self.verify()
    def test_arbitrary_python_flags_rejected(self):
        self.c['collector_command'][1:4]=['-c','print(1)','#'];self.save()
        with self.assertRaises(ValueError):self.verify()
    def test_phase_value_does_not_accept_incidental_word(self):
        i=self.c['collector_command'].index('--phase');self.c['collector_command'][i+1]='formal';self.c['collector_command']+=['--run-label','calibration-v1'];self.save()
        with self.assertRaises(ValueError):self.verify()
    def test_output_escaping_new_root_rejected(self):
        self.c['collection']=str(self.root.parent/'old-v6-output');self.c['collector_command']=phase.collector_command(self.c,'{CONFIG_SHA}');self.save()
        with self.assertRaises(ValueError):self.verify()
    def test_missing_gate_and_policy_command_binding_rejected(self):
        i=self.c['collector_command'].index('--physics-gate');del self.c['collector_command'][i:i+2];self.save()
        with self.assertRaises(ValueError):self.verify()
    def test_actual_worker_source_required_in_bindings(self):
        self.c['bindings']=[row for row in self.c['bindings']if Path(row['path'])!=PHASE];self.save()
        with self.assertRaises(ValueError):self.verify()
    def test_detachment_cannot_be_removed_from_command(self):
        self.c['collector_command'].remove('--require-detached');self.save()
        with self.assertRaises(ValueError):self.verify()
    def test_expected_command_contains_no_old_factory_or_training(self):
        command=phase.collector_command(self.c,self.digest)
        self.assertIn('--require-detached',command);self.assertIn(self.digest,command)
        self.assertFalse(any('contact_safe' in s or 'train.py' in s for s in command))
        self.assertEqual(phase.environment(self.c)['CUDA_VISIBLE_DEVICES'],'')
    def test_check_mode_starts_no_subprocess(self):
        with patch.object(sys,'argv',[str(PHASE),'check','--config',str(self.config),'--config-sha',self.digest]),patch.object(phase.subprocess,'Popen')as popen:
            self.assertEqual(phase.main(),0)
        popen.assert_not_called()
    def test_existing_claim_prevents_relaunch(self):
        (self.runtime/'launch-claim.json').write_text('{}')
        with patch.object(sys,'argv',[str(PHASE),'launch','--config',str(self.config),'--config-sha',self.digest]),patch.object(phase.subprocess,'Popen')as popen:
            with self.assertRaises(RuntimeError):phase.main()
        popen.assert_not_called()

class FakeChild:
    pid=424242;_handle=123
    def __init__(self,polls=(0,)):self.polls=iter(polls);self.alive=True;self.terminated=False;self.returncode=None
    def poll(self):
        result=next(self.polls,None)
        if result is not None:self.alive=False;self.returncode=result
        return result
    def terminate(self):self.terminated=True;self.alive=False
    def wait(self,timeout=None):self.alive=False;return -1 if self.terminated else 0

class LifecycleTests(Fixture):
    def valid_receipt(self):
        return dict(run_id=self.c['run_id'],runtime_config_sha256=self.digest,phase=self.c['phase'],run_label=self.c['run_label'],
            physics_protocol_sha256=self.c['protocol_sha256'],plan_sha256=phase.sha(self.c['plan']),physics_gate_sha256=phase.sha(self.c['physics_gate']),policy_sha256=phase.sha(self.c['policy']),
            completed=True,failed=False,graceful_stop=False,actual_attempts=18,model_calls=0,optimizer_updates=0)
    def run_fixture(self,child,receipt=None,job=False):
        events=[]
        if receipt is not None:Path(self.c['collector_completion']).write_text(json.dumps(receipt))
        args=SimpleNamespace(config=str(self.config),config_sha=self.digest,launcher_pid=31337)
        def wait(pid):events.append('launcher-exit');return dict(pid=pid,exited=True)
        def popen(*args,**kwargs):events.append('collector-start');return child
        with patch.object(phase,'in_job',side_effect=[job,False]),patch.object(phase,'pe',return_value=sys.executable),patch.object(phase,'wait_exit',side_effect=wait),patch.object(phase,'storage',return_value={}),patch.object(phase,'validate',return_value=(self.c,self.runtime)),patch.object(phase.subprocess,'Popen',side_effect=popen),patch.object(phase.time,'sleep',side_effect=AssertionError('No real waits in fixture')):
            result=phase.run_worker(args,self.c,self.runtime)
        return result,events,phase.read(self.runtime/'completion.json')
    def test_launcher_exit_precedes_collector_start(self):
        result,events,terminal=self.run_fixture(FakeChild(),self.valid_receipt())
        self.assertLess(events.index('launcher-exit'),events.index('collector-start'))
        self.assertEqual(result,0);self.assertEqual(events.count('collector-start'),1);self.assertFalse(terminal['next_phase_released'])
    def test_job_bound_worker_starts_no_child(self):
        result,events,terminal=self.run_fixture(FakeChild(),job=True)
        self.assertEqual(result,1);self.assertNotIn('collector-start',events);self.assertFalse(terminal['next_phase_released'])
    def test_failed_partial_receipt_is_preserved_no_nextphase(self):
        receipt=dict(run_id=self.c['run_id'],phase=self.c['phase'],status='failed',actual_controls=3,actual_substeps=24,failure='fixture only')
        result,_,terminal=self.run_fixture(FakeChild((2,)),receipt)
        self.assertEqual(result,1);self.assertEqual(terminal['collector_completion'],receipt);self.assertFalse(terminal['auto_retry']);self.assertFalse(terminal['next_phase_released']);self.assertEqual(terminal['model_calls'],0)
    def test_wrong_run_completion_never_success(self):
        result,_,terminal=self.run_fixture(FakeChild(),dict(run_id='OLD-v6-run',phase='formal',status='complete'))
        self.assertEqual(result,1);self.assertIsNotNone(terminal['failure'])
    def test_exit0_with_failed_completion_never_success(self):
        result,_,terminal=self.run_fixture(FakeChild(),dict(run_id=self.c['run_id'],phase=self.c['phase'],status='failed',failure='invalid packet'))
        self.assertEqual(result,1);self.assertIsNotNone(terminal['failure'])
    def test_worker_exception_stops_owned_child_before_terminal(self):
        Path(self.c['collector_progress']).write_text('{bad-json')
        child=FakeChild((None,None,0));result,_,terminal=self.run_fixture(child)
        self.assertEqual(result,1);self.assertFalse(child.alive,'Owned collector remained alive after worker terminal failure');self.assertIsNotNone(terminal['failure'])
        self.assertTrue(Path(self.c['stop_file']).is_file())
    def test_malformed_completion_retains_failed_terminal(self):
        Path(self.c['collector_completion']).write_text('{bad-json')
        result,_,terminal=self.run_fixture(FakeChild())
        self.assertEqual(result,1);self.assertIsNotNone(terminal['failure']);self.assertTrue(Path(self.c['collector_completion']).is_file())

if __name__=='__main__':unittest.main(verbosity=2)
