"""Release already authorized calibration only after actual evidence gates."""
import datetime,hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent
repo=p.parent.parent
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
read=lambda f:json.loads(Path(f).read_text(encoding='utf-8-sig'))
audit_path=p/'physics-regression/audit/AUDIT-v4.json'
browser_path=p/'ui-validation/actual-browser-check.json'
cpu_path=p/'ui-validation/CPU-v6-check.json'
audit,browser,cpu=map(read,(audit_path,browser_path,cpu_path))
assert audit['passed'] and browser['passed'] and browser['actual_browser'] and cpu['passed']
protocol='2ef7a9562c7451b502e23a5da08e5110349d25c8c2cfac37886bb990086b3d69'
physics=repo/'tdyssey/native_rgb_contact_safe_physics.py'
assert audit['protocol_sha256']==browser['protocol_sha256']==protocol
assert sha(physics)==audit['physics_source_sha256']=='768118e17ba69df4cca57c7041eaf84ecfd760466c1315451e134673585120bf'
for name,wanted in cpu['staged_source_SHA256_before'].items():
 assert sha(repo/'apps/r3b-playground'/name)==wanted,(name,'actual UI source differs from CPU tested source')
proof_path=p/'physics-design/contact-proof/exact-frame-proof.json'
proof=read(proof_path)
v6=[r for r in proof['selected_exact_contact_frames']if r['branch']=='AFTER V6']
assert len(v6)>=2 and all(r['AO_post_events']>0 and r['AO_impulse_norm_sum']>0 and r['OW_post_events']>0 and r['agent_object_overlap_world']<=1 for r in v6)
checkpoint=repo/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/run/last.pt'
assert sha(checkpoint)=='23cc0f847fe585f894a7528bd5db1ea3c8541228c6de4396c82f11ce373ce5e7'
paths=[physics,repo/'tdyssey/native_rgb_original_physics.py',repo/'tdyssey/original_contact_pilot.py',repo/'tdyssey/original_contact_pilot_report.py',repo/'scripts/original_contact_pilot.py',p/'collection/plan.json',p/'collection/rules.json',audit_path,browser_path,cpu_path,proof_path]
binding={str(f.resolve()):sha(f)for f in paths}
gate=dict(kind='original-kinematic-contact-physics-and-actual-UI-gate.v1',passed=True,
 released_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_authorized=False,
 protocol_sha256=protocol,physics_source_sha256=sha(physics),checkpoint_sha256=sha(checkpoint),
 independent_audit=str(audit_path),actual_browser_check=str(browser_path),CPU_interface_check=str(cpu_path),
 actual_contact_frame_proof=str(proof_path),rejected_contact_picture='B-v6-before-after-control32.png: separated at31/32, excluded as contact proof',
 candidate_cases=96,ordinary_matched_cases=24,max_overlap_OW_AO_AW_world_px=audit['max_candidate_penetrations_OW_AO_AW_world_px'],
 intervention_time_ratio=audit['candidate_intervention_time_ratio'],bindings=binding,
 action_contract='Absolute requested target retained as model action; constrained endpoint/velocity recorded separately, same factory UI/collector.',
 scope='18calibration-v1 first, bounded revised calibration if needed, then approved360candidate trajectories with finite predeclared supplement. Never train.',
 model_inference_test_calls=5,concurrent_live_model_usage_separate=True,
 limitations=['Finite96-case regression is not proof for all possible user commands.','94/96 final-hold cases actually separated; L/X long_fast still contact, not inertial drift.','Numerical/execution settings differ from original, although kinematic/damping0/object properties retained.','Public snapshots do not serialize native constraint cache/hidden bias velocities.','7499 remains trained on historical dynamics.'])
(p/'physics-gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
baseline=read(repo/'artifacts/r3b-playground-diagnostics-20261030/baseline-storage.json')
roots=list(dict.fromkeys([*baseline['roots'],str(p),str(repo/'apps/r3b-playground'),*[str(f)for f in paths if f.suffix=='.py'],str(repo/'scripts/original_contact_regression.py'),str(repo/'scripts/original_contact_pilot_overview.py')]))
(p/'storage-roots.json').write_text(json.dumps(dict(roots=roots,shared_cap_bytes=3*1024**3,atomic_reserve_bytes=256*1024**2),indent=2)+'\n',encoding='utf-8')
deployment_path=p/'ui-validation/deployment.json';deployment=read(deployment_path)
deployment['deployed_staged_files']=6
deployment['deployed_file_names']=list(read(p/'ui-staging.json')['ui-next'])
deployment_path.write_text(json.dumps(deployment,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in gate.items()if k not in('bindings','limitations')},ensure_ascii=False,indent=2))
