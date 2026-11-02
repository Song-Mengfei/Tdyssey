"""Saved-only exact-duration/contact/admission/report checks; no dynamics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tdyssey import strict_original_contact_report as report
from tdyssey.strict_original_contact_pilot import save_attempt


class SavedReportTest(unittest.TestCase):
    def setUp(self):
        fixture_root=report.ROOT/'artifacts/strict-original-contact-pilot-20261031/validation/report-fixtures'
        fixture_root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=fixture_root)
        if not Path(self.tmp.name).resolve().is_relative_to(fixture_root.resolve()):
            raise RuntimeError('Synthetic fixture must stay inside its verified workspace root')
        self.addCleanup(self.tmp.cleanup)
        self.output=Path(self.tmp.name)/'collection'
        self.directory=self.output/'trajectories/strict-O-0'
        self.directory.mkdir(parents=True)
        states=np.zeros((7,15),np.float64)
        states[:,:2]=[100,200]
        states[:,2]=256+np.arange(7)*4
        states[:,3]=256
        states[:,4]=np.arange(7)*.1
        self.valid=np.array([True]*5+[False])
        telemetry=[]
        for control,count in enumerate((10,10,10,10,10,2)):
            substeps=[]
            for micro in range(count):
                before=states[control]+(states[control+1]-states[control])*micro/10
                after=states[control]+(states[control+1]-states[control])*(micro+1)/10
                positive=micro<5
                contacts=[dict(kind='agent_object',phase='post_solve',total_impulse=[1 if positive else 0,0],
                    shapes_order=['agent','object'],points=[])]
                if not positive:contacts.append(dict(kind='object_wall',phase='post_solve',total_impulse=[0,0],shapes_order=['object','wall'],points=[]))
                substeps.append(dict(executed=True,dt=.01,pre_state=before.tolist(),post_state=after.tolist(),
                    penetrations=[0,0,0],contacts=contacts,valid=not(control==5 and micro==1)))
            telemetry.append(dict(substeps=substeps))
        # The partial terminal frame is the actual saved .02s state, not a .1s extrapolation.
        states[-1]=np.asarray(telemetry[-1]['substeps'][-1]['post_state'])
        class Lease:
            def write_bytes(self,path,payload):Path(path).write_bytes(payload)
        self.raw=self.directory/'raw.npz'
        raw=save_attempt(self.raw,states,np.full((6,2),256.),self.valid,
            np.full((7,3,96,96),255,np.uint8),telemetry,[dict(phase='push')]*6,reservation=Lease())
        self.metadata=dict(kind='strict-original-contact-attempt.v1',
            slot=dict(slot_id='strict-O-0',family_id='family-O-0',seed=123,letter='O',phase='formal',split='train_candidate',scenario='translation_nearCOM',location='center'),
            raw=raw,complete=False,failure='synthetic original invalid terminal',actual_valid_controls=5,
            protocol_sha256=report.PROTOCOL_SHA256,old_physics_data=False,
            physics_source_sha256=report.sha256(report.ROOT/'tdyssey/native_rgb_original_physics.py'),
            collector_source_sha256=report.sha256(report.ROOT/'tdyssey/strict_original_contact_pilot.py'),
            initial_identity=dict(protocol=report.PROTOCOL,protocol_sha256=report.PROTOCOL_SHA256,
                physics=dict(controller='original-upstream-kinematic-pd',kp=100,kv=20,control_dt=.1,substep_dt=.01,
                    substeps_per_control=10,space=dict(iterations=10,damping=0),agent=dict(body_type=1),object=dict(body_type=0))),
            contiguous_frames=True,valid_contiguous_control_ranges=[[0,5]],training_eligible=False,pending_user_approval=True)
        policy_path=self.output.parent/'policies/requested-target-v1.json'
        policy_path.parent.mkdir(parents=True,exist_ok=True)
        policy_path.write_text(json.dumps(dict(policy_execution_version='strict-original-requested-target-policy.v1')),encoding='utf8')
        self.metadata.update(policy_sha256=report.sha256(policy_path),
            policy_execution_version='strict-original-requested-target-policy.v1')
        (self.directory/'metadata.json').write_text(json.dumps(self.metadata),encoding='utf8')
        self.data=report.load_attempt(self.raw)

    def test_positive_impulse_actual_time_and_invalid_partial_are_separate(self):
        controls,windows,arcs=report.trajectory_metrics(self.metadata,self.data)
        self.assertEqual(len(controls),6)
        self.assertAlmostEqual(sum(r['seconds'] for r in controls),.52)
        self.assertAlmostEqual(sum(r['AO_pair_seconds'] for r in controls),.52)
        self.assertAlmostEqual(sum(r['AO_positive_seconds'] for r in controls),.27)
        self.assertAlmostEqual(sum(r['contact_moving_seconds'] for r in controls),.27)
        self.assertAlmostEqual(sum(r['wall_only_seconds'] for r in controls),0.)
        self.assertAlmostEqual(sum(r['wall_without_positive_AO_seconds'] for r in controls),.25)
        self.assertEqual(sum(w['legal'] and w['horizon']==1 for w in windows),5)
        self.assertEqual(sum(w['legal'] and w['horizon']==4 for w in windows),2)
        h4=next(w for w in windows if w['horizon']==4 and w['legal'])
        self.assertAlmostEqual(h4['translation_world_px'],16.)
        self.assertAlmostEqual(h4['translation_image_px'],3.)
        self.assertFalse(controls[-1]['h1_eligible'])
        summary=report.summarize(controls,windows,[self.metadata])
        self.assertEqual(summary['groups'][0]['actual_controls'],6)
        self.assertEqual(summary['groups'][0]['valid_controls'],5)
        self.assertEqual(summary['formal_supplement_trigger']['O']['valid_formal_H1_controls'],5)

    def test_trigger_uses_strict_half_all_valid_formal_h1_and_no_calibration(self):
        rows=[dict(phase='formal',letter='B',h1_eligible=True,translation_image_px=x,
            trajectory_id='one',rotation_deg=5.) for x in (.1,.2,1.,2.)]
        rows += [dict(phase='calibration-v1',letter='B',h1_eligible=True,translation_image_px=0.,trajectory_id='cal',rotation_deg=0.)]*20
        summary=report.summarize(rows,[],[])
        self.assertEqual(summary['formal_supplement_trigger']['B']['weak_fraction'],.5)
        self.assertFalse(summary['formal_supplement_trigger']['B']['trigger'])
        rows[2]['translation_image_px']=.4
        self.assertTrue(report.summarize(rows,[],[])['formal_supplement_trigger']['B']['trigger'])

    def test_actual_writer_full_guarded_report_membership_and_decoded_video(self):
        upstream=report.ROOT.parent/'Tdyssey/third_party/diffusion_policy'
        policy=report.ROOT/'artifacts/original-physics-contact-pilot-20261030/policies/resource-policy-v2.json'
        summary=report.build_report(self.output,resource_policy=policy,shared_roots=[self.output,upstream],videos=True)
        self.assertEqual(summary['candidate_admission']['admitted_attempts'][0]['valid_controls'],5)
        self.assertFalse(summary['training_authorized'])
        self.assertEqual(len(summary['videos']),1)
        self.assertEqual(summary['videos'][0]['frames'],7)
        self.assertEqual(summary['videos'][0]['fps'],10.)
        self.assertTrue(summary['videos'][0]['final_partial_control'])
        self.assertTrue((self.output/'REPORT.md').is_file())
        self.assertTrue((self.output/'report/resource-receipt.json').is_file())


class VideoSelectionTest(unittest.TestCase):
    def test_all_zero_motion_ties_do_not_create_positive_video_roles(self):
        sources={key:(dict(slot=dict(phase='calibration-v1'),complete=True),Path(key)) for key in ('B0','C0')}
        controls=[dict(trajectory_id=key,AO_positive_seconds=0.,boundary_seconds=0.) for key in sources]
        windows=[dict(trajectory_id=key,horizon=1,legal=True,translation_image_px=0.,rotation_deg=0.) for key in sources]
        choices,selection=report.select_video_examples(sources,controls,windows)
        self.assertFalse(selection['roles']['translation']['available'])
        self.assertFalse(selection['roles']['rotation']['available'])
        self.assertIn('No complete trajectory',selection['roles']['translation']['reason'])
        self.assertNotIn('translation',{role for role,_ in choices})
        self.assertNotIn('rotation',{role for role,_ in choices})
        self.assertTrue(selection['roles']['weak-motion']['available'])
        self.assertEqual(selection['roles']['weak-motion']['evidence']['H1_translation_max_image_px'],0.)

    def test_only_qualified_actual_legal_motion_is_ranked_despite_zero_medians(self):
        sources={key:(dict(slot=dict(phase='calibration-v1'),complete=key!='partial'),Path(key))
            for key in ('B0','C0','O0','partial')}
        controls=[dict(trajectory_id=key,AO_positive_seconds=0. if key!='C0' else .1,boundary_seconds=0.) for key in sources]
        windows=[]
        for key,translation,rotation,legal in (
                ('B0',0.,0.,True),('C0',.5,5.,True),('O0',100.,100.,False),('partial',500.,500.,True)):
            windows += [dict(trajectory_id=key,horizon=1,legal=True,translation_image_px=0.,rotation_deg=0.)]*10
            windows.append(dict(trajectory_id=key,horizon=1,legal=legal,
                translation_image_px=translation,rotation_deg=rotation))
        choices,selection=report.select_video_examples(sources,controls,windows)
        selected={role:item[0] for role,item in choices}
        self.assertEqual(selected['translation'],'C0')
        self.assertEqual(selected['rotation'],'C0')
        self.assertEqual(selection['roles']['translation']['evidence']['H1_translation_P50_image_px'],0.)
        self.assertEqual(selection['roles']['translation']['evidence']['H1_translation_max_image_px'],.5)
        self.assertEqual(selected['failure'],'partial')

    def test_v2_policy_binds_actual_module_cli_and_rejects_changed_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'collection';output.mkdir()
            policy_path=output.parent/'policies/revised.json';policy_path.parent.mkdir()
            implementation=report.ROOT/'tdyssey/strict_original_target_policy_v2.py'
            cli=report.ROOT/'scripts/strict_original_contact_pilot_v2.py'
            policy=dict(policy_execution_version='strict-original-requested-target-policy.v2',
                implementation_module='tdyssey.strict_original_target_policy_v2',
                implementation_source=str(implementation),implementation_source_sha256=report.sha256(implementation),
                cli_source=str(cli),cli_source_sha256=report.sha256(cli))
            policy_path.write_text(json.dumps(policy),encoding='utf8')
            metadata_path=output/'metadata.json'
            metadata_path.write_text(json.dumps(dict(policy_sha256=report.sha256(policy_path),
                policy_execution_version=policy['policy_execution_version'])),encoding='utf8')
            bindings,evidence=report.policy_source_bindings(output,[metadata_path])
            self.assertIn(implementation.resolve(),bindings)
            self.assertIn(cli.resolve(),bindings)
            self.assertIn(policy_path.resolve(),bindings)
            self.assertEqual(evidence[0]['implementation_module'],policy['implementation_module'])
            policy['implementation_source_sha256']='0'*64
            policy_path.write_text(json.dumps(policy),encoding='utf8')
            metadata_path.write_text(json.dumps(dict(policy_sha256=report.sha256(policy_path),
                policy_execution_version=policy['policy_execution_version'])),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'actual CLI source SHA differs'):
                report.policy_source_bindings(output,[metadata_path])


if __name__=='__main__':unittest.main()
