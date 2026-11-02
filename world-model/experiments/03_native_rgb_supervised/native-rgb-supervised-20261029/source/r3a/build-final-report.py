"""Bind completed real R3A evidence into a concise report; never launch models."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

stage=Path(__file__).resolve().parent;repo=stage.parents[2];sys.path.insert(0,str(repo))
from tdyssey.native_rgb_stage_budget import native_storage_paths,storage_footprint
from tdyssey.native_rgb_training import atomic_json


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest=read(stage/'manifest.json');ledger=read(stage/'resource-ledger.json');index=read(stage/'index.json')
    if not manifest['collection_closed']or not ledger['closed']or len(manifest['entries'])!=912 or ledger['successful_scenes']!=912:
        raise ValueError('A complete recorded library is required; a plan or partial collection does not count as completion')
    physics=read(stage/'report/physics-audit.json');interface=read(stage/'data-interface/interface-gate.json')
    support=read(stage/'data-interface/support-exposure-summary.json')
    if support['source_gate_sha256']!=sha(stage/'data-interface/interface-gate.json'):
        raise ValueError('Motion-support summary does not bind the final actual data-interface gate')
    H1_train=support['horizons']['h1']['train']['ALL'];H1_val=support['horizons']['h1']['seen_val']['ALL']
    H1_X_val=support['horizons']['h1']['seen_val']['X']
    throughput=read(stage/'data-interface/two-worker-throughput/throughput.json')
    integrity=read(stage/'numeric-integrity-review.json')
    proposal=read(repo/'configs/native_rgb_r3b_proposal.json')
    proof=read(stage/'r3b-proposal-check.json')
    if not interface['passed']or not integrity['passed']or not proof['passed']:raise ValueError('Actual interface/integrity/proposal gates must pass')
    visual=read(stage/'report/videos/video-provenance.json')
    failed_visual=read(stage/'report/failure-videos/failure-video-provenance.json')
    time_bins=read(stage/'report/time-bin-motion-summary.json')['statistics']
    sources=[]
    for folder,pattern in [('tdyssey','*r3a*.py'),('scripts','*r3a*.py'),('tests','*r3a*.py'),('configs','*r3a*.json')]:
        for file in (repo/folder).glob(pattern):sources.append(dict(file=str(file),sha256=sha(file),bytes=file.stat().st_size))
    atomic_json(stage/'interface-source-identity-final.json',dict(kind='R3A-final-CPU-source-interface-identity.v1',files=sources))
    roots=native_storage_paths(repo,stage.parent,read(repo/'configs/native_rgb_r3a.json')['storage_roots'])
    roots.extend([repo/'configs/native_rgb_r3b_proposal.json',repo/'scripts/native_rgb_r3b_proposal_check.py'])
    current=storage_footprint(roots)
    new_sources=[Path(r['file'])for r in sources]+[repo/'configs/native_rgb_r3b_proposal.json',repo/'scripts/native_rgb_r3b_proposal_check.py']
    size=storage_footprint([stage]+new_sources)['bytes']
    reserve=144*1024**2+256*1024**2
    budget=dict(kind='R3A-final-full-pipeline-storage.v1',R3A_stage_plus_source_bytes=size,R3A_cap_bytes=96*1024**2,
        shared_current=current,remaining_R3B_reserved_bytes=144*1024**2,atomic_reserved_bytes=256*1024**2,
        projected_bytes=current['bytes']+reserve,shared_cap_bytes=3*1024**3,
        headroom_after_reserved_bytes=3*1024**3-current['bytes']-reserve,
        passed=size<=96*1024**2 and current['bytes']+reserve<=3*1024**3,
        cleanup_receipt_sha256=sha(stage/'authorized-cleanup-deletions.jsonl'),historic_audits_not_rewritten=True)
    if not budget['passed']:raise ValueError('Full pipeline storage cap/reserves not met: '+str(budget))
    atomic_json(stage/'final-storage.json',budget)
    attempts=[a for s in index['slots']for a in s['attempts']]
    failures=[a for a in attempts if a['status'].startswith('failed')]
    failure_counts=dict(Counter(a['status']for a in failures))
    if len(attempts)!=ledger['attempted_scenes']or len(failures)!=ledger['failed_scenes']:
        raise ValueError('Report attempt/failure counts differ from the retained index and ledger')
    handoff=read(stage/'background/handoff-receipt.json');background=read(stage/'background/run/completion.json')
    if (handoff.get('successful_scenes_preserved')!=587 or handoff.get('no_successful_trajectory_recollection')is not True
            or background.get('completed')is not True or background.get('collector_exit_code')!=0
            or background.get('worker_exit_code')!=0):raise ValueError('Completed audited background handoff evidence required')
    result=dict(kind='R3A-complete-real-independent-data-and-conditioning.v1',completed=True,
        collection_counts=physics['counts'],successful_trajectories=912,successful_controls=43776,
        charged_controls_including_failures=ledger['charged_controls'],completed_controls_including_failed_partials=ledger['completed_controls'],
        executed_substeps=ledger['executed_substeps'],reserved_not_started_controls=ledger.get('reserved_not_started_controls',0),
        actual_attempts=len(attempts),failure_count=len(failures),failure_counts_by_status=failure_counts,
        failures=[dict(trajectory_id=a['trajectory_id'],status=a['status'],reason=a.get('error'),
            charged_controls=a['charged_controls'],completed_controls=a['completed_controls'])for a in failures],
        engineering_interruption_trajectory_ids=[a['trajectory_id']for a in failures if a['status']=='failed_system'],
        background_handoff=dict(successful_scenes_preserved=587,remaining_successes_completed=325,
            receipt_sha256=sha(stage/'background/handoff-receipt.json'),completion_sha256=sha(stage/'background/run/completion.json'),
            collector_exit_code=0,worker_exit_code=0,offline_callback_attempted=background['callback_attempted']),
        TRAIN_legal_H1=interface['fulllegal_TRAIN_H1'],selected_pool=interface['selected_pool'],
        H1_motion_support=dict(complete_legal_TRAIN_motion_fraction=H1_train['moving_windows']/H1_train['legal_windows'],
            TRAIN_moving_trajectories=H1_train['moving_trajectory_support'],seen_val_moving_trajectories=H1_val['moving_trajectory_support'],
            TRAIN_zero_obvious_motion_trajectories=H1_train['trajectories_with_zero_moving_windows'],
            seen_val_zero_obvious_motion_trajectories=H1_val['trajectories_with_zero_moving_windows'],
            X_seen_val_both_windows=H1_X_val['category_counts']['both'],
            X_seen_val_both_trajectory_support=H1_X_val['category_trajectory_support']['both'],
            support_summary_sha256=sha(stage/'data-interface/support-exposure-summary.json')),
        actual_training_draws=0,model_calls=0,checkpoint_loads=0,optimizer_attempts=0,GPU_model_work=0,
        manifest_sha256=sha(stage/'manifest.json'),physics_audit_sha256=sha(stage/'report/physics-audit.json'),
        interface_gate_sha256=sha(stage/'data-interface/interface-gate.json'),integrity_sha256=sha(stage/'numeric-integrity-review.json'),
        proposed_R3B_config_sha256=sha(repo/'configs/native_rgb_r3b_proposal.json'),
        max_penetration_worldpx=physics['max_penetration_worldpx'],
        duplicates=physics['duplicate_state_or_action_sequences'],near_initial_scenes=physics['near_initial_scenes'],
        independent_motion_support_by_letter=physics['independent_support_by_letter'],
        observed_time_bin_motion=time_bins,
        representative_video_count=len(visual['records']),failure_video_count=len(failed_visual['records']),
        original_control_RGB_frames_verified_before_encoding=sum(r['original_collection_frame_hashes_pixel_exact']for r in visual['records'])+sum(r.get('original_collection_frame_hashes_pixel_exact',0)for r in failed_visual['records']),
        visual_review_file='report/qa/VISUAL_REVIEW.md',
        callback_required=True,callback_sent_at_report_build=False,
        actual_callback_receipt_file='callback-result.json',live_callback_status_file='../stage-state.json',
        next_action='Send the complete R3A evidence packet, save the receipt, then STOP waiting_parent. R3B training requires review of the recorded movement and videos.',
        storage=budget)
    atomic_json(stage/'r3a-result.json',result)
    lines=['# R3A: real independent trajectories and clean/weak conditioning','',
        'The new library contains 912 successful independent trajectories, each with 48 controls: 128 TRAIN and 24 new seen-validation trajectories per letter (T/L/O/C/X/B). Scenes, families, seeds, splits and up to two replacement candidates per slot were frozen before physics collection. No held-out shapes, dynamics lookahead, model construction, checkpoint loading, GPU inference or optimizer work were used.','',
        'The selected physical velocity half-life remains 0.1 s (Space.damping = 0.0009765625), with 10 Hz controls, 100 Hz physics, original masses and inertias, finite-force PD control, the original speed cap and walls. The successful trajectories contain 43,776 controls; failed attempts, partial trajectories and reservations incur additional costs.','',
        '| Item | Actual |','|---|---:|',
        f"| Successful trajectories | 912 |",f"| Actual attempts, including retained failures | {len(attempts)} |",f"| Charged controls, including failures/reservations | {ledger['charged_controls']} |",
        f"| Completed controls, including failed partials | {ledger['completed_controls']} |",f"| Actual known physics substeps | {ledger['executed_substeps']} |",
        f"| Failed attempts retained | {len(failures)} ({failure_counts.get('failed_system',0)} engineering, {failure_counts.get('failed_physics',0)} physical) |",f"| Maximum penetration among successful trajectories, world pixels | {physics['max_penetration_worldpx']:.6f} |",
        f"| Legal TRAIN H1 windows | {interface['fulllegal_TRAIN_H1']} |",f"| Selected distinct TRAIN H1 windows | {interface['selected_pool']['selected_windows']} |",
        f"| Complete legal TRAIN obvious-motion fraction | {H1_train['moving_windows']/H1_train['legal_windows']:.3%} |",
        f"| Selected distinct obvious-motion fraction | {interface['selected_pool']['selected_motion_fraction']:.3%} |",'','Each selected window appears once in the complete CPU audit epoch; actual optimizer training draws remain zero. There is no repeated quota for the rare category with both translation and rotation. The full legal and valid library remains referenced separately from the selected TRAIN pool. The obvious-motion rule is unchanged: area-centroid translation of at least 2 pixels in the 96-pixel image OR an absolute signed rotation of at least 5 degrees.','',
        f"The full legal TRAIN library has {H1_train['moving_windows']:,} obvious-motion windows among {H1_train['legal_windows']:,} ({H1_train['moving_windows']/H1_train['legal_windows']:.3%}). The selected pool keeps all of those and {interface['selected_pool']['selected_other']:,} distinct other windows, giving 50% motion by selection rather than by changing the recorded library. Small movements, static intervals, release phases and wall stalls remain real negative examples. The category other means below both fixed thresholds; it does not mean every such window is completely motionless.",'',
        '## Independent real movement support','',
        f"At least one obvious-motion H1 window is supported by {H1_train['moving_trajectory_support']}/{H1_train['legal_trajectory_support']} TRAIN trajectories and {H1_val['moving_trajectory_support']}/{H1_val['legal_trajectory_support']} new seen-validation trajectories. Conversely, {H1_train['trajectories_with_zero_moving_windows']} TRAIN and {H1_val['trajectories_with_zero_moving_windows']} validation trajectories have no H1 window crossing either threshold; those trajectories remain in the library. The rare TRAIN both category has {H1_train['category_counts']['both']} different windows from {H1_train['category_trajectory_support']['both']} trajectories. X seen-validation has only {H1_X_val['category_counts']['both']} both windows from {H1_X_val['category_trajectory_support']['both']} trajectory, so its joint translation-and-rotation support is weak and must not be described as broad independent coverage. See data-interface/support-exposure-summary.json.",'',
        'The table below combines TRAIN and new seen validation. Separate counts for each split are retained in report/physics-audit.json.','',
        '| Letter | Trajectories with H1 translation >= 2 pixels | With H1 rotation >= +5 degrees | With H1 rotation <= -5 degrees |','|---|---:|---:|---:|']
    for letter,v in physics['independent_support_by_letter'].items():lines.append(f"| {letter} | {v['translation_H1']} | {v['positive_H1']} | {v['negative_H1']} |")
    lines+=['','The complete H1/H4 distributions (zero fraction, P10, median, P90, P95 and maximum), per-trajectory and scenario measurements, 48-control time curves, actual contact runs and wall contacts are recorded in report/physics-audit.json and report/trajectory-audit.csv. These files also report post-contact local and world arcs, yaw, lever arms and normal/tangent projections of the executed target commands. Coverage gaps are disclosed: an unobserved arc or hole is neither covered nor proven unreachable. A finite sample does not establish coverage of every combination.','',
        'All five exact-signature duplicate checks found zero groups. One near-initial-scene pair remains within TRAIN (L013 and L121); there are zero declared cross-split near pairs. Both trajectories have independent frozen seeds and distinct state, action and RGB signatures. The similarity is disclosed rather than treated as extra independent coverage.','',
        f"Observed TRAIN H1 obvious-motion fractions are {time_bins['early_controls_3_to_21']['fraction']:.2%} for control anchors 3–21, {time_bins['middle_controls_22_to_29']['fraction']:.2%} for anchors 22–29 and {time_bins['late_controls_30_to_47']['fraction']:.2%} for anchors 30–47. These are actual time bins; release/recontact scenario timings vary. Later pushes retain substantial stalls and are less motion-rich than the first segment. See report/time-bin-motion-summary.json and the complete control-time curves.",'',
        'Observed exterior-ring arc counts are T55/60, L49/53, O45/56, C59/75, X48/84 and B49/60. Closed interior rings were not initial-contact targets in this exterior-approach collector: O0/37 and B0/19+0/19 interior arcs were observed. Those holes and missing exterior arcs are explicit coverage gaps, with no claim of tested physical impossibility or full Cartesian coverage. Actual local/world contact points, pose-bin diversity and lever/action distributions remain in the audit.','',
        '## Inputs and measured CPU contracts','',
        'All 1,325 original saved R2B frames render with identical pixels, covering all six letters, holes, rotations, contact and poses near walls. Complete saved float64 states with 15 components supply privileged rendering, motion-based sample selection and audit metadata. They are not network features. The training batch contains five RGB frames (four history frames and the H1 target) and four aligned executed actions, normalized once. Only the four past frames condition history; the fifth frame supplies the native noised target and residual objective. The target is never included in the history conditioner. No persistent RGB bank or library-wide float preload is created. Two workers reuse their renderers, with a bounded 16 MiB RGB cache and 8 MiB state cache per worker and at most two batches in flight. Actual throughput on the new library is recorded in data-interface/two-worker-throughput/throughput.json. The old-library benchmark is historical performance evidence only.','',
        'The native FP32 residual MSE and target-noise distribution remain unchanged. The implemented conditioning strategy assigns 50% of microbatches to clean history (sigma_cond=None); the remaining microbatches use per-sample log-uniform history noise in [0.002, 0.05], with zero history offset. Branch, target-noise and weak-history RNG streams are independent and resumable. Seven focused conditioning contracts and the data, sampler, geometry and I/O contracts passed on CPU. These checks do not constitute GPU training or demonstrate prediction quality.','',
        f"The actual new-library 512-sample, two-worker benchmark measured {throughput['samples_per_second_including_start']:.3f} samples/s including startup and {throughput['steady_samples_per_second_excluding_first_batch']:.3f} samples/s after the first batch. The first batch took {throughput['cold_start_to_first_batch_seconds']:.3f} seconds from startup. Each worker reused at most six renderers within its cache limits; maximum in-flight batches were two. Whole-process RSS and a full epoch wall time were not measured. The 512 CPU benchmark samples do not represent model training.",'',
        'Root inspected all 35 representative strips, the six last actual failed states and all three scientific figures. Before encoding, 1,715 representative frames and 116 failed-prefix control observations matched their original collection RGB hashes exactly. MP4 files are compressed previews; encoded MP4 pixels are not asserted lossless or used as training targets. Original PNG strips and on-demand RGB retain their original pixels. See report/qa/VISUAL_REVIEW.md.','',
        '## Failures, storage and next training proposal','',
        'Both engineering interruptions remain in the six failed attempts. First, a ledger write failed with WinError 5 before the next engine step, after T completed 48 controls and L000a0 completed 18. An independent audit established that the extra reservation never entered env.step. A versioned I/O-only repair added bounded retries when atomically replacing the same file. Successful T was never recollected; failed L000a0 was retained and its previously frozen a1 was used. The one reserved but unstarted control remains charged.','',
        'During the later user-requested handoff to an independent background process, X074a0 stopped at post-save ledger persistence with WinError 5 after exactly one completed control and ten known substeps. The old owner exited. Independent reconciliation retained its partial data and all costs, confirmed zero unknown engine advances and no new unstarted reservation, and preserved all 587 successes already collected. Version 3 changed ownership and continuation selection only; it used the frozen X074a1 instead of restoring or replaying the partial physical state. The independent worker completed the remaining 325 successes, with worker and collector exit codes both zero. It saved local evidence and made no offline chat callback. See background/handoff-receipt.json, background/handoff-independent-review.json and background/run/completion.json.','',
        'The other four failures exceeded the fixed two-world-pixel penetration guard and remain physical failures. Their prefixes, failed substeps, costs and videos are retained. The maximum penetration in the summary table covers successful trajectories only; it must not be read as the maximum across failed attempts. No successful trajectory was recollected and no charged work was refunded.','',
        'The planning chat separately deleted fourteen earlier failed or redundant binaries under direct human cleanup authorization. The exact deletion receipt, hashes and post-cleanup baseline are retained. Historical audits still describe their original state and were not rewritten. All formal models and original datasets remain.','',
        f"The R3A stage and its sources occupy {size:,} bytes of the 96 MiB allowance. Current shared storage is {current['bytes']:,} bytes. Reserving 144 MiB for R3B and 256 MiB for atomic writes leaves {budget['headroom_after_reserved_bytes']:,} bytes below the 3 GiB shared cap. See final-storage.json. Elapsed time is recorded as a statistic; there is no human-imposed model-time cap.",'',
        'The CPU-validated R3B proposal binds the new dataset, selected pool and cursor identity to the preserved full checkpoint at update 6000. Migration of the dataset and noise streams is explicit; optimizer and learning-rate choices remain subject to parent review. The proposal schedules eight complete epochs over the new library, with observations based on actual exposure. It does not use a 300-update trial as a verdict. Short batches receive weights based on their actual sample counts; AMP skips and successful updates are counted separately. No R3B model work has run, and the new data and conditioning have not yet been shown to improve model predictions.','',
        'The bound plan has 1,499 batches of 16 per epoch, 11,992 planned attempts and 191,872 new-library exposures over eight epochs. The lineage would reach 17,992 successful updates only if every new attempt succeeds; AMP skips are separate. The 16-family validation proposal contains six natural, six motion and four near-static cases; the static letters are B/C/T/T, with no claim of static coverage for every letter. GPU training integration and actual optimizer/scaler restoration are still pending R3B work, not CPU-validated accomplishments.','',
        '## Review files','',
        '- report/videos/index.html — recorded 49-frame clips spanning 0.0–4.8 s at 10 fps. These include the strongest available per-letter examples of translation, each rotation sign and boundary contact, plus fixed-seed random and least-motion trajectories. Unsupported categories are identified.',
        '- report/failure-videos/index.html — retained short failed attempts, no padding or physics replay.',
        '- report/physics-audit.json and report/trajectory-audit.csv — complete distributions and coverage of the recorded data.',
        '- report/motion-distributions.png, report/control-time-curves.png and report/independent-motion-support.png — complete-library scientific figures, including stalls and separate TRAIN/validation support.',
        '- data-interface/interface-gate.json and data-interface/window-catalogue.npz — full legal and selected counts, unique exposures in the complete CPU audit epoch and throughput with two workers.',
        '- data-interface/support-exposure-summary.json — full versus selected motion fractions, independent trajectory support, zero-motion-threshold trajectories and rare-category gaps.',
        '- conditioning-contracts.json; collector-v2-contracts.json; numeric-integrity-review.json; key-interface-review.json — focused CPU and runtime evidence.',
        '- background/handoff-receipt.json; background/handoff-independent-review.json; background/run/completion.json — both preserved engineering interruptions and the completed independent background continuation.',
        '- configs/native_rgb_r3b_proposal.json and r3b-proposal-check.json — the proposal bound to the actual new library and its CPU-only validation.',
        '', 'After sending the authorized callback and saving its receipt, STOP waiting_parent. Review of the recorded movement and videos must precede R3B training.']
    (stage/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    main_html='<!doctype html><meta charset="utf-8"><title>R3A: recorded independent data review</title><style>body{font:18px system-ui;max-width:1000px;margin:40px auto}li{margin:15px 0}</style><h1>R3A: 912 recorded independent trajectories</h1><p>Velocity half-life: 0.1 s. Each trajectory contains 48 controls. No R3B training has run.</p><ul><li><a href="videos/index.html">Recorded 10 Hz representative videos and frame strips</a></li><li><a href="failure-videos/index.html">Retained partial failures</a></li><li><a href="physics-audit.json">Complete motion, contact and boundary audit</a></li><li><a href="trajectory-audit.csv">Measurements for all 912 trajectories (CSV)</a></li><li><a href="../REPORT.md">Full stage report</a></li><li><a href="../data-interface/interface-gate.json">Data inputs, distinct sampling and throughput on the new library</a></li></ul>'
    main_html+='<h2>Complete recorded library</h2><p>These figures describe real recorded data, not model predictions. Dashed movement thresholds are unchanged.</p><img width="1000" src="motion-distributions.png"><img width="1000" src="independent-motion-support.png"><img width="1000" src="control-time-curves.png">'
    (stage/'report/index.html').write_text(main_html,encoding='utf-8')
    print(json.dumps(dict(report_ready=True,successful_trajectories=912,stage_bytes=size,shared_bytes=current['bytes'],training_draws=0)))


if __name__=='__main__':main()
