# Active task continuation

User explicitly authorized completing the fixed two-group physical comparison,
then delivering reviewable Chinese results; no further permission needed.
Do not add training, eval groups, retry failures, change cases/seeds or refill.

- Isolated repo: `__LOCAL_WORKSPACE__/Tdyssey-dppo-milestone-a`.
- Original source/goal baseline `6096e56`; tested evaluation implementation
  commit `1c4c66ab501d930ae02ffc95db5e9b2eaafd1047` (current source).
- Tests: installed full337:290pass47skip; final focused48pass; source full338:
  189pass149skip. Earlier source attempt failed old Windows sandbox TEMP setup;
  final rerun used workspace TEMP. No installs/downloads/push.
- Active `exec_command` session **31074** runs the self-monitoring evaluator.
  Wait via write_stdin, at most60seconds per tool; actual worker7996,
  redirector32552. Do not kill/restart unless current own guard does so.
- Account `c-tl-paired-dev-20261024`, separate evaluation-only authority
  `artifacts/policy-c-evaluation-registry`; run outputs
  `artifacts/milestone-c-evaluation-20261024`.
- Monitor/logs are in this meta directory's `monitor/`; outer command log is
  `evaluate-command.log`. The parent uses actual retained worker handle and
  kill-on-close Job Object, not the venv redirector PID. Model loads sequential.
- Start about2026-10-24 15:29:57SGT; wall7200seconds shared acrossbothgroups.
- Methods fixed base_training then delta_training, same stdfloor.01 training
  sampler, ordered T then L, scene100000..100004, seed5478/5479.20each/40total.
  max300controls percase; caps12000controls/1520plans/80resets/2groups,
  512MiBresults,64MiBhost telemetry. No train/optimizer operation is allowed.
- Runtime guards RAMavailable1GiB/commitheadroom2GiB/GPUfree1GiB;
  startup4/6/4GiB anddisk2GiB. No user programs closed.
- Config canonicalSHA0c682afbaf03360dc7af77c3c0917c3cfd120d5e7a8da10532c1d3d5251a8db3.
- Base existing original venv and checkpoint reused. Base pin
  37d661ce6a487905fa4ff81ab7d65bf1b56b1c06adf480a961f65338fa0e38a0.
- Final delta ONLY original C run checkpoint-0018-0052-audit.pt,
  16003916B SHA d3128c7fcfb1a23842b4863d96ae88b7f124e93d751b371b23696eb24d6703dd.
- Original C/B files and trees are hashed before/init/start/groups/after.
  Original C training remains2048controls52actor52critic18batches; do not open
  its ExperimentBudget (deadline includes downtime) or modify artifacts.

On terminal monitor/worker exit, invoke only stdlib offline helpers:
1. `python -S -B artifacts/milestone-c-evaluation-20261024-meta/reconcile_paired_evaluation.py`
   (exclusive output paired-reconciliation.json, original inventory/code/groups/
   accounting/physics/casejournal/isolation/modelrelease/comparison checks).
2. `python -S -B artifacts/milestone-c-evaluation-20261024-meta/audit_resources.py`
   (exclusive resources-review-final.json).
3. `python -S -B artifacts/milestone-c-evaluation-20261024-meta/export_comparison_tables.py --baseline artifacts/milestone-c-evaluation-20261024/base_training.json --delta artifacts/milestone-c-evaluation-20261024/delta_training.json --comparison artifacts/milestone-c-evaluation-20261024/paired-comparison.json --output-dir artifacts/milestone-c-evaluation-20261024-meta/tables`
   (requirescomplete, recomputescomparison, exports2shape/20case/5scene rows).

Helpers are prepared by idle agents but have NOT been run on live outputs.
All three can be used in independent read-only parallel calls after complete.
If incomplete, preserve recorded partial results and fixed denominator remains
undefined; no hidden retries/newaccount. Diagnose guard/error and stop honestly.

After successful checks, update docs/DPPO_C_EVALUATION.md (protocol currently
committed but outcome pending), STATUS.md and relevant runtime/learningplan links;
run source/message/diff checks and local docs commit only. No fullsuite repeat
needed for docs. Final Chinese result must give Tretention/Limprovement signals,
base vsdelta fixed10denom success, pairedWTL/coverage diffs, exactusage/resources,
no newtraining/originalledgerunchanged, file paths andcommit.5scenes x2correlated
seeds are development evidence; no significance/generalization claim. Preserve
localbasepublishedEMAidentity limitation.
