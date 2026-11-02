# Resume after the user-requested independent collection

The user explicitly requested stopping this chat and closing Codex while the
finite R3A collection continues. The current stage is `background_running`, not
complete or waiting for R3B approval. Do not start a second collector.

Read these files first:

- `background/run/status.json`, `heartbeat.json`, `completion.json` and logs.
- `background/handoff-success.json`: real launcher-exit, live worker/collector
  job independence and progress proof; verification observed 629 successes.
- `collector-owner.lock`, `manifest.json`, `index.json`, `resource-ledger.json`.
- `source-identity-v3.json`, `background/handoff-receipt.json` and the independent
  handoff review. The original v2 owner exited; 587 successes were preserved.

Worker PID 15776 and collector PID 25876 were independently verified alive and
outside any Windows Job after launcher exit. Read current receipts rather than
assuming these PIDs remain valid. The actual production directory is
`background/run/`, not any `background-smoke-*` directory.

The old v2 was stopped at a durable write boundary for this requested handoff.
X074a0 had exactly one complete control and ten known substeps. Its partial data
and full costs remain; the frozen X074a1 replaced it. There was no new unstarted
reservation. The historical L000 unstarted reservation remains charged at one.
No successful trajectory was recollected. Source v3 changes ownership and
continuation selection only; physical control, fixed plan and 0.1 s half-life
are unchanged. Approved existing-Explorer desktop brokering, rather than direct
tool subprocess launch, achieved real job independence.

If the worker is still running, follow current progress without restarting it.
If it failed, first cold-audit the stopped state and all costs; do not erase owner
tokens/claims, regenerate scenes, resume a partially restored physical state or
launch a blind retry. Unknown physics advances must never be assumed zero.

When completion is successful, verify all 912 entries, closed flags, return codes
and zero model work. Then complete the already prepared CPU-only R3A handoff:

1. Run `r3a/audit-state-force-integrity.py` once, producing
   `numeric-integrity-review.json` (runner agent can do this).
2. Run `scripts/native_rgb_r3a_data_check.py --manifest r3a/manifest.json
   --source-root ../Tdyssey/third_party/diffusion_policy --out r3a/data-interface
   --batches 32` using full repository-relative artifact paths. It creates the
   compact legal/selected catalogue, full epoch audit and actual new-library
   two-worker throughput (data agent can do this).
3. Run `scripts/native_rgb_r3a_report.py` with the full manifest and
   `--output .../r3a/report --source-root ../Tdyssey/third_party/diffusion_policy
   --videos`. It performs no physics replay. Videos verify each recorded RGB
   frame hash; failure clips retain partial controls and show the last actual
   physics state separately. Then run `r3a/plot-motion.py` and visually inspect
   representative strips and the three figures.
4. After the actual data-interface gate, run `r3a/prepare-r3b-proposal.py`, then
   `scripts/native_rgb_r3b_proposal_check.py --config
   configs/native_rgb_r3b_proposal.json`. This binds actual data sizes and eight
   full new-library epochs. It is a CPU proposal check, not a GPU training loop.
5. Run `r3a/build-final-report.py`, verify reports/video/frame/CSV and source
   identities, fresh stage96MiB/shared3GiB budgets including futureR3B144MiB and
   atomic256MiB reserve. Update the report to include this background handoff
   and both preserved engineering interruptions. Do not repeat broad old audits.
6. Actually send the complete R3A report, data summaries and absolute local paths
   to parent chat `__LOCAL_CHAT__` (local), titled
   “制定 Tdyssey 世界模型实施计划”, under the saved standing direct human
   authorization. Save the real tool receipt, update STATUS/SUPERVISION/state,
   then STOP `waiting_parent` without polling.

Use the existing `../Tdyssey/.venv/Scripts/python.exe -X utf8 -B` for these CPU
commands. Root report helpers live inside
`artifacts/native-rgb-supervised-20261029/r3a/`. All current agents have prepared
their scripts; do not recreate work merely because their sessions are idle.

No GPU model/training/inference, held-out shapes, server work or automatic R3B is
authorized in R3A. Model time remains statistics only, with no human time cap.
The retained full checkpoint at lineage6000 remains the R3B starting point.
Fourteen obsolete historic binaries were separately removed under direct human
cleanup authorization; exact receipts and a new baseline remain, old audits are
unchanged. No formal model or original dataset was deleted.

While Codex is closed, the worker only writes local data/status/logs. It cannot
invoke MCP to message the parent. The final R3A callback is still required on
reopening; do not claim it was already sent offline.
