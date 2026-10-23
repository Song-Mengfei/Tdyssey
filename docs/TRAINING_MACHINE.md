# Move training to another GPU machine

This is the handoff for an author or their coding agent. The runnable policy path
uses **physical simulator rewards** with the official CNN and NFT/AdvantageFlow.
It does not yet train inside a reliable learned world model. The custom visual
world-model trainer is separate. Useful L adaptation has not been established.

## 1. Copy code and verified assets

Use the team's reviewed branch/commit in a separate checkout. Do not copy `.venv`,
caches or credentials: recreate the selected locked profile on the new machine.
Record `git rev-parse HEAD` in the experiment notes. Read [status](STATUS.md),
[forward training](FORWARD_POLICY_LEARNING.md) and [the learning plan](LEARNING_PLAN.md).

Policy RL generates its own interaction; it needs **no downloaded demonstration
dataset and no trained world model**. It needs the pinned upstream source and
the verified compact original policy, about 1.02 GiB. Follow the sparse source
checkout in [official baseline setup](OFFICIAL_BASELINE.md#setup-and-resources).
Copy `checkpoints/official-dp/ema_policy.pt` from a verified installation, or use
the documented verified download/conversion. Preserve the manifest's hash.

Prefer copying the compact asset to a CUDA 12.8 machine: the converter's exact
serialized hash was produced with PyTorch 2.5.1; reserialization with another
PyTorch version is not verified. The transferred compact file can be loaded with
the newer profile, subject to that machine's actual smoke check. Do not change
the accepted base hash to make a newly serialized file pass.

For world-model work, transfer the catalog **and every dataset directory it
references**, keeping their relative layout. Catalog paths are already relative;
their manifest hashes, episode IDs and frozen splits must stay unchanged. Verify
using `scripts/prepare_training_data.py --catalog PATH/catalog.json`. Do not
reassign splits to accommodate a move. Keep original and official-domain data separate.

For recovery, copy the complete run directory, including `last.pt`, `best.pt`,
`report.json`, `metrics.jsonl` and `tracking.json`. Source hashes must match the
training code. Old checkpoints from before a source change are **not** resumable
under different code; use them only through their matching version. Base weights
remain a separate required asset. Cross-GPU training need not be bit-identical.

## 2. Recreate one environment

Reference: Linux x86_64, Python 3.12.3, uv 0.13.0. These setup commands are for
the machine's owner to run after allocating its disk/compute budget.

| GPU/profile | Extras | State |
| --- | --- | --- |
| Existing CUDA 12.4-compatible setup | `cu124 dp tracking` | Local CUDA/training checks passed |
| Blackwell, including an RTX PRO 6000 Blackwell | `cu128 dp tracking` | Locked PyTorch 2.7.1 / torchvision 0.22.1; target hardware verification required |

Choose the profile for the **exact** GPU and installed driver, not its memory
size. PyTorch 2.7 added Blackwell support and CUDA 12.8 wheels; the version pair
is published in the [official release](https://pytorch.org/blog/pytorch-2-7/)
and [installation table](https://pytorch.org/get-started/previous-versions/).
Driver compatibility and the actual model still need checking on the target.

```bash
# On the new Blackwell machine; use cu124 instead for the older profile.
uv sync --locked --extra cu128 --extra dp --extra tracking --no-cache --link-mode copy --python /path/to/python3.12 --no-python-downloads
.venv/bin/python -B scripts/doctor.py --runtime --require-cuda --backend cu128 --dp --tracking
```

The `dp` extra supplies official-policy dependencies; always select exactly one
torch backend explicitly. `tracking` adds the pinned W&B package independently
of GPC. GPC is unnecessary for physical policy training. Use the same extras on
every later sync. No runtime command below installs packages or downloads assets.

Choose an output disk (the directory must exist), then run the read-only preflight:

```bash
.venv/bin/python -B scripts/check_training_setup.py --config configs/nft_extended.json --backend cu128 --output-root /path/to/training-storage --tracking
```

Preflight checks package pins, GPU/torchvision/convolution gradients, source/base
hashes and space for one configured run. It does not prove training fits or
reserve space. For several jobs, the launcher prints their summed output caps.
Allow additional space for the environment, the base checkpoint, transferred data
and W&B's separate spool. No measured footprint for cu128 is available yet.

## 3. Check one complete run before scaling

```bash
.venv/bin/python -B scripts/train_forward_policy.py --config configs/tracking_smoke.json --output /path/to/training-storage/check-001 --tracking offline --log-every 1
```

This bounded profile checks an actual CUDA update, per-letter evaluation, fixed
noise diagnostics, recovery and tracking. It is not a useful-training claim.
Inspect `report.json`: frozen parameters, checkpoint reload action parity and
physical counts must pass. Tied-score batches can correctly make no update.
Before dense full-network jobs, also run `configs/nft_full_network_smoke.json`
on the target GPU. The image encoder stays frozen in every current policy mode.

## 4. See live training

Every trainer writes flushed scalar events to `metrics.jsonl` and its tracking
status to `tracking.json`. Local logging is the default and needs no account.
In another terminal, use the Python already available on that machine:

```bash
python3 -B scripts/view_training.py --run /path/to/training-storage/check-001
```

Open **http://127.0.0.1:8765**. Select metrics in the four panels; they refresh
every five seconds. For a remote machine, forward its local port, for example
`ssh -L 8765:127.0.0.1:8765 USER@TRAINING_HOST`, and open that address locally.
The viewer serves only the chart and scalar events, without exposing run files.

For shared W&B charts, each author authenticates locally once with
`.venv/bin/wandb login`, or provides `WANDB_API_KEY` through their machine's secret
settings. Never put keys in configuration, commands committed to Git, or chat.
Choose an existing account/team with project access, then run:

```bash
.venv/bin/python -B scripts/train_forward_policy.py --config configs/nft_pilot.json --output /path/to/training-storage/nft-001 --tracking online --wandb-project tdyssey --wandb-entity TEAM_OR_ACCOUNT --run-group tl-pilot --run-name nft-seed42
```

The terminal and `tracking.json` identify the W&B run. Local metrics remain even
if W&B fails; `external_status=failed` makes that failure visible. The adapter
logs scalar metrics and declared configuration/provenance. It does not upload
model weights, raw trajectories, videos, source code or terminal output.

`--tracking offline` records W&B locally without an account or network. It does
not provide a cloud live view until uploaded; the local viewer works immediately.
After authenticating, optionally sync the printed offline-run directory with
`.venv/bin/wandb sync PATH/tracking/wandb/offline-run-...`. W&B's
[custom-axis guide](https://docs.wandb.ai/models/track/log/customize-logging-axes)
explains the optimizer-step axis used here.

| Chart | Meaning |
| --- | --- |
| `train/loss`, `train/update_mean_loss` | Actual stochastic NFT/AdvantageFlow objective; different contexts/noise across updates, not an improvement guarantee |
| `probe/SHAPE/fixed_noise_mse` | Same baseline scene/plan/noise at timesteps 0/25/50/99 across the run; never used for gradients or checkpoint selection |
| `probe/SHAPE/initial_policy_drift_mse` | Difference from the initial checkpoint on that diagnostic; change alone is not progress |
| `eval/SHAPE/success`, `final_coverage`, `peak_coverage` | Actual physical development results; check final alignment as well as a transient near miss |
| `rollout/SHAPE/contact_fraction`, `tied_group_fraction`, `score_spread` | Whether trials reach contact and supply distinct learning feedback; candidate success is training data, not validation |
| `runtime/*`, `progress/*` | Memory, elapsed time, update/minibatch counts, training and evaluation controls |

The fixed probe uses one **baseline-generated** plan/context per shape, not expert
action labels or a full validation benchmark. Its MSE is not the RL objective or
the probability of completing the task. Multi-scene physical evaluations remain
the evidence of useful adaptation. Empty/no-contact comparisons must stay visible.

## 5. Run longer comparisons

The extended presets are **proposed, unexecuted settings**: at most 500 updates,
one million training controls including replay, 48 hours and five development
scenes per shape. The first reached limit stops the job. Evaluation is additional
to the training-control cap and included in wall time. These are caps, not usage
predictions or evidence they are sufficient.

- `configs/nft_extended.json`: distributed rank-8 LoRA, minibatch 16, 2 GiB run cap.
- `configs/nft_full_network_extended.json`: dense action U-Net, 16 GiB run cap;
  needs the full-network smoke check and actual memory measurement first.
- Use the launcher's objective override for an otherwise matched AdvantageFlow run.
- For first learning experiments, prefer the smaller 20k-control pilot before
  committing the extended budget. More compute will not fix uniformly uninformative trials.

The launcher plans without writing files, importing torch or contacting W&B:

```bash
python3 -B scripts/launch_training.py --config configs/nft_extended.json --output-root /path/to/training-storage/tl-comparison-001 --seeds 42 43 44 --objectives nft advantage_flow --tracking online --wandb-entity TEAM_OR_ACCOUNT --run-group tl-comparison-001
```

Inspect the six jobs and summed budgets. To run that plan, use the installed
environment and append `--execute`:

```bash
.venv/bin/python -B scripts/launch_training.py --config configs/nft_extended.json --output-root /path/to/training-storage/tl-comparison-001 --seeds 42 43 44 --objectives nft advantage_flow --tracking online --wandb-entity TEAM_OR_ACCOUNT --run-group tl-comparison-001 --execute
```

It runs jobs **sequentially** on the selected visible GPU, writes fully resolved
configurations and `comparison.json`, and stops after a failure unless explicitly
given `--keep-going`. The output root must be new. Set `CUDA_VISIBLE_DEVICES`
locally to select a GPU. This is a single-GPU trainer, without distributed training
or parallel physics; a larger GPU does not remove sequential simulation/sampling
cost. Full precision and the original 100-step sampler remain the checked behavior.

Do not compare different methods by their loss values: their objectives differ.
Compare per-shape physical outcomes at matched interaction budgets, with the same
training seeds, development scenes and reward/control protocol. Keep fresh final
test scenes out of tuning. Training all letters does not establish unseen-letter
generalization. Use `nft_multishape.json` only after the T/L check.

## 6. Recovery, selection and storage

Forward training writes an atomic `last.pt` after each complete update and before
periodic evaluation. `eval_every=25` enables intermediate physical checks in the
extended presets; older profiles default to before/after evaluation. The best
**trained** development checkpoint is `best.pt`, selected by macro-average
per-shape success, then final coverage. Baseline results are logged separately;
“best trained” can still be worse than the frozen baseline. Save the entire run
for interpretation, not only the best weight file.

Only the most recent `report_detail_updates` retain full candidate/minibatch
traces. Every update retains scalar summaries; active episode prefixes remain
intact for physical replay. `0` retains no detailed update traces. Raw artifacts
stay ignored. Local scalar events default to a separate 16 MiB cap and mark
`local_events_capped` visibly; the W&B spool has its own disk use and is not inside
the trainer's top-level output cap. Periodically inspect disk use for long jobs.

To extend a saved forward run, copy its resolved configuration, increase only
`updates`, `max_train_controls`, `seconds` and/or `max_bytes`, then:

```bash
.venv/bin/python -B scripts/train_forward_policy.py --config PATH/extended-resume.json --resume PATH/old-run/last.pt --output /path/to/training-storage/resumed-001 --tracking online --wandb-entity TEAM_OR_ACCOUNT --run-group tl-comparison-001
```

Each recovery starts a **new** tracking run, retaining cumulative optimizer/control
axes and checkpoint lineage. It does not overwrite/reuse an external run's history
after rolling back an unfinished update. `best.pt` and its selection score carry
forward from the parent; copy the matching file beside `last.pt`. The original
frozen baseline remains in the lineage report.
Ctrl-C/SIGTERM writes a failure report; recovery uses the last completed boundary.
A hard kill may leave unreported work, which is explicitly flagged in cost audits.

## World-model training uses the same logging

The custom visual CNN accepts mixed official-shape catalogs; it is not the
released GPC adaptation trainer. It still predicts only horizons 1–4 controls,
and its physical action-ranking/contact gate is not passed. Longer training is
not permission to trust imagined rewards.

```bash
.venv/bin/python -B scripts/train_world_model.py --catalog PATH/catalog.json --output /path/to/training-storage/world-model-001 --epochs 100 --batch-size 32 --max-steps 100000 --seconds 86400 --max-bytes 268435456 --checkpoint-every 250 --tracking online --wandb-entity TEAM_OR_ACCOUNT --run-group dynamics-001
```

It logs each supervised loss component and validation motion/contact/reward
errors by horizon/contact group, leaving the model-test split unused. `last.pt`
saves model/optimizer/RNG and the next training-window cursor every configured
interval and at validation boundaries; `best.pt` remains the validation-selected
inference model. Resume to a **new** output using `--resume OLD/last.pt`; copy the
matching `best.pt` beside it if one existed. Keep the same catalog, sources,
batch size and training settings. Only epoch/step/time/byte caps may change;
the new step/epoch caps cannot precede the saved boundary.
Partial epochs continue at their saved cursor. Work after the checkpoint may be
repeated; the sidecar audit records discarded work or incomplete accounting.
RNG and data order are restored, but ordinary CUDA resizing/backward operations
need not produce bit-identical training weights across executions or machines.
Do not use `best.pt` as an optimizer recovery checkpoint.

Tail-PPO also accepts the common tracking flags and logs before/after physical
metrics, policy/value losses and clipping diagnostics. Its older trainer still
has no automatic recovery or periodic physical evaluation; use the forward path
for recoverable long runs until that separate trainer is extended.
