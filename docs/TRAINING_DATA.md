# Prepare training batches

The collector stores episodes. The training loader selects complete history/future
windows from those episodes and feeds ordinary PyTorch batches. It does not train
a model, install packages, copy image data, or prepare a GPU machine.

## Commands

After collecting a run as described in `docs/COLLECTION.md`:

```bash
.venv/bin/python -B scripts/prepare_training_data.py --datasets artifacts/play-L-001 --output artifacts/training-L-001/catalog.json --history 2 --future 4 --batch-size 4
# Reuse the frozen catalog without collecting or indexing again:
.venv/bin/python -B scripts/prepare_training_data.py --catalog artifacts/training-L-001/catalog.json --history 2 --future 4 --split validation
```

Pass multiple directories after `--datasets` to combine runs. The command verifies
each source episode before writing the catalog, then checks at most two CPU
batches by default. `--max-batches` is capped at ten. It refuses to overwrite an
existing catalog. CLI help works without third-party packages.

The catalog contains relative dataset paths, source manifest hashes, episode IDs,
initial-scene hashes, and saved split membership. Keep it with ignored artifacts.
Moving the entire catalog/data directory structure preserves the relative paths;
moving only the catalog requires an explicit new catalog with the correct paths.

## Timing and returned values

For an episode containing T actions, choose history H and future F. A window with
anchor a contains:

```text
history observations:  a-H+1 ... a
future actions:        a     ... a+F-1
future target images:  a+1   ... a+F
```

Valid anchors run from H-1 through T-F. Each episode contributes
`max(0, T-H-F+2)` windows. There is no padding, repeated terminal frame, fabricated
action, or cross-episode window. Episodes too short for H/F contribute zero
windows. Overlapping windows are built when read, never saved as duplicate data.
History and future lengths are each limited to 1–32 for this bounded pilot.

| Sample field | Shape / meaning |
| --- | --- |
| `observations.image` | `(H,3,96,96)` float32 in `[0,1]` |
| `observations.agent_pos` | `(H,2)` float32, world pixels |
| `observations.goal_pose` | `(H,3)` float32, world pixels and radians |
| `actions` | `(F,2)` float32, absolute world position targets in pixels |
| `targets` | F post-action observations in the same image/position/goal format |
| `labels` | F rewards, terminal flags, source/collector cutoff flags, contact counts, coverage, and IoU |
| `episode_id`, `anchor`, `shape`, `control_dt` | Metadata for diagnostics and reproducibility |

Model observations contain image/proprioception/goal, not simulator object state.
Reward/contact/overlap labels are **privileged supervision** and are grouped
separately. Shapes and IDs are metadata, not an added geometry input to a visual
model. The training code must decide which labels it actually uses and report that.

Coordinates/actions retain their original units. This loader does not fit a
normalizer or apply a pretrained image encoder's transforms. A pretrained policy
requires its original histories, action horizon, image transforms, and normalizer;
these windows are not a claim of direct checkpoint compatibility. Any fitted
training statistics must use training episodes only.

Terminal flags are exposed unchanged, including GPC's possible pair of true
upstream flags and the collector's separate cutoff. No PPO bootstrap mask is
silently inferred; that policy-training decision remains explicit.

## Split and integrity rules

- Preserve each run's existing whole-episode splits; never redistribute episodes
  implicitly because a validation/test split is empty.
- Reject repeated dataset roots or duplicate trajectory IDs across runs.
- Reject an identical physical initial scene appearing in different splits even
  when its trajectory ID differs, for example after recollection with a new horizon.
- Require matching simulation/control rates and trajectory time intervals.
- Pin each source manifest's hash. Check file hashes and observation/state alignment
  when an episode is loaded; stale or corrupted input is rejected.

An initial-scene hash is an exact state comparison, not a near-duplicate scene
detector. Held-out shapes and policy-evaluation initial states still require
separate experimental manifests. Freezing a catalog does not itself make a tiny
dataset suitable for reporting model quality.

The batch report shows source episode/step counts and the fraction of steps with
reported contact for each split. This is not a measure of spatial contact coverage
or task success, and it does not multiply interaction counts by overlapping windows.

## Python API and memory

```python
from torch.utils.data import DataLoader
from tdyssey.training_data import WindowDataset, WindowSpec

dataset = WindowDataset("artifacts/training-L-001/catalog.json", "train", WindowSpec(2, 4))
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
```

One decoded episode is cached per dataset instance. Stored images stay uint8;
only the requested windows are converted to float32. Start with `num_workers=0`
to avoid extra worker copies. Each `.npz` archive must be decoded on a cache miss;
large/shuffled datasets may eventually need a different archive or sampler after
throughput measurement. No extra window or float32 dataset files are written.

```bash
python3 -B -m unittest discover -s tests -p 'test_training_data.py' -v
TDYSSEY_DATA_TESTS=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_training_data_runtime.py' -v
```

The installed tests use independent frame/action marker fixtures to test timing,
boundaries, split isolation, corruption detection, and real PyTorch collation.
Real collected T/I/L examples also passed the batch command; see
`docs/VALIDATION.md`. Model training and the original GPC data conversion remain
separate work.
