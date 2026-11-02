# Letter-pushing prediction app

## Model selection

The owner selected **HMF500** on 3 November 2026. Among 17 comparable saved evaluation entries on the same 24 O/C/B validation family/anchor windows, it has the highest macro AR4 gray IoU (0.798170). BASE128 and AR4-A128 repeat the same baseline result; these are not17 independent models. HMF3000 has the highest AR16 macro IoU (0.459269), with lower AR4 IoU (0.780247). This is repeated-validation selection, not a new independent benchmark or proof that every letter improves.

The app still predicts H1 or AR4. O/C/B errors remain; T/L/X were not revalidated in these HMF comparisons. No hidden per-letter checkpoint switching, output repair or changes to original physics/sampler are used. See [comparison evidence](../../docs/APP_MODEL_COMPARISON.json).

## Configure external assets

Copy `runtime.example.json` to `runtime.local.json` in this folder and set paths. Paths resolve relative to the config file. `asset_root` may point to a separate local experiment workspace; the model manifest supplies its relative checkpoint path. Alternatively, set each entry in `checkpoints` to your external HMF weight file. Configure `gpc_root` and `physics_root` to pinned upstream sources. Weight bytes/SHA and progress are checked against [model profiles](../../manifests/playground-models.json).

On the current development machine, the ignored config already references the original existing assets; no weights or environments were duplicated. The optional local `shared_storage_plan` retains the previous shared experiment budget when saving recordings. A clean source installation otherwise bounds the app's local outputs with the existing per-recording/ring/shared limits.

## Launch

With an existing compatible GPU environment:

```bash
python -X utf8 -B apps/r3b-playground/start.py --check
python -X utf8 -B apps/r3b-playground/start.py --open
```

Windows: double-click `Start.cmd`. It uses `TDYSSEY_PYTHON` if provided, then a repository `.venv`, then the existing sibling Tdyssey environment, otherwise `python` on PATH. It installs nothing.

For the longer-horizon model or a non-default port:

```bash
python -X utf8 -B apps/r3b-playground/start.py --model hmf3000 --port 8768 --open
```

Only the exact same repository, model and checkpoint identity may be reused from an existing service. An occupied port or another model is reported before loading a duplicate GPU model. Use the app's close button before switching models; no unknown process is terminated automatically. Startup logs stay in ignored `artifacts/r3b-playground/`, and the actual server error is echoed by the launcher.

`--check` and `test_runtime_config.py` do not load Torch or start a model/server. Real CUDA launch and prediction checks are separate runtime evidence. Historical native-GPC and experiment scripts are not run by opening this app.
