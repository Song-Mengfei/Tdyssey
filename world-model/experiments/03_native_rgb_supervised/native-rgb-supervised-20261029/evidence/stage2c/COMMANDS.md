# Stage2C recipe and actual execution evidence

Use the repository root and existing ../Tdyssey/.venv/Scripts/python.exe,
with SDL video/audio=dummy, CUDA_VISIBLE_DEVICES empty, HF_HUB_OFFLINE=1,
MPLBACKEND=Agg. No install or download.

```powershell
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B -W error::RuntimeWarning -m unittest tests.test_native_rgb_stage2c tests.test_native_rgb_stage2b tests.test_native_rgb_supervised_planned_attempts tests.test_native_rgb_reference tests.test_native_rgb_data tests.test_native_rgb_stage2c_indices -v
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c.py prepare --physics-root ../Tdyssey/third_party/diffusion_policy
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c.py freeze --physics-root ../Tdyssey/third_party/diffusion_policy
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c.py collect --physics-root ../Tdyssey/third_party/diffusion_policy --execute
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c.py audit --physics-root ../Tdyssey/third_party/diffusion_policy
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c_report.py --data artifacts/native-rgb-supervised-20261029/stage2c/data --output artifacts/native-rgb-supervised-20261029/stage2c/report-main-staging --physics-root ../Tdyssey/third_party/diffusion_policy --examples
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c_coverage.py --stage artifacts/native-rgb-supervised-20261029/stage2c
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B artifacts/native-rgb-supervised-20261029/stage2c/offline-repair/native_rgb_stage2c_coverage_v1_1.py --stage artifacts/native-rgb-supervised-20261029/stage2c
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B artifacts/native-rgb-supervised-20261029/stage2c/dataset-audit/run-cold-dataset-audit.py --execute-after-root-cold-confirmation
```

The recipe is not evidence that all commands ran. The actual ledgers, logs,
physical audit, report/index files and callback receipt establish completion.
The joint reference view is built only after passed collection/audit through
tdyssey.native_rgb_reference.build_joint_reference; four predeclared direct
Dataset items are checked by validate_cpu_samples without a training iterator.
Never replay prepare/freeze/collect over inspected outputs. Stop after callback;
stage3/model work is not authorized.

Actual postprocessing notes: the original coverage invocation failed its strict
snapshot-name guard before replay because the source-snapshot/ prefix was
omitted. Original source/log remain preserved. The separate offline v1_1 copy
corrects only that exact path and repository discovery for its new location;
hash checks are retained. Its single execution passed, adding0 physics controls.
See offline-repair/repair-record.json and source.diff.

The main report was run once into the fresh report-main-staging directory because
coverage had already created report/. Sixteen owned output files were moved into
report/ with identical SHA values after absolute-path checks; no existing coverage
file was changed and no stale staging paths remain. See report-relocation.json.
The four original examples remain intact; a separate1-row blocked-proxy truth
example uses an already-frozen extra index, without changing evaluation indices.
