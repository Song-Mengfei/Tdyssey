# Stage2B commands

Working directory: repository root. Reuse `../Tdyssey/.venv/Scripts/python.exe`;
no dependency installation or asset download. Set SDL video/audio to `dummy`,
CUDA_VISIBLE_DEVICES empty, HF_HUB_OFFLINE=1 and MPLBACKEND=Agg.

```powershell
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B -W error::RuntimeWarning -m unittest tests.test_native_rgb_stage2b tests.test_native_rgb_stage2a tests.test_native_rgb_supervised_planned_attempts -v
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b.py prepare --physics-root ../Tdyssey/third_party/diffusion_policy
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b.py freeze --physics-root ../Tdyssey/third_party/diffusion_policy
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b.py collect --physics-root ../Tdyssey/third_party/diffusion_policy --execute
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b.py audit --physics-root ../Tdyssey/third_party/diffusion_policy
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b_report.py --root artifacts/native-rgb-supervised-20261029 --requests artifacts/native-rgb-supervised-20261029/stage2b/frozen-requests.json --output artifacts/native-rgb-supervised-20261029/stage2b/report --physics-root ../Tdyssey/third_party/diffusion_policy --examples
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b_coverage.py --stage artifacts/native-rgb-supervised-20261029/stage2b
```

This is an execution recipe, not an assertion that later commands have run.
Only the actual ledger, audits and result/callback files establish completion.
Never replay `prepare`, `freeze` or `collect` over inspected outputs. The
collector counts every charged/failed/partial attempt and stops at the first
physical, resource or program exception. No replacement slots are available.
