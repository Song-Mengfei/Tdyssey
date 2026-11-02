# Letter-pushing world-model app

The default is HMF500, selected for current H1/AR4 use. On the same 24 O/C/B
validation windows, its macro AR4 IoU is 0.798170. HMF3000 has macro AR4 IoU
0.780247 and the highest recorded AR16 macro IoU 0.459269 in that comparison.
T/L/X were not revalidated in the HMF screen, and C/long-horizon errors remain.
This does not establish reliable physics or downstream policy improvement.

## External assets and launch

Copy `runtime.example.json` to ignored `runtime.local.json` and configure
checkpoint, pinned GPC and Diffusion Policy source paths. Relative paths resolve
from the config file. Weights stay external and must match the selected profile's
size/SHA/progress in `manifests/playground-models.json`. Use the existing documented
GPU dependency profile; the launcher installs nothing.

```bash
python -X utf8 -B apps/r3b-playground/start.py --check
python -X utf8 -B apps/r3b-playground/start.py --open
# Explicit alternative; close the previous service before changing models.
python -X utf8 -B apps/r3b-playground/start.py --model hmf3000 --port 8768 --open
```

Windows users can run `Start.cmd` with a configured `TDYSSEY_PYTHON` or compatible
repository environment. Only the exact same repository/model/SHA service is
reused. A different application/occupied port is reported without killing it.

H1 uses four actual history frames; AR4 feeds raw generated frames into later
steps before advancing the unchanged original-physics comparison environment.
RGB display conversion does not modify feedback. Current physics is original
100Hz/solver10/PD/kinematic pusher/damping0. Interactive session/recording controls
are diagnostic only and do not write training data.

Temporary smoke-test scripts, old app backups, runtime receipts and recordings
are not repository content. Configuration tests need no Torch/GPU. The earlier
bounded five-frame GPU acceptance did not complete; CUDA loading/health alone
is not an accuracy benchmark. See [experiments](../../docs/WM_EXPERIMENTS.md).
