# Try the world model yourself

This page describes the preserved **legacy T1** profile. The separate
[official original-colour entry](OFFICIAL_RAW_ALIGNMENT.md) uses raw official RGB
and explicit trained-weight provenance; it never treats checkpoint500 as that
model. Use `scripts/visualize_world_model.py` for the new H1 baseline.

中文简用：双击 `start_manual_demo.cmd`，在左边点目标，预测完成后会自动执行一次。
`H` 切换 1/4 个控制步；`C` 用另一个检查点比较刚才的同一份输入，不再推动物体。
`S` 保存完整案例和截图，`Esc` 退出。500 步权重仍是已知失败的检查点。

Double-click `start_manual_demo.cmd` in the repository folder on Windows. It uses
the existing environment in this checkout, or the sibling `Tdyssey/.venv`. It
does not install dependencies, download weights, train, or resume an experiment.
The first prediction loads the selected model; opening the window alone does not.

If starting from PowerShell with an existing environment:

```powershell
& ..\Tdyssey\.venv\Scripts\python.exe -B scripts\manual_world_model.py
```

## Controls

1. Click a target in the **left** real-simulator panel.
2. The model predicts, then the same action executes automatically, once.
   There is no Space confirmation. While busy, further clicks are ignored.
3. Compare the **left real outcome** with the **right prediction**. The real
   result updates in place. Both use the fixed T1 display palette; simulator
   observations and physics are unchanged. The interface is English-only,
   with no title banner, panel headings or footer instructions.

| Key/button | Effect |
| --- | --- |
| T / L | Change shape and reset the simple manual scene |
| 1 / 2 | Select initial GPC / the known degraded 500-update checkpoint |
| H | Switch 1-control-step / 4-control-step prediction |
| P | Repeat prediction and automatic execution for the selected target |
| R | Reset the current shape |
| C | Compare the other checkpoint on the last captured input, without another push |
| S | Save PNG, raw prediction NPY and a complete JSON/NPZ case bundle |
| V | Start/stop MP4 recording of the application window |
| Escape / window close | Exit; finish the recording and release the model |

Four-step mode repeats the selected absolute target for four control steps. It
feeds raw generated frames back to the native sampler; real future frames are
never used while predicting. Reset establishes four real observations through
three real hold actions. These initialization actions are included in counters.
Changing the target, shape, horizon or checkpoint invalidates the old preview.
After success, physical stepping stops until reset. If success happens before the
requested horizon, the comparison uses that actual horizon and the decoder result
for the originally requested later horizon is hidden.

Selecting a different checkpoint with 1/2 starts the next action from the current
physical state. Use C for a matched comparison: it reuses the archived four
real frames, three past actions, future target, full states and noise seed.
It displays the already executed real outcome and the other checkpoint's
prediction; no simulator restore or additional control occurs. The retained
execution's MAE names its original prediction role. A changed physical endpoint
is rejected. Four-step mode displays H4, or the matched earlier success prefix.

## What is being shown

The world model is GPC; the environment and T/L geometry are the official PushT
extension. These are not the other team member's model, whose weights are not
available here. The original four-frame input, three-Euler native sampler, fixed
T1 transformation and fixed image measurement are reused. T1 is applied once to
each real input/actual frame. Raw generated feedback is not transformed or clipped;
only displayed RGB is clipped to [0,1]. The physical goal is known context.

Initial weights: `artifacts/frozen-wm-assets/denoiser.pth`. Adapted weights:
`artifacts/gpc-supervised-adaptation-pilot03-20261026/adapted-latest.pt`.
Both use the existing hash/provenance-checked, model-only loader. No optimizer,
training RNG or old experiment account is restored. Only one model is active;
switching checkpoints releases the previous model before loading the next.

The 500-update checkpoint failed the formal evaluation. The demo deliberately
allows inspecting that failure. `coverage = unknown` means the existing decoder
rejected the predicted state; it is not zero coverage. Real coverage comes from
simulator geometry. Pixel MAE is descriptive, not a control-success metric.
Manual interaction is exploration, not an independent benchmark or training data.

## Files and runtime limits

Every launch creates a new ignored `artifacts/manual-world-model/<timestamp>/`
folder. It contains a small event log, requested screenshots/raw predictions and
recordings. These files are not connected to any training loader. Errors are kept
in `error.txt`; a model error stops model operations instead of retrying silently.
No old output or weight is overwritten. Model-hub access is disabled.

S writes a new `view-*-case/` with `case.json` and `arrays.npz`: raw/T1 real
history, full body/controller states, requested/executed actions, every raw
predicted horizon, actual prefix and checkpoint/sampling trace. Per-array and
payload identities are verified on load. The in-memory cache keeps eight recent
cases and at most three previews per case (latest per role plus the executed
reference). Saving never performs inference or simulation. Failed sampling or
rejected returned results preserve the input and returned prefix in `failed-*/`.
Call reservations remain spent even if a later call, decoder or acceptance fails;
logs distinguish reserved/returned calls from accepted forecasts.

The UI remains responsive while inference runs in one worker thread. Session time
is at most 30 minutes. Host memory is sampled once a second (2GiB working set,
3GiB private; at least 1.5GiB free RAM and 3GiB commit headroom). CUDA peaks are
checked after loading and sampling (512MiB allocated, 1GiB reserved). These are
sampled application limits, not a new OS Job framework or a per-step disk
heartbeat. A recording stops at approximately 128MiB. Closing waits up to five
seconds for inference cleanup, then exits this demo process if a native call is
still blocked. No scheduled task or unattended training loop is created.

## Verification on 2026-10-26

Three focused tests passed: real history/action alignment with no physics during
preview, stale-preview rejection, and early-success matching/termination.

One real desktop smoke run completed eight cases: both checkpoints, both shapes,
and horizons 1/4. It used two serial model loads, 20 predictions/60 Euler steps,
35 real control steps including 15 initialization holds, and zero training
updates. Preview did not move the simulator; execution used the same request.
The run completed in31.735s, worker stopped, exit0. Observed peaks were
2114818048B private,1277308928B working set,109831168B CUDA allocated and
136314880B CUDA reserved. Eight screenshots were retained and representative
initial/adapted L panels were visually inspected; the MP4's214 frames were decoded
successfully. This verifies the tool, not model quality.

Evidence: `artifacts/manual-world-model-verification-20261026/run-01/`.
This native run preceded the two-panel click-to-step UI change. Four additional
synthetic UI checks cover automatic execution exactly once, ignored busy clicks,
no stepping on prediction failure/exit/deadline, and two-panel drawing. No model
or physical run was added for this layout change while foundation work is active.
The native run printed a third-party cache-location warning; subsequent launch
configuration uses an offline cache inside each new manual output folder.

The Windows launcher was subsequently started in a persistent terminal session.
The new manual session recorded initial-weight predictions and matching physical
executions with the offline-cache configuration. It was left open for the user.
An additional OS-level automated mouse check could not run because both available
desktop automation interfaces failed during kernel setup with Windows error 3;
the eight-case smoke test above used the application's event handlers directly.

The stopped conditioning experiment remains stopped. Its pre-existing changes
are not part of this manual-demo implementation.

The subsequent [foundation repair](GPC_FOUNDATION_REPAIR.md) adds full case
capture, C comparison and partial-work accounting.21 synthetic checks pass;
one real core diagnostic reproduces the raw duplicate pusher and native parity.
Its outer receipt footer failed after the child completed; the failure is saved
without a real retry. It does not establish improved learned prediction or new
OS-level mouse automation. Current manual cases remain diagnostic-only.
