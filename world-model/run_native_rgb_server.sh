#!/usr/bin/env bash
# User-run Linux entry only. This file never connects to a remote machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-preflight}"; shift || true
output="artifacts/native-rgb-server"
config="configs/native_rgb_server.json"
physics_root="vendor/official"
gpc_root="vendor/gpc"
init_checkpoint="assets/denoiser.pth"
local_report="reports/local/local-report.json"
python_bin=".venv-native-rgb-server/bin/python"
setup=false; seed=5478; resume=""; checkpoint=""; split="seen_val"
while (($#)); do
  case "$1" in
    --setup) setup=true; shift ;;
    --output) output="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --physics-root) physics_root="$2"; shift 2 ;;
    --gpc-root) gpc_root="$2"; shift 2 ;;
    --init-checkpoint) init_checkpoint="$2"; shift 2 ;;
    --local-report) local_report="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --resume) resume="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --split) split="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$mode" in prepare|preflight|pilot|full|evaluate|resume) ;; *) echo 'Use prepare/preflight/pilot/full/evaluate/resume' >&2; exit 2 ;; esac
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 required' >&2; exit 2; }
mkdir -p "$output"
export HF_HOME="$output/runtime-cache/huggingface"
export HF_HUB_OFFLINE=1
bootstrap_python="$python_bin"
if $setup; then
  command -v python3.12 >/dev/null || { echo 'Install Python3.12 first' >&2; exit 2; }
  bootstrap_python=python3.12
fi
[[ -x "$bootstrap_python" ]] || command -v "$bootstrap_python" >/dev/null || { echo 'Supply --python or run prepare --setup' >&2; exit 2; }
# Verify uploaded sources and dependency files before using them for installation.
"$bootstrap_python" - <<'PY'
import hashlib,json,pathlib,sys
if sys.version_info[:2]!=(3,12):raise SystemExit('Python3.12 is required')
p=pathlib.Path('upload-manifest.json')
if p.exists():
    inventory=json.loads(p.read_text());root=pathlib.Path.cwd().resolve()
    for row in inventory['files']:
        target=(root/row['target']).resolve()
        if not target.is_relative_to(root):raise SystemExit('Unsafe upload inventory target')
        if not target.is_file() or target.stat().st_size!=row['bytes']:raise SystemExit('Upload size mismatch: '+row['target'])
        h=hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda:stream.read(1048576),b''):h.update(block)
        if h.hexdigest()!=row['sha256']:raise SystemExit('Upload hash mismatch: '+row['target'])
PY
if $setup; then
  if [[ ! -x .venv-native-rgb-server/bin/python ]]; then python3.12 -m venv .venv-native-rgb-server; fi
  python_bin=".venv-native-rgb-server/bin/python"
  "$python_bin" -m pip install --no-cache-dir -r requirements/native-rgb-server-torch.txt
  "$python_bin" -m pip install --no-cache-dir -r requirements/native-rgb-server.txt
  "$python_bin" -m pip check
  "$python_bin" -m pip freeze --all > "$output/installation-freeze.txt"
fi
[[ -x "$python_bin" ]] || { echo 'Supply an existing --python or run prepare --setup' >&2; exit 2; }
"$python_bin" - "$output" <<'PY'
import json,pathlib,platform,sys
if sys.version_info[:2]!=(3,12):raise SystemExit('Python3.12 is required')
pathlib.Path(sys.argv[1],'host.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform()),indent=2)+'\n')
PY
if [[ "$mode" == prepare ]]; then echo 'Prepared dependencies and verified uploaded bytes; no collection/training started.'; exit 0; fi
common=(--physics-root "$physics_root" --gpc-root "$gpc_root" --init-checkpoint "$init_checkpoint")
if [[ "$mode" == pilot ]]; then config=configs/native_rgb_local.json; fi
"$python_bin" -B scripts/native_rgb_world_model.py preflight --config "$config" --output "$output" "${common[@]}" --gpu
if [[ "$mode" == preflight ]]; then exit 0; fi
if [[ "$mode" == pilot ]]; then
  if [[ ! -f "$output/data/manifest.json" ]]; then
    if [[ ! -f "$output/physics-probe/report.json" ]]; then "$python_bin" -B scripts/native_rgb_world_model.py probe --config "$config" --output "$output" "${common[@]}"; fi
    "$python_bin" -B scripts/native_rgb_world_model.py collect --config "$config" --output "$output" "${common[@]}"
  fi
  "$python_bin" -B scripts/native_rgb_world_model.py audit --config "$config" --output "$output" "${common[@]}"
  "$python_bin" -B scripts/native_rgb_world_model.py engineering --config "$config" --output "$output" "${common[@]}"
  config="$output/pilot-effective.json"
  for stage in microfit train evaluate report; do "$python_bin" -B scripts/native_rgb_world_model.py "$stage" --config "$config" --output "$output" "${common[@]}"; done
  "$python_bin" - "$output/local-report.json" <<'PY'
import json,sys
if not json.load(open(sys.argv[1]))['qualified_for_server_scaleup']:raise SystemExit(3)
PY
  exit 0
fi
if [[ "$mode" == full ]]; then
  "$python_bin" - "$local_report" <<'PY'
import json,sys
evidence=json.load(open(sys.argv[1]))
if not evidence.get('physics_passed') or not evidence.get('qualified_for_server_scaleup'):
    raise SystemExit('Local data/independent RGB validation has not qualified scale-up. No override is provided.')
PY
  if [[ ! -f "$output/data/manifest.json" ]]; then
    if [[ ! -f "$output/physics-probe/report.json" ]]; then "$python_bin" -B scripts/native_rgb_world_model.py probe --config "$config" --output "$output" "${common[@]}"; fi
    "$python_bin" -B scripts/native_rgb_world_model.py collect --config "$config" --output "$output" "${common[@]}"
  fi
  "$python_bin" -B scripts/native_rgb_world_model.py audit --config "$config" --output "$output" "${common[@]}"
  "$python_bin" -B scripts/native_rgb_server_probe.py probe --gpc-root "$gpc_root" --checkpoint "$init_checkpoint" --output "$output/resource-probe" --data "$output/data" --physics-root "$physics_root"
  "$python_bin" - "$config" "$output" "$physics_root" "$gpc_root" "$init_checkpoint" <<'PY'
import json,pathlib,sys
from scripts.native_rgb_server_probe import run_jobs
config_file,out,physics,gpc,initial=sys.argv[1:]
root=pathlib.Path(out).resolve();cfg=json.load(open(config_file))
choice=json.load(open(root/'resource-probe/recommendation.json'))
jobs=[]
for seed in cfg['seeds']:
    folder=root/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
    effective=dict(cfg,seed=seed,microbatch=choice['microbatch'])
    effective_path=folder/'effective-config.json';effective_path.write_text(json.dumps(effective,indent=2)+'\n')
    common=['--config',str(effective_path),'--output',str(folder),'--data',str(root/'data'),
            '--physics-root',physics,'--gpc-root',gpc,'--init-checkpoint',initial]
    commands=[[sys.executable,'-B','scripts/native_rgb_world_model.py',stage,*common]
              for stage in ('preflight','engineering','microfit','train','evaluate','report')]
    commands.append([sys.executable,'-c',
        "import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))['qualified_for_server_scaleup'] else 3)",
        str(folder/'local-report.json')])
    jobs.append(dict(id=f'seed-{seed}',stages=commands,timeout_seconds=90000))
result=run_jobs(jobs,concurrency=choice['concurrency'],output=root/'jobs',job_timeout_seconds=90000)
if result.get('status') not in ('complete','passed'):raise SystemExit('An independent seed job failed; inspect jobs report')
PY
  exit 0
fi
seed_output="$output/seed-$seed"
effective_config="$seed_output/effective-config.json"
[[ -f "$effective_config" ]] || { echo 'Missing seed effective-config.json' >&2; exit 2; }
seed_common=(--config "$effective_config" --output "$seed_output" --data "$output/data" "${common[@]}")
if [[ "$mode" == resume ]]; then
  [[ -n "$resume" ]] || { echo 'Supply explicit --resume COMPLETE_CHECKPOINT' >&2; exit 2; }
  "$python_bin" -B scripts/native_rgb_world_model.py train "${seed_common[@]}" --resume "$resume"
  exit 0
fi
if [[ "$mode" == evaluate ]]; then
  [[ -n "$checkpoint" ]] || checkpoint="$seed_output/training/best.pt"
  if [[ "$split" != seen_val && ! -f "$seed_output/frozen-model.json" ]]; then
    "$python_bin" -B scripts/native_rgb_world_model.py freeze "${seed_common[@]}" --checkpoint "$checkpoint"
  fi
  "$python_bin" -B scripts/native_rgb_world_model.py evaluate "${seed_common[@]}" --checkpoint "$checkpoint" --split "$split"
fi
