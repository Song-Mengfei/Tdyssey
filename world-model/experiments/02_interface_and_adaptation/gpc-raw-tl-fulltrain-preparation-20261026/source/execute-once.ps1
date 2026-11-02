param([Parameter(Mandatory=$true)][string]$ExpectedCommit)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$receipt=Join-Path $PSScriptRoot 'real-launch.json'
$actual=Join-Path $repo 'artifacts/gpc-raw-tl-fulltrain-20261026'
if((Test-Path -LiteralPath $receipt) -or (Test-Path -LiteralPath $actual)){throw 'The fixed real attempt is already spent'}
if(Get-Process | Where-Object { $_.ProcessName -like '*python*' -or $_.MainWindowTitle -like '*Tdyssey*' }){throw 'Active Python/demo process; no launch and no model budget spent'}
$python=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.tools/python/cpython-3.12.3-windows-x86_64-none/python.exe'
$env:PYTHONPATH=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.venv/Lib/site-packages'
$env:PYTHONDONTWRITEBYTECODE='1';$env:HF_HUB_OFFLINE='1';$env:TRANSFORMERS_OFFLINE='1';$env:HF_HOME=Join-Path $actual 'runtime-cache'
$env:HF_HUB_CACHE=Join-Path $env:HF_HOME 'hub';$env:PYGAME_HIDE_SUPPORT_PROMPT='1';$env:SDL_AUDIODRIVER='dummy';$env:SDL_VIDEODRIVER='dummy'
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
foreach($name in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'2','Process')}
$head=(git -C $repo rev-parse HEAD).Trim()
if($head -ne $ExpectedCommit -or $head.Length -ne 40 -or (git -C $repo status --porcelain)){throw 'Exact frozen clean source required'}
$configSha=(Get-FileHash -LiteralPath (Join-Path $repo 'configs/gpc_raw_tl_fulltrain.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if($configSha -ne 'fe10ddb999cc9b58811a1347832d7a16da0fcb5e75e514af0ab817b9fc413650'){throw 'Frozen config changed'}
$sources=[ordered]@{}
foreach($relative in @('tdyssey/gpc_raw_fulltrain_entry.py','tdyssey/gpc_raw_fulltrain_runtime.py','tdyssey/gpc_raw_fulltrain_worker.py','tdyssey/gpc_raw_fulltrain_data.py','tdyssey/gpc_raw_fulltrain_outputs.py','tdyssey/gpc_adaptation_guard.py','tdyssey/gpc_evaluation_status.py','tdyssey/gpc_supervised_adaptation.py','tdyssey/official_world_model.py','scripts/gpc_raw_tl_fulltrain.py','configs/gpc_raw_tl_fulltrain.json')){
    $sources[$relative]=(Get-FileHash -LiteralPath (Join-Path $repo $relative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
[ordered]@{commit=$head;config_sha256=$configSha;sources=$sources;upstream='a96146e568826ef46134f73c22fdf59a675e7264';attempts=1}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'source-freeze.json') -Encoding utf8
$arguments=@('-B','scripts/gpc_raw_tl_fulltrain.py','--execute','--authorization-id','20261026-user-raw-tl-fulltrain-01','--reviewed-commit',$head,'--authorized-config-sha256',$configSha,'--python',$python,'--gpc-root',(Join-Path (Split-Path -Parent $repo) 'Tdyssey/third_party/gpc_code'),'--checkpoint',(Join-Path $repo 'artifacts/frozen-wm-assets/denoiser.pth'))
$watch=[Diagnostics.Stopwatch]::StartNew();$reason=$null;$exitCode=$null;$process=$null
try{
    $process=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'real.stdout.txt') -RedirectStandardError (Join-Path $PSScriptRoot 'real.stderr.txt')
    $process.ProcessorAffinity=3
    while(-not $process.WaitForExit(200)){
        if($watch.Elapsed.TotalSeconds -ge 1770){$reason='External1770s cutoff leaves cleanup inside1800s';break}
    }
    if($reason){$process.Kill($true);if(-not $process.WaitForExit(5000)){throw 'Supervisor process cleanup uncertain'}}
    $process.WaitForExit();$exitCode=$process.ExitCode
    if(-not $reason -and $exitCode -ne 0){$reason='First guarded refusal/error; no retry'}
}finally{
    $watch.Stop()
    [ordered]@{source_commit=$head;config_sha256=$configSha;exit_code=$exitCode;reason=$reason;seconds=$watch.Elapsed.TotalSeconds;retry=$false;process_exited=if($process){$process.HasExited}else{$null}}|ConvertTo-Json|Set-Content -LiteralPath $receipt -Encoding utf8
}
Get-Content -LiteralPath $receipt
if($reason){exit 1}
