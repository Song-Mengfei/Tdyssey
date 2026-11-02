$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$receipt=Join-Path $PSScriptRoot 'real-launch.json'
$actual=Join-Path $repo 'artifacts/gpc-conditioning-contract-diagnostic-20261026/auth20261026-user-autonomous-gpc-conditioning-contract-diagnostic-02'
if((Test-Path -LiteralPath $receipt) -or (Test-Path -LiteralPath $actual)){throw 'This fixed attempt is already spent'}
if(Get-Process | Where-Object { $_.ProcessName -like '*python*' -or $_.MainWindowTitle -like '*Tdyssey*' }){throw 'Active Python/demo process; user GUI has priority. No launch.'}
$python=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.tools/python/cpython-3.12.3-windows-x86_64-none/python.exe'
$env:PYTHONPATH=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.venv/Lib/site-packages'
$env:PYTHONDONTWRITEBYTECODE='1';$env:HF_HUB_OFFLINE='1';$env:TRANSFORMERS_OFFLINE='1';$env:HF_HOME=Join-Path $actual 'runtime-cache'
$env:HF_HUB_CACHE=Join-Path $env:HF_HOME 'hub';$env:PYGAME_HIDE_SUPPORT_PROMPT='1';$env:SDL_AUDIODRIVER='dummy'
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
foreach($name in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'2','Process')}
$head=(git -C $repo rev-parse HEAD).Trim()
if($head -ne '67d2fd7eabbf79c2df8260b18c3334efcf2c058f' -or (git -C $repo status --porcelain)){throw 'Frozen exact clean source required'}
$configSha=(Get-FileHash -LiteralPath (Join-Path $repo 'configs/gpc_conditioning_contract02.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if($configSha -ne 'b2d2cd30ff1104cd84e9677a4c4e2a4c54b819384e34ba595be75f8286721547'){throw 'Frozen config changed'}
$sources=[ordered]@{}
foreach($relative in @('tdyssey/gpc_conditioning_contract.py','tdyssey/gpc_conditioning_data.py','tdyssey/gpc_conditioning_diagnostic.py','tdyssey/gpc_conditioning_entry.py','tdyssey/gpc_adaptation_guard.py','tdyssey/gpc_checkpoint_evaluation.py','scripts/gpc_conditioning_contract.py','configs/gpc_conditioning_contract02.json')){
    $sources[$relative]=(Get-FileHash -LiteralPath (Join-Path $repo $relative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
[ordered]@{commit=$head;config_sha256=$configSha;sources=$sources;upstream='a96146e568826ef46134f73c22fdf59a675e7264';plan_sha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'run-plan.json') -Algorithm SHA256).Hash.ToLowerInvariant();attempts=1}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'source-freeze.json') -Encoding utf8
$arguments=@('-B','scripts/gpc_conditioning_contract.py','--execute','--authorization-id','auth20261026-user-autonomous-gpc-conditioning-contract-diagnostic-02','--reviewed-commit',$head,'--authorized-config-sha256',$configSha,'--python',$python,'--gpc-root',(Join-Path (Split-Path -Parent $repo) 'Tdyssey/third_party/gpc_code'),'--checkpoint',(Join-Path $repo 'artifacts/frozen-wm-assets/denoiser.pth'))
$watch=[Diagnostics.Stopwatch]::StartNew();$reason=$null;$exitCode=$null;$process=$null
try{
    $process=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'real.stdout.txt') -RedirectStandardError (Join-Path $PSScriptRoot 'real.stderr.txt')
    $process.ProcessorAffinity=3
    while(-not $process.WaitForExit(200)){
        if($watch.Elapsed.TotalSeconds -ge 290){$reason='External290s cutoff leaves cleanup within300s';break}
    }
    if($reason){$process.Kill($true);if(-not $process.WaitForExit(5000)){throw 'Supervisor process cleanup uncertain'}}
    $process.WaitForExit();$exitCode=$process.ExitCode
    if(-not $reason -and $exitCode -ne 0){$reason='First guarded refusal/error; no retry'}
}finally{
    $watch.Stop()
    [ordered]@{source_commit=$head;config_sha256=$configSha;exit_code=$exitCode;reason=$reason;seconds=$watch.Elapsed.TotalSeconds;retry=$false;process_exited=if($process){$process.HasExited}else{$null};training_updates=0}|ConvertTo-Json|Set-Content -LiteralPath $receipt -Encoding utf8
}
Get-Content -LiteralPath $receipt
if($reason){exit 1}
