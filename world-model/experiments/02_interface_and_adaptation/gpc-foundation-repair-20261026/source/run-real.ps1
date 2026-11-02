$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$receipt=Join-Path $PSScriptRoot 'real-external.json'
if((Test-Path -LiteralPath $receipt) -or (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'real-01'))){throw 'The sole real attempt is already spent'}
if(Get-Process | Where-Object { $_.ProcessName -like '*python*' -or $_.MainWindowTitle -like '*Tdyssey*' }){throw 'Another Python/Tdyssey process is active'}
$sourcePaths=@('tdyssey/manual_world_model.py','scripts/manual_world_model.py','tdyssey/gpc_adaptation.py','tdyssey/gpc_supervised_adaptation.py','tdyssey/gpc_evaluation_assets.py','tdyssey/envs/official.py','tdyssey/state.py','tdyssey/wm_compat_transform.py','tdyssey/wm_image_measurement_v2.py','artifacts/gpc-foundation-repair-20261026/real-plan.json','artifacts/gpc-foundation-repair-20261026/run-real.py','artifacts/gpc-foundation-repair-20261026/run-real.ps1','artifacts/manual-world-model/20261026-115956-905921/events.jsonl')
$sources=[ordered]@{}
foreach($relative in $sourcePaths){$sources[$relative]=(Get-FileHash -LiteralPath (Join-Path $repo $relative) -Algorithm SHA256).Hash.ToLowerInvariant()}
[ordered]@{sources=$sources;time_utc=[DateTime]::UtcNow.ToString('o');head=(git -C $repo rev-parse HEAD)}|ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'real-freeze.json') -Encoding utf8
$python=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.tools/python/cpython-3.12.3-windows-x86_64-none/python.exe'
$env:PYTHONPATH=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.venv/Lib/site-packages'
$env:PYTHONDONTWRITEBYTECODE='1';$env:SDL_VIDEODRIVER='dummy';$env:SDL_AUDIODRIVER='dummy';$env:PYGAME_HIDE_SUPPORT_PROMPT='1'
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$env:HF_HUB_OFFLINE='1';$env:TRANSFORMERS_OFFLINE='1';$env:HF_HOME=Join-Path $PSScriptRoot 'real-01/runtime-cache'
foreach($name in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'2','Process')}
$watch=[Diagnostics.Stopwatch]::StartNew();$peakWS=0L;$peakPrivate=0L;$reason=$null;$exitCode=$null;$process=$null
try{
    $process=Start-Process -FilePath $python -ArgumentList @('-B',(Join-Path $PSScriptRoot 'run-real.py')) -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'real.stdout.txt') -RedirectStandardError (Join-Path $PSScriptRoot 'real.stderr.txt')
    $process.ProcessorAffinity=3
    while(-not $process.WaitForExit(200)){
        $process.Refresh();$peakWS=[Math]::Max($peakWS,$process.PeakWorkingSet64);$peakPrivate=[Math]::Max($peakPrivate,$process.PrivateMemorySize64)
        if($watch.Elapsed.TotalSeconds -ge 300){$reason='Real 300s timeout including cleanup';break}
        if($peakWS -gt 2GB -or $peakPrivate -gt 3GB){$reason='Real process memory ceiling';break}
    }
    if($reason){$process.Kill($true);$process.WaitForExit(1000)|Out-Null}
    $process.WaitForExit();$exitCode=$process.ExitCode
    if(-not $reason -and $exitCode -ne 0){$reason='First technical error; sole attempt stopped'}
}finally{
    $watch.Stop()
    [ordered]@{exit_code=$exitCode;reason=$reason;seconds=$watch.Elapsed.TotalSeconds;observed_peak_working_set=$peakWS;observed_peak_private=$peakPrivate;training_updates=0;retry=$false;process_exited=if($process){$process.HasExited}else{$null}}|ConvertTo-Json|Set-Content -LiteralPath $receipt -Encoding utf8
}
Get-Content -LiteralPath $receipt
if($reason){exit 1}
