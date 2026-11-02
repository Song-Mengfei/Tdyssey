param([Parameter(Mandatory=$true)][ValidateSet('before','after','after-fix')][string]$Stage)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$out=$PSScriptRoot
$receipt=Join-Path $out "cpu-$Stage.json"
if(Test-Path -LiteralPath $receipt){throw 'CPU evidence stage already exists; preserve it'}
$python=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.tools/python/cpython-3.12.3-windows-x86_64-none/python.exe'
$env:PYTHONPATH=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.venv/Lib/site-packages'
$env:PYTHONDONTWRITEBYTECODE='1';$env:CUDA_VISIBLE_DEVICES='';$env:SDL_VIDEODRIVER='dummy';$env:SDL_AUDIODRIVER='dummy'
foreach($name in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'2','Process')}
if($Stage -eq 'before'){
    $env:TDYSSEY_FOUNDATION_MANUAL_SOURCE=Join-Path $out 'baseline/tdyssey/manual_world_model.py'
    $arguments=@('-B','-m','unittest','tests.test_manual_prediction_usage')
}else{
    Remove-Item Env:TDYSSEY_FOUNDATION_MANUAL_SOURCE -ErrorAction SilentlyContinue
    $arguments=@('-B','-m','unittest','tests.test_manual_prediction_usage','tests.test_gpc_foundation_contract','tests.test_manual_world_model','tests.test_manual_world_model_ui')
}
$watch=[Diagnostics.Stopwatch]::StartNew();$peak=0L;$reason=$null;$process=$null;$exitCode=$null
try{
    $process=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $out "cpu-$Stage.stdout.txt") -RedirectStandardError (Join-Path $out "cpu-$Stage.stderr.txt")
    $process.ProcessorAffinity=3
    while(-not $process.WaitForExit(100)){
        $process.Refresh();$peak=[Math]::Max($peak,$process.PeakWorkingSet64)
        if($watch.Elapsed.TotalSeconds -ge 120){$reason='CPU 120s timeout';break}
        if($peak -gt 1GB){$reason='CPU working set 1GiB';break}
    }
    if($reason){$process.Kill($true);$process.WaitForExit(1000)|Out-Null}
    $process.WaitForExit();$exitCode=$process.ExitCode
    if(-not $reason -and (($Stage -eq 'before' -and $exitCode -ne 1) -or ($Stage -ne 'before' -and $exitCode -ne 0))){$reason='Unexpected test exit code'}
    if($watch.Elapsed.TotalSeconds -gt 120){$reason='CPU 120s timeout'}
    if($peak -gt 1GB){$reason='CPU working set 1GiB'}
}finally{
    $watch.Stop()
    [ordered]@{stage=$Stage;arguments=$arguments;exit_code=$exitCode;reason=$reason;expected_before_failure=($Stage -eq 'before');seconds=$watch.Elapsed.TotalSeconds;observed_peak_working_set=$peak;real_models=0;physics=0;note='Synthetic/ordinary-object checks only; no user GUI or real model touched.'}|ConvertTo-Json -Depth 4|Set-Content -LiteralPath $receipt -Encoding utf8
}
Get-Content -LiteralPath $receipt
if($reason){exit 1}
