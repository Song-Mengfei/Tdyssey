$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$receipt=Join-Path $PSScriptRoot 'video-cpu.json'
if(Test-Path -LiteralPath $receipt){throw 'Video stage already used'}
$python=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.tools/python/cpython-3.12.3-windows-x86_64-none/python.exe'
$env:PYTHONPATH=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.venv/Lib/site-packages'
$env:PYTHONDONTWRITEBYTECODE='1';$env:CUDA_VISIBLE_DEVICES=''
foreach($name in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'2','Process')}
$watch=[Diagnostics.Stopwatch]::StartNew();$peak=0L;$reason=$null;$exitCode=$null
try{
    $process=Start-Process -FilePath $python -ArgumentList @('-B',(Join-Path $PSScriptRoot 'review-video.py')) -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'video.stdout.txt') -RedirectStandardError (Join-Path $PSScriptRoot 'video.stderr.txt')
    $process.ProcessorAffinity=3
    while(-not $process.WaitForExit(100)){
        $process.Refresh();$peak=[Math]::Max($peak,$process.PeakWorkingSet64)
        if($watch.Elapsed.TotalSeconds -ge 120){$reason='Video CPU 120s timeout';break}
        if($peak -gt 1GB){$reason='Video CPU working set 1GiB';break}
    }
    if($reason){$process.Kill($true);$process.WaitForExit(1000)|Out-Null}
    $process.WaitForExit();$exitCode=$process.ExitCode
    if(-not $reason -and $exitCode -ne 0){$reason='Video command failed'}
}finally{
    $watch.Stop()
    [ordered]@{stage='video';exit_code=$exitCode;reason=$reason;seconds=$watch.Elapsed.TotalSeconds;observed_peak_working_set=$peak;real_models=0;physics=0}|ConvertTo-Json|Set-Content -LiteralPath $receipt -Encoding utf8
}
Get-Content -LiteralPath $receipt
if($reason){exit 1}
