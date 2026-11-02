param([Parameter(Mandatory=$true)][ValidateSet('first','repair','final')][string]$Stage)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$receipt=Join-Path $PSScriptRoot "cpu-$Stage.json"
if(Test-Path -LiteralPath $receipt){throw 'This engineering command is spent'}
$python=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.tools/python/cpython-3.12.3-windows-x86_64-none/python.exe'
$env:PYTHONPATH=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.venv/Lib/site-packages'
$env:PYTHONDONTWRITEBYTECODE='1';$env:CUDA_VISIBLE_DEVICES='';$env:SDL_VIDEODRIVER='dummy';$env:SDL_AUDIODRIVER='dummy';$env:PYGAME_HIDE_SUPPORT_PROMPT='1'
foreach($name in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'2','Process')}
$watch=[Diagnostics.Stopwatch]::StartNew();$peak=0L;$reason=$null;$exitCode=$null;$process=$null
try{
    $process=Start-Process -FilePath $python -ArgumentList @('-B',(Join-Path $PSScriptRoot 'check.py'),$Stage) -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot "cpu-$Stage.stdout.txt") -RedirectStandardError (Join-Path $PSScriptRoot "cpu-$Stage.stderr.txt")
    $process.ProcessorAffinity=3
    while(-not $process.WaitForExit(100)){
        $process.Refresh();$peak=[Math]::Max($peak,$process.PeakWorkingSet64)
        if($watch.Elapsed.TotalSeconds -ge 59){$reason='Engineering60s cap with cleanup reserve';break}
        if($peak -gt 1GB){$reason='Engineering1GiB WS cap';break}
    }
    if($reason){$process.Kill($true);$process.WaitForExit(1000)|Out-Null}
    $process.WaitForExit();$exitCode=$process.ExitCode
    if(-not $reason -and $exitCode -ne 0){$reason='Focused check failed'}
}finally{
    $watch.Stop()
    [ordered]@{stage=$Stage;exit_code=$exitCode;reason=$reason;seconds=$watch.Elapsed.TotalSeconds;observed_peak_working_set=$peak;real_torch=0;source_numeric_frames=0;model_loads=0;physics=0;training=0}|ConvertTo-Json|Set-Content -LiteralPath $receipt -Encoding utf8
}
Get-Content -LiteralPath $receipt
if($reason){exit 1}
