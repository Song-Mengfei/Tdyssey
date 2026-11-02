$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$receipt=Join-Path $PSScriptRoot 'source-cpu.json'
if(Test-Path -LiteralPath $receipt){throw 'Source check evidence already exists'}
$python=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.tools/python/cpython-3.12.3-windows-x86_64-none/python.exe'
$watch=[Diagnostics.Stopwatch]::StartNew();$peak=0L;$reason=$null;$exitCode=$null
try{
    $process=Start-Process -FilePath $python -ArgumentList @('-S','-B',(Join-Path $PSScriptRoot 'check-source.py')) -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'source.stdout.txt') -RedirectStandardError (Join-Path $PSScriptRoot 'source.stderr.txt')
    $process.ProcessorAffinity=3
    while(-not $process.WaitForExit(100)){
        $process.Refresh();$peak=[Math]::Max($peak,$process.PeakWorkingSet64)
        if($watch.Elapsed.TotalSeconds -ge 120){$reason='Source CPU 120s timeout';break}
    }
    if($reason){$process.Kill($true);$process.WaitForExit(1000)|Out-Null}
    $process.WaitForExit();$exitCode=$process.ExitCode
    if(-not $reason -and $exitCode -ne 0){$reason='Source/message check failed'}
}finally{
    $watch.Stop()
    [ordered]@{exit_code=$exitCode;reason=$reason;seconds=$watch.Elapsed.TotalSeconds;observed_peak_working_set=$peak;models=0;physics=0}|ConvertTo-Json|Set-Content -LiteralPath $receipt -Encoding utf8
}
Get-Content -LiteralPath $receipt
if($reason){exit 1}
