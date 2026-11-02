param([Parameter(Mandatory=$true)][ValidateSet('spy','integration','final')][string]$Stage)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$receipt=Join-Path $PSScriptRoot "cpu-$Stage.json"
if(Test-Path -LiteralPath $receipt){throw 'Synthetic stage already spent; preserve failure'}
$python=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.tools/python/cpython-3.12.3-windows-x86_64-none/python.exe'
$env:PYTHONPATH=Join-Path (Split-Path -Parent $repo) 'Tdyssey/.venv/Lib/site-packages'
$env:PYTHONDONTWRITEBYTECODE='1';$env:CUDA_VISIBLE_DEVICES=''
foreach($name in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'2','Process')}
$arguments=if($Stage -eq 'spy'){@('-B','-m','unittest','tests.test_gpc_conditioning_contract')}else{@('-B',(Join-Path $PSScriptRoot 'synthetic-check.py'),$Stage)}
$watch=[Diagnostics.Stopwatch]::StartNew();$peak=0L;$reason=$null;$exitCode=$null
try{
    $process=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot "cpu-$Stage.stdout.txt") -RedirectStandardError (Join-Path $PSScriptRoot "cpu-$Stage.stderr.txt")
    $process.ProcessorAffinity=3
    while(-not $process.WaitForExit(100)){
        $process.Refresh();$peak=[Math]::Max($peak,$process.PeakWorkingSet64)
        if($watch.Elapsed.TotalSeconds -ge 60){$reason='Synthetic60s ceiling';break}
        if($peak -gt 1GB){$reason='Synthetic1GiB WS ceiling';break}
    }
    if($reason){$process.Kill($true);$process.WaitForExit(1000)|Out-Null}
    $process.WaitForExit();$exitCode=$process.ExitCode
    if(-not $reason -and $exitCode -ne 0){$reason='Synthetic/source failure'}
}finally{
    $watch.Stop()
    [ordered]@{stage=$Stage;arguments=$arguments;exit_code=$exitCode;reason=$reason;seconds=$watch.Elapsed.TotalSeconds;observed_peak_working_set=$peak;real_torch=0;models=0;physics=0}|ConvertTo-Json|Set-Content -LiteralPath $receipt -Encoding utf8
}
Get-Content -LiteralPath $receipt
if($reason){exit 1}
