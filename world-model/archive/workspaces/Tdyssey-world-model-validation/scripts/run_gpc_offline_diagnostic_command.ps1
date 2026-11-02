param(
    [Parameter(Mandatory=$true)][ValidateSet(1,2,3)][int]$Command,
    [string]$ManifestSha256 = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$out = Join-Path $repo 'artifacts/gpc-checkpoint500-offline-diagnosis-20261026'
$python = Join-Path (Split-Path -Parent $repo) 'Tdyssey/.tools/python/cpython-3.12.3-windows-x86_64-none/python.exe'
$env:PYTHONPATH = Join-Path (Split-Path -Parent $repo) 'Tdyssey/.venv/Lib/site-packages'
$receipt = Join-Path $out "command-$Command-receipt.json"
if (Test-Path -LiteralPath $receipt) { throw 'Diagnostic command already spent' }
if (Get-ChildItem -LiteralPath $out -Filter 'command-*-receipt.json' | Where-Object {
    (Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json).status -ne 'complete'
}) { throw 'An earlier command failed; remaining numerical work is forbidden' }
foreach ($name in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')) {
    [Environment]::SetEnvironmentVariable($name, '2', 'Process')
}
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$arguments = switch ($Command) {
    1 { @('-B','scripts/gpc_checkpoint500_offline_diagnosis.py','freeze') }
    2 { @('-B','-m','unittest','tests.test_gpc_checkpoint500_offline_pixels','tests.test_gpc_checkpoint500_offline_contract') }
    3 {
        if ($ManifestSha256 -notmatch '^[a-f0-9]{64}$') { throw 'Frozen sample SHA256 required' }
        @('-B','scripts/gpc_checkpoint500_offline_diagnosis.py','pixels','--manifest-sha256',$ManifestSha256)
    }
}
$claim = [IO.File]::Open($receipt, 'CreateNew', 'Write', 'Read')
$claim.Close()
$watch = [Diagnostics.Stopwatch]::StartNew()
$process = $null
$peak = 0L
$reason = $null
$code = $null
try {
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $out "command-$Command.stdout.txt") -RedirectStandardError (Join-Path $out "command-$Command.stderr.txt")
    $process.ProcessorAffinity = 3
    while (-not $process.WaitForExit(100)) {
        $process.Refresh()
        $peak = [Math]::Max($peak, $process.PeakWorkingSet64)
        if ($watch.Elapsed.TotalSeconds -ge 60) { $reason = '60-second external timeout'; break }
        if ($peak -gt 1GB) { $reason = '1 GiB working-set limit'; break }
        $bytes = (Get-ChildItem -LiteralPath $out -File -Recurse | Measure-Object -Property Length -Sum).Sum
        if ($bytes -gt 32MB) { $reason = '32 MiB diagnostic evidence limit'; break }
    }
    if ($reason) { $process.Kill($true); $process.WaitForExit(1000) | Out-Null }
    $process.Refresh()
    $peak = [Math]::Max($peak, $process.PeakWorkingSet64)
    $process.WaitForExit()
    $code = $process.ExitCode
    if (-not $reason -and $code -ne 0) { $reason = 'Nonzero exit code' }
    $bytes = (Get-ChildItem -LiteralPath $out -File -Recurse | Measure-Object -Property Length -Sum).Sum
    if (-not $reason -and $watch.Elapsed.TotalSeconds -gt 60) { $reason = '60-second external timeout' }
    if (-not $reason -and $peak -gt 1GB) { $reason = '1 GiB working-set limit' }
    if (-not $reason -and ($bytes + 4096) -gt 32MB) { $reason = '32 MiB diagnostic evidence limit including receipt' }
} catch {
    $reason = $_.Exception.ToString()
    if ($process -and -not $process.HasExited) { $process.Kill($true); $process.WaitForExit(1000) | Out-Null }
} finally {
    $watch.Stop()
    [ordered]@{command=$Command; python=$python; arguments=$arguments; status=$(if ($reason) {'failed'} else {'complete'}); reason=$reason; exit_code=$code; elapsed_seconds=$watch.Elapsed.TotalSeconds; peak_working_set_bytes=$peak; cpu_affinity=3; library_thread_limit=2; external_timeout_seconds=60} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receipt -Encoding utf8
}
Get-Content -LiteralPath $receipt
if ($reason) { exit 1 }
