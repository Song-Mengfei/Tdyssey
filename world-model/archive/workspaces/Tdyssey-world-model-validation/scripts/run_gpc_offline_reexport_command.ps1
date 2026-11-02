param(
    [Parameter(Mandatory=$true)][ValidateSet(1,2)][int]$Command
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$out = Join-Path $repo 'artifacts/gpc-checkpoint500-offline-reexport-20261026'
$old = Join-Path $repo 'artifacts/gpc-checkpoint500-offline-diagnosis-20261026'
$manifest = Join-Path $old 'fixed-samples.json'
if ((Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne '193096b5f23df1bec4e8f2ed73df5d15fbc4f2ef9c3edc571ad9701eb61a7ed7') { throw 'Original sample identity changed' }
$env:TDYSSEY_OFFLINE_PIXEL_BASELINE = Join-Path $out 'baseline-helper.py'
if ((Get-FileHash -LiteralPath $env:TDYSSEY_OFFLINE_PIXEL_BASELINE -Algorithm SHA256).Hash.ToLowerInvariant() -ne '4ec121d2b1b61601a5883be39a50695214b1a32aed6fd97699b7b3e07e57e8b8') { throw 'Original helper identity changed' }
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
    1 { @('-B','-m','unittest','tests.test_gpc_checkpoint500_offline_pixels','tests.test_gpc_checkpoint500_offline_contract') }
    2 {
        $first = Get-Content -LiteralPath (Join-Path $out 'command-1-receipt.json') -Raw | ConvertFrom-Json
        if ($first.status -ne 'complete') { throw 'The fixed regression command must pass first' }
        @('-B','scripts/gpc_checkpoint500_offline_pixels.py','--root',$repo,'--output',$out,'--manifest',$manifest)
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
    if (-not $reason -and $Command -eq 2) {
        $report = Get-Content -LiteralPath (Join-Path $out 'pixels/pixels.json') -Raw | ConvertFrom-Json
        $wanted = @{prediction_frames=48; physical_frames=64; training_frames=40; training_small_state_rows=80; verified_input_hashes=32; unknown_AR4_input_hashes=16; verified_action_hashes=48}
        foreach ($key in $wanted.Keys) { if ($report.decoded_counts.$key -ne $wanted[$key]) { throw "Durable count mismatch: $key" } }
        $predictions = @($report.branches | ForEach-Object { $_.predictions })
        if ($report.branches.Count -ne 8 -or $report.train_windows.Count -ne 8 -or $predictions.Count -ne 48) { throw 'Full JSON grid mismatch' }
        if (@($predictions | Where-Object normalized_action_hash_check -ne 'matched').Count -ne 0 -or @($predictions | Where-Object input_hash_check -eq 'matched').Count -ne 32 -or @($predictions | Where-Object input_hash_check -eq 'unknown_AR4_intermediate_raw_frames_not_opened').Count -ne 16) { throw 'Input or action check mismatch' }
        if ($report.fixed_manifest_sha256 -ne '193096b5f23df1bec4e8f2ed73df5d15fbc4f2ef9c3edc571ad9701eb61a7ed7') { throw 'Exported manifest identity mismatch' }
        $oldPngs = @(Get-ChildItem -LiteralPath (Join-Path $old 'pixels') -Filter '*.png' -File)
        $newPngs = @(Get-ChildItem -LiteralPath (Join-Path $out 'pixels') -Filter '*.png' -File)
        if ($oldPngs.Count -ne 32 -or $newPngs.Count -ne 32) { throw 'PNG grid mismatch' }
        $matches = foreach ($file in $oldPngs) {
            $peer = Join-Path (Join-Path $out 'pixels') $file.Name
            $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            if ((Get-FileHash -LiteralPath $peer -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hash) { throw "PNG mismatch: $($file.Name)" }
            [pscustomobject]@{name=$file.Name; bytes=$file.Length; sha256=$hash; identical=$true}
        }
        [ordered]@{full_json_parsed=$true; finite_serialization='producer allow_nan=False; regression retains NaN rejection'; decoded_counts=$report.decoded_counts; pngs=$matches; scientific_scores_changed=$false} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $out 'export-verification.json') -Encoding utf8
    }
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
