param([ValidateSet('journal')][string]$Mode)
$ErrorActionPreference = 'Stop'
$outerClock = [System.Diagnostics.Stopwatch]::StartNew()
$reviewDirectory = $PSScriptRoot
$repositoryDirectory = Split-Path (Split-Path $reviewDirectory -Parent) -Parent
$pythonExecutable = Join-Path (Split-Path $repositoryDirectory -Parent) 'Tdyssey/.tools/python/cpython-3.12.3-windows-x86_64-none/python.exe'
$fixtureScript = Join-Path $reviewDirectory 'native-journal.py'
$fixturePlan = Join-Path $reviewDirectory 'native-plan.json'
$stdoutFile = Join-Path $reviewDirectory ('native-'+$Mode+'-outer.stdout')
$stderrFile = Join-Path $reviewDirectory ('native-'+$Mode+'-outer.stderr')
$reportFile = Join-Path $reviewDirectory ('native-'+$Mode+'-outer.json')
foreach ($existingPath in @($stdoutFile,$stderrFile,$reportFile,(Join-Path $reviewDirectory ('native-'+$Mode)))) {
    if (Test-Path -LiteralPath $existingPath) {throw ('Existing run refuses execution: '+$existingPath)}
}
$fixtureProcess = $null
$outerReport = @{mode=$Mode; parent_pid=$PID; max_task_processes_including_powershell=3; external_timeout_ms=58800; cleanup_wait_ms=800; forced_kill=$false}
try {
    $fixtureProcess = Start-Process -FilePath $pythonExecutable -ArgumentList @('-S','-B',$fixtureScript,'--mode',$Mode,'--plan',$fixturePlan) -WorkingDirectory $repositoryDirectory -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
    $outerReport.python_parent_pid = $fixtureProcess.Id
    $remainingMilliseconds = [Math]::Max(0, 58800 - [int]$outerClock.ElapsedMilliseconds)
    if (-not $fixtureProcess.WaitForExit($remainingMilliseconds)) {
        $outerReport.forced_kill = $true
        $fixtureProcess.Kill($true)
        $outerReport.external_tree_exit_observed = $fixtureProcess.WaitForExit(800)
        throw 'External deadline stopped this run and its process tree; no retry'
    }
    $fixtureProcess.Refresh()
    $outerReport.exit_code = $fixtureProcess.ExitCode
} catch {
    $outerReport.error = $_.Exception.ToString()
    if (-not $outerReport.forced_kill -and $null -ne $fixtureProcess -and -not $fixtureProcess.HasExited) {
        $fixtureProcess.Kill($true)
        $outerReport.external_tree_exit_observed = $fixtureProcess.WaitForExit(800)
    }
} finally {
    $outerReport.wall_seconds = $outerClock.Elapsed.TotalSeconds
    $reportBytes = [System.Text.Encoding]::UTF8.GetBytes(($outerReport | ConvertTo-Json -Depth 4))
    $reportStream = [System.IO.File]::Open($reportFile,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write)
    try {$reportStream.Write($reportBytes,0,$reportBytes.Length)} finally {$reportStream.Dispose()}
}
$outerReport | ConvertTo-Json -Depth 4
if ($outerReport.ContainsKey('error')) {exit 124}
exit $outerReport.exit_code
