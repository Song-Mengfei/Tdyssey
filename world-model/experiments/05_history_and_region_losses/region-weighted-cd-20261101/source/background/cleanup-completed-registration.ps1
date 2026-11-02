$ErrorActionPreference = 'Stop'
$rootPath = $PSScriptRoot | Split-Path -Parent
$runId = 'RWCD-20261101-81bdec2c68934570a46dac1db04b3a72'
$expectedHash = '6ae814d58c7af48b6c2cc6e63181a6f254281453b2f1d7a1c63f387725142c46'
$taskName = 'Tdyssey-RWCD-81bdec2c'
$cfg = Get-Content -LiteralPath (Join-Path $rootPath 'runtime-config.json') -Raw | ConvertFrom-Json
$freeze = Get-Content -LiteralPath (Join-Path $rootPath 'freeze-receipt.json') -Raw | ConvertFrom-Json
$digest = (Get-FileHash -LiteralPath (Join-Path $rootPath 'runtime-config.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($digest -ne $expectedHash -or $cfg.run_id -ne $runId -or $freeze.run_id -ne $runId -or $freeze.config_sha256 -ne $expectedHash -or $cfg.windows_task_name -ne $taskName) { throw 'Identity mismatch' }
$callback = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'final-callback-result.json') -Raw | ConvertFrom-Json
$completion = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'completion.json') -Raw | ConvertFrom-Json
$serviceExit = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'service-exit.json') -Raw | ConvertFrom-Json
$registration = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'task-registration.json') -Raw | ConvertFrom-Json
if ($callback.sent -ne $true -or $callback.run_id -ne $runId -or $callback.actual_tool_result.isError -ne $false) { throw 'Actual successful callback required' }
if ($completion.run_id -ne $runId -or $completion.config_sha256 -ne $expectedHash -or $completion.completed -ne $true -or $completion.failed -ne $false -or $serviceExit.exit_code -ne 0) { throw 'Successful terminated completion required' }
if ($registration.run_id -ne $runId -or $registration.config_sha256 -ne $expectedHash -or $registration.task_name -ne $taskName) { throw 'Registration receipt differs' }
$scheduler = New-Object -ComObject 'Schedule.Service'
$scheduler.Connect()
$folder = $scheduler.GetFolder('\')
$task = $folder.GetTask($taskName)
$definition = $task.Definition
if ($definition.Actions.Count -ne 1 -or $definition.Triggers.Count -ne 0) { throw 'Unexpected task actions or triggers' }
$action = $definition.Actions.Item(1)
$expectedPython = Join-Path (Split-Path $cfg.python_executable -Parent) 'pythonw.exe'
$expectedArguments = '-X utf8 -B "' + (Join-Path $rootPath 'scheduled_entry.py') + '" --launcher-pid 32860'
if ($action.Path -ne $expectedPython -or $action.Path -ne $registration.action_path -or $action.Arguments -ne $expectedArguments -or $action.Arguments -ne $registration.action_arguments -or $action.WorkingDirectory -ne $cfg.repo) { throw 'Action identity differs' }
$instanceCount = $task.GetInstances(0).Count
$state = [int]$task.State
$lastResult = [long]$task.LastTaskResult
if ($state -eq 4 -or $state -eq 2 -or $instanceCount -ne 0 -or $lastResult -ne 0) { throw 'Task has not verifiably ended successfully with zero instances' }
$receipt = [ordered]@{run_id=$runId;config_sha256=$expectedHash;task_name=$taskName;final_callback_sent=$true;service_exit_code=$serviceExit.exit_code;state_before=$state;instances_before=$instanceCount;last_task_result=$lastResult;last_run_time=$task.LastRunTime.ToString('o');action_path=$action.Path;action_arguments=$action.Arguments;action_working_directory=$action.WorkingDirectory;trigger_count=$definition.Triggers.Count;removed_registration_only=$true;process_control_calls=0;other_task_mutations=0;deleted=$false;at_utc=[DateTime]::UtcNow.ToString('o')}
$folder.DeleteTask($taskName,0)
$remaining = @($folder.GetTasks(1) | Where-Object { $_.Name -eq $taskName }).Count
$receipt.deleted = $true
$receipt.registration_matches_remaining = $remaining
$receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'windows-task-delete-result.json') -Encoding utf8
if ($remaining -ne 0) { throw 'Task registration still present after delete' }
$receipt | ConvertTo-Json -Depth 10

