$ErrorActionPreference='Stop'
[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
$root=$PSScriptRoot
$config=Get-Content -LiteralPath (Join-Path $root 'continuation-config.json') -Raw -Encoding utf8|ConvertFrom-Json
$release=Get-Content -LiteralPath (Join-Path $root 'continuation-release.json') -Raw -Encoding utf8|ConvertFrom-Json
$digest=(Get-FileHash -LiteralPath (Join-Path $root 'continuation-config.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $release.resume_authorized -or $release.config_sha256 -ne $digest){throw 'Continuation release mismatch'}
if(Test-Path -LiteralPath (Join-Path $root 'continuation-execution-claim.json')){throw 'Continuation already claimed'}
$old=Get-Content -LiteralPath (Join-Path $root 'frozen-config.json') -Raw -Encoding utf8|ConvertFrom-Json
$scheduler=New-Object -ComObject 'Schedule.Service';$scheduler.Connect();$folder=$scheduler.GetFolder('\')
$previous=$folder.GetTask($old.windows_task_name)
if($previous.State -eq 4 -or $previous.State -eq 2 -or $previous.GetInstances(0).Count -ne 0){throw 'Original task remains active; no parallel launch'}
if(@($folder.GetTasks(1)|Where-Object{$_.Name -eq $config.windows_task_name}).Count -ne 0){throw 'Continuation task already exists'}
& $old.python_executable -X utf8 -B -c "import sys;sys.path.insert(0,r'$root');from continuation_io import install;install().guard(65536,refresh=True)"
if($LASTEXITCODE -ne 0){throw 'Metadata reservation failed'}
$definition=$scheduler.NewTask(0);$definition.Settings.Enabled=$true;$definition.Settings.Hidden=$true;$definition.Settings.Priority=7;$definition.Settings.ExecutionTimeLimit='PT0S'
$definition.Settings.DisallowStartIfOnBatteries=$false;$definition.Settings.StopIfGoingOnBatteries=$false
$definition.Principal.UserId=[Security.Principal.WindowsIdentity]::GetCurrent().Name;$definition.Principal.LogonType=3
$action=$definition.Actions.Create(0);$action.Path=[IO.Path]::Combine([IO.Path]::GetDirectoryName($old.python_executable),'pythonw.exe')
$action.Arguments='-X utf8 -B "'+(Join-Path $root 'continuation_worker.py')+'" --config-sha '+$digest;$action.WorkingDirectory=$old.repo
$task=$folder.RegisterTaskDefinition($config.windows_task_name,$definition,2,$null,$null,3,$null);$instance=$task.Run($null)
$receipt=[ordered]@{run_id=$config.run_id;config_sha256=$digest;task_name=$config.windows_task_name;action_path=$action.Path;action_arguments=$action.Arguments;working_directory=$action.WorkingDirectory;priority=7;hidden=$true;trigger_count=0;instance_guid=$instance.InstanceGuid;task_state=[int]$task.State;original_task_ended=$true;at_utc=[DateTime]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $root 'continuation-task-registration.json'),($receipt|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
$receipt|ConvertTo-Json -Depth 8

