$ErrorActionPreference = 'Stop'
$taskRepository = '__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main'
$taskPython = '__LOCAL_WORKSPACE__\Tdyssey\.venv\Scripts\python.exe'
$taskLauncher = Join-Path $taskRepository 'artifacts\native-rgb-supervised-20261029\r3b\background\launch.py'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { throw 'Existing Python not found' }
if (-not (Test-Path -LiteralPath $taskLauncher -PathType Leaf)) { throw 'Frozen launcher not found' }
$taskBroker = New-Object -ComObject Shell.Application
$taskHwnd = 0
$taskDesktop = $taskBroker.Windows().FindWindowSW(0, $null, 8, [ref]$taskHwnd, 1)
if ($null -eq $taskDesktop) { throw 'Existing Explorer desktop broker unavailable' }
$taskArguments = '-X utf8 -B "' + $taskLauncher + '" --launch'
$taskDesktop.Document.Application.ShellExecute($taskPython, $taskArguments, $taskRepository, 'open', 0)
[pscustomobject]@{ submitted = $true; launcher = $taskLauncher; hidden = $true; broker = 'existing Explorer desktop Document.Application' } | ConvertTo-Json -Compress
