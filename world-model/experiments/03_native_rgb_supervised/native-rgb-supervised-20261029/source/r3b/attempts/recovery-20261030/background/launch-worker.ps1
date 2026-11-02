$ErrorActionPreference = 'Stop'
$taskBroker = New-Object -ComObject Shell.Application
$taskHwnd = 0
$taskDesktop = $taskBroker.Windows().FindWindowSW(0, $null, 8, [ref]$taskHwnd, 1)
if ($null -eq $taskDesktop) { throw 'Existing Explorer desktop unavailable' }
$taskDesktop.Document.Application.ShellExecute('__LOCAL_WORKSPACE__\Tdyssey\.venv\Scripts\python.exe', '-X utf8 -B "__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\native-rgb-supervised-20261029\r3b\attempts\recovery-20261030\background\launcher.py" --spec "__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\native-rgb-supervised-20261029\r3b\attempts\recovery-20261030\background\launch-spec.json"', '__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main', 'open', 0)
'Verified recovery firstepoch pipeline submitted'
