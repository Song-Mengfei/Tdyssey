$ErrorActionPreference = 'Stop'
$taskBroker = New-Object -ComObject Shell.Application
$taskHwnd = 0
$taskDesktop = $taskBroker.Windows().FindWindowSW(0, $null, 8, [ref]$taskHwnd, 1)
if ($null -eq $taskDesktop) { throw 'Explorer desktop unavailable' }
$taskDesktop.Document.Application.ShellExecute('__LOCAL_WORKSPACE__\Tdyssey\.venv\Scripts\python.exe', '-X utf8 -B "__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\apps\r3b-playground\start.py"', '__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main', 'open', 0)
'Updated own playground service submitted'
