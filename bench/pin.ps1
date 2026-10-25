# Runs a program pinned to one logical CPU (a performance core by default)
# and prints its output: for tuning runs, which must not land on an E-core.
#
#   powershell -File bench\pin.ps1 [-Core 8] program.exe [args...]
param([int]$Core = 8, [Parameter(Mandatory = $true)][string]$Exe, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)

$log = [System.IO.Path]::GetTempFileName()
if ($Rest) { $p = Start-Process -FilePath $Exe -ArgumentList $Rest -NoNewWindow -PassThru -RedirectStandardOutput $log }
else { $p = Start-Process -FilePath $Exe -NoNewWindow -PassThru -RedirectStandardOutput $log }
$p.ProcessorAffinity = [IntPtr]([long](1 -shl $Core))
$p.WaitForExit()
Get-Content $log
Remove-Item $log
