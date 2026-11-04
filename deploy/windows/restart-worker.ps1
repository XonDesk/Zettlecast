# Restart the Zettlecast worker task, stopping any worker processes it leaves behind, and make sure
# it is running again (a start right after a stop can be ignored while the task is still stopping).
Stop-ScheduledTask "Zettlecast worker"
for ($i = 0; $i -lt 20 -and (Get-ScheduledTask "Zettlecast worker").State -eq "Running"; $i++) { Start-Sleep 1 }
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -in @("python.exe", "zettlecast.exe") -and $_.CommandLine -like "*zettlecast*worker*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "stopped $($_.ProcessId) $($_.Name)" }
Start-Sleep 2
for ($try = 1; $try -le 3; $try++) {
    Start-ScheduledTask "Zettlecast worker"
    Start-Sleep 5
    if ((Get-ScheduledTask "Zettlecast worker").State -eq "Running") { break }
    "start attempt $try didn't take, retrying"
}
Start-Sleep 10
"task: " + (Get-ScheduledTask "Zettlecast worker").State
Get-Content D:\code\Zettlecast\worker.log -Tail 2
