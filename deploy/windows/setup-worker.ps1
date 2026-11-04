# Set this PC up as the Zettlecast worker (ROLE=worker). Run once, in an elevated PowerShell:
#
#   .\deploy\windows\setup-worker.ps1 -PrimaryIp 192.168.1.20 [-EnableRdp]
#
# What it does (each step prints what it changed):
#   1. Fast Startup off: a "shut down" PC is then really off (S5), the state Wake-on-LAN wakes from
#   2. Wake on Magic Packet on for the wired network card(s); prints the MAC address for the primary
#   3. A scheduled task that starts `zettlecast worker` at boot, signed in or not (runs as you, so it
#      finds your models and storage folder; S4U logon, so no password is asked for or stored)
#   4. Firewall: the worker's file server (WORKER_PORT, 8010) open to the primary only
#   5. -EnableRdp: Remote Desktop on, for the private network (VPN included)
#
# Still by hand: Wake-on-LAN in the BIOS/UEFI (see README.md), and ROLE=worker, PRIMARY_URL,
# WORKER_TOKEN in .env.

param(
    [Parameter(Mandatory = $true)][string]$PrimaryIp,
    [int]$Port = 8010,
    [switch]$EnableRdp
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$Exe = Join-Path $Repo ".venv\Scripts\zettlecast.exe"

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Run this in an elevated PowerShell (Run as administrator)." }
if (-not (Test-Path $Exe)) { throw "No ${Exe}: install Zettlecast in .venv first." }

Write-Host "`n1. Fast Startup" -ForegroundColor Cyan
Set-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power" -Name HiberbootEnabled -Value 0
Write-Host "   off (shut down = fully off, wakeable from the network)"

Write-Host "`n2. Network card" -ForegroundColor Cyan
$nics = Get-NetAdapter -Physical | Where-Object { $_.Status -eq "Up" -and $_.MediaType -eq "802.3" }
if (-not $nics) { Write-Warning "   No wired network card is connected: Wake-on-LAN needs Ethernet." }
foreach ($nic in $nics) {
    try {
        Set-NetAdapterPowerManagement -Name $nic.Name -WakeOnMagicPacket Enabled -ErrorAction Stop
        Write-Host "   $($nic.Name): wake on magic packet enabled"
    } catch {
        Write-Warning "   $($nic.Name): couldn't enable wake on magic packet ($_)"
    }
    # driver settings differ by vendor: "Wake on Magic Packet", "Shutdown Wake-On-Lan", "Wake from S5"...
    Get-NetAdapterAdvancedProperty -Name $nic.Name |
        Where-Object { $_.DisplayName -match "Magic Packet|Shutdown Wake|Wake from S5|WOL" -and $_.ValidDisplayValues -contains "Enabled" } |
        ForEach-Object {
            Set-NetAdapterAdvancedProperty -Name $nic.Name -DisplayName $_.DisplayName -DisplayValue "Enabled"
            Write-Host "   $($nic.Name): $($_.DisplayName) = Enabled"
        }
    $mac = $nic.MacAddress -replace "-", ":"
    Write-Host "   MAC address for the primary's .env:  DESKTOP_MAC_ADDRESS=$mac" -ForegroundColor Green
}

Write-Host "`n3. Start the worker at boot" -ForegroundColor Cyan
# not "$env:USERDOMAIN\$env:USERNAME": over SSH USERDOMAIN is WORKGROUP
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
# S4U: runs as you, signed in or not, with no stored password (works over SSH too). It gets no network
# credentials, which the worker doesn't need: local files and plain HTTP to the primary.
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
$log = Join-Path $Repo "worker.log"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -WorkingDirectory $Repo `
    -Argument "/c `"`"$Exe`" worker >> `"$log`" 2>&1`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable
Register-ScheduledTask -TaskName "Zettlecast worker" -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Force | Out-Null
Write-Host "   task 'Zettlecast worker' registered (log: $log)"

Write-Host "`n4. Firewall" -ForegroundColor Cyan
Get-NetFirewallRule -DisplayName "Zettlecast worker files" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName "Zettlecast worker files" -Direction Inbound -Protocol TCP -LocalPort $Port `
    -RemoteAddress $PrimaryIp -Action Allow -Profile Private | Out-Null
Write-Host "   TCP $Port open to $PrimaryIp only (private networks)"

if ($EnableRdp) {
    Write-Host "`n5. Remote Desktop" -ForegroundColor Cyan
    Set-ItemProperty "HKLM:\System\CurrentControlSet\Control\Terminal Server" -Name fDenyTSConnections -Value 0
    Enable-NetFirewallRule -DisplayGroup "Remote Desktop"
    Write-Host "   on (Network Level Authentication stays on); reach it over the VPN at this PC's LAN address"
}

Write-Host "`nDone. Next: BIOS Wake-on-LAN (README.md), .env (ROLE=worker, PRIMARY_URL, WORKER_TOKEN)," `
    "then: Start-ScheduledTask 'Zettlecast worker'" -ForegroundColor Green
