$ErrorActionPreference = "Continue"
$device = "USB\VID_10C4&PID_EA60\C049D5011E9CEF11AD47B79061CE3355"
$result = Join-Path $env:TEMP "codex-reset-cp210x-result.json"
$restartOutput = (& "$env:WINDIR\System32\pnputil.exe" /restart-device $device 2>&1 | Out-String)
Start-Sleep -Seconds 3
$pnp = Get-PnpDevice -InstanceId $device -ErrorAction SilentlyContinue
$open = "not_attempted"
$errorText = $null
try {
  $port = [System.IO.Ports.SerialPort]::new("COM3", 115200, "None", 8, "One")
  $port.ReadTimeout = 500
  $port.Open()
  $open = "opened"
  $port.Close()
} catch {
  $open = "failed"
  $errorText = $_.Exception.Message
}
[pscustomobject]@{
  utc = [DateTime]::UtcNow.ToString("o")
  elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  device_instance = $device
  restart_output = $restartOutput
  pnp_status = $pnp.Status
  pnp_problem = $pnp.Problem
  com3_open = $open
  com3_error = $errorText
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $result
