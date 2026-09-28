# Export SketchUp's own view of every scene (ground truth for render matching).
# Works on a copy of the model and force-quits SketchUp without saving.
param(
    [Parameter(Mandatory)] [string] $Skp,
    [Parameter(Mandatory)] [string] $OutDir,
    [string] $SketchUp = "C:\Program Files\SketchUp\SketchUp 2023\SketchUp.exe",
    [int] $TimeoutSec = 300
)
New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
Get-ChildItem $OutDir -Filter *.png | Remove-Item
$copy = Join-Path $OutDir "model_copy.skp"
Copy-Item $Skp $copy -Force
$env:SKP_REF_OUT = $OutDir
$rb = Join-Path $PSScriptRoot "export_reference.rb"
$p = Start-Process $SketchUp -ArgumentList "-RubyStartup `"$rb`" `"$copy`"" -PassThru
if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    Stop-Process -Id $p.Id -Force
    Write-Warning "SketchUp timed out; killed PID $($p.Id)"
}
Remove-Item $copy -ErrorAction SilentlyContinue
Get-ChildItem $OutDir -Filter *.png | Select-Object Name
