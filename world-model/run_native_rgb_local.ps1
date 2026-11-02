param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$PhysicsRoot,
    [Parameter(Mandatory=$true)][string]$GpcRoot,
    [Parameter(Mandatory=$true)][string]$InitCheckpoint,
    [string]$Output = "$PSScriptRoot/artifacts/native-rgb-20261028",
    [string]$Config = "$PSScriptRoot/configs/native_rgb_local.json",
    [ValidateSet('preflight','pilot','evaluate','resume')][string]$Mode = 'pilot',
    [string]$ResumeCheckpoint = ''
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply an existing Python; this launcher installs nothing.' }
$Output = [IO.Path]::GetFullPath($Output)
$env:HF_HOME = "$Output/runtime-cache/huggingface"
$env:HF_HUB_OFFLINE = '1'
$activeConfig = $Config
function Invoke-NativeStage([string]$Stage, [string[]]$Extra = @()) {
    $arguments = @('-X','utf8','-B',"$PSScriptRoot/scripts/native_rgb_world_model.py",$Stage,
        '--config',$activeConfig,'--output',$Output,'--physics-root',$PhysicsRoot,
        '--gpc-root',$GpcRoot,'--init-checkpoint',$InitCheckpoint) + $Extra
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Native RGB $Stage failed; inspect its preserved error report." }
}
Invoke-NativeStage 'preflight' @('--gpu')
if ($Mode -eq 'preflight') { exit 0 }
if (Test-Path -LiteralPath "$Output/pilot-effective.json") { $activeConfig = "$Output/pilot-effective.json" }
if ($Mode -in @('pilot','resume')) {
    if (-not (Test-Path -LiteralPath "$Output/data/manifest.json")) {
        if (-not (Test-Path -LiteralPath "$Output/physics-probe/report.json")) { Invoke-NativeStage 'probe' }
        Invoke-NativeStage 'collect'
    }
    Invoke-NativeStage 'audit'
    if (-not (Test-Path -LiteralPath "$Output/engineering/report.json")) {
        Invoke-NativeStage 'engineering'
        $activeConfig = "$Output/pilot-effective.json"
    }
    if (-not (Test-Path -LiteralPath "$Output/microfit/result.json")) { Invoke-NativeStage 'microfit' }
    if ($Mode -eq 'resume') {
        if (-not $ResumeCheckpoint) { throw 'Resume requires an explicit complete checkpoint.' }
        Invoke-NativeStage 'train' @('--resume',$ResumeCheckpoint)
    } elseif (-not (Test-Path -LiteralPath "$Output/training/result.json")) {
        Invoke-NativeStage 'train'
    }
}
$needsEvaluation = -not (Test-Path -LiteralPath "$Output/evaluation/seen_val/report.json")
if (-not $needsEvaluation -and (Test-Path -LiteralPath "$Output/training/best.pt")) {
    $validationPath = "$Output/evaluation/seen_val/report.json"
    if (Test-Path -LiteralPath "$Output/evaluation/current-seen-val.json") {
        $pointer = Get-Content -LiteralPath "$Output/evaluation/current-seen-val.json" -Raw | ConvertFrom-Json
        $validationPath = Join-Path $Output "$($pointer.directory)/report.json"
    }
    $validation = Get-Content -LiteralPath $validationPath -Raw | ConvertFrom-Json
    $needsEvaluation = $validation.checkpoint_sha256 -ne (Get-FileHash -LiteralPath "$Output/training/best.pt" -Algorithm SHA256).Hash.ToLowerInvariant()
}
if ($needsEvaluation) { Invoke-NativeStage 'evaluate' }
Invoke-NativeStage 'report'
$report = Get-Content -LiteralPath "$Output/local-report.json" -Raw | ConvertFrom-Json
if (-not $report.qualified_for_server_scaleup) {
    Write-Output 'Local evidence is preserved; validation has not qualified this run for server scale-up.'
    exit 3
}
