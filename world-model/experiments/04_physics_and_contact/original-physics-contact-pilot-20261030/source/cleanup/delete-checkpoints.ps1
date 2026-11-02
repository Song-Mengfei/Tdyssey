$ErrorActionPreference = 'Stop'
$cleanupDirectory = Split-Path -Parent $PSCommandPath
$deletionPlan = Get-Content -LiteralPath (Join-Path $cleanupDirectory 'delete-plan-checkpoints.json') -Raw | ConvertFrom-Json
$expectedRoot = '__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts'
$resolvedRoot = (Resolve-Path -LiteralPath $deletionPlan.verified_project_experiment_root).ProviderPath.TrimEnd('\')
if ($resolvedRoot -ne $expectedRoot) { throw 'Unexpected experiment root' }
$receiptPath = Join-Path $cleanupDirectory 'checkpoint-deletion-receipt.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Existing deletion receipt; do not repeat' }
function Assert-CheckedFile($entry) {
    $resolvedFile = (Resolve-Path -LiteralPath $entry.path).ProviderPath
    if (-not $resolvedFile.StartsWith($resolvedRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Path escapes experiment root' }
    $checkedFile = Get-Item -LiteralPath $resolvedFile -Force
    if ($checkedFile.PSIsContainer -or $checkedFile.Extension -ne '.pt') { throw 'Expected exact checkpoint file' }
    $ancestorItem = $checkedFile
    while ($null -ne $ancestorItem) {
        if (($ancestorItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse point in resolved target ancestry' }
        if ($ancestorItem -is [IO.FileInfo]) { $ancestorItem = $ancestorItem.Directory } else { $ancestorItem = $ancestorItem.Parent }
    }
    if ($checkedFile.Length -ne $entry.bytes) { throw 'File size changed since inventory' }
    if ((Get-FileHash -LiteralPath $resolvedFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw 'File identity changed since inventory' }
    return $resolvedFile
}
$validatedFiles = @()
foreach ($candidateEntry in $deletionPlan.files) {
    if ($candidateEntry.active_dependency) { throw 'Active dependency cannot be deleted' }
    $validatedFiles += Assert-CheckedFile $candidateEntry
}
foreach ($protectedEntry in $deletionPlan.protected_checkpoints) { $null = Assert-CheckedFile $protectedEntry }
$freeBefore = (Get-PSDrive -Name C).Free
$bytesBefore = (Get-ChildItem -LiteralPath $resolvedRoot -File -Recurse -Force | Measure-Object -Property Length -Sum).Sum
$deletionReceipt = [ordered]@{
    kind = 'literal-checkpoint-deletion-actual-receipt.v1'
    authorization = (Join-Path (Split-Path -Parent $cleanupDirectory) 'PARENT_HANDOFF.txt')
    verified_experiment_root = $resolvedRoot
    all_paths_resolved_and_no_reparse_ancestry = $true
    all_files_prevalidated_before_any_delete = $true
    artifact_bytes_before = $bytesBefore
    filesystem_free_bytes_before = $freeBefore
    deleted = @()
    actual_deleted_logical_bytes = [long]0
    complete = $false
}
$deletionReceipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $receiptPath -Encoding utf8
for ($entryIndex = 0; $entryIndex -lt $validatedFiles.Count; $entryIndex++) {
    $candidateEntry = $deletionPlan.files[$entryIndex]
    $actualPath = Assert-CheckedFile $candidateEntry
    Remove-Item -LiteralPath $actualPath -Force
    if (Test-Path -LiteralPath $actualPath) { throw 'Deletion did not complete' }
    $deletionReceipt.deleted += [ordered]@{
        path = $actualPath; bytes = [long]$candidateEntry.bytes; sha256 = $candidateEntry.sha256
        category = $candidateEntry.category; reason = $candidateEntry.reason
        deleted_at_utc = [DateTime]::UtcNow.ToString('o'); confirmed_absent = $true
    }
    $deletionReceipt.actual_deleted_logical_bytes += [long]$candidateEntry.bytes
    $deletionReceipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $receiptPath -Encoding utf8
}
$preservedVerification = @()
foreach ($protectedEntry in $deletionPlan.protected_checkpoints) {
    $actualPath = Assert-CheckedFile $protectedEntry
    $preservedVerification += [ordered]@{ path = $actualPath; sha256 = $protectedEntry.sha256; verified_after_deletion = $true }
}
$deletionReceipt.preserved_checkpoints = $preservedVerification
$deletionReceipt.artifact_bytes_after = (Get-ChildItem -LiteralPath $resolvedRoot -File -Recurse -Force | Measure-Object -Property Length -Sum).Sum
$deletionReceipt.filesystem_free_bytes_after = (Get-PSDrive -Name C).Free
$deletionReceipt.complete = $true
$deletionReceipt.completed_at_utc = [DateTime]::UtcNow.ToString('o')
$deletionReceipt.note = 'Logical deleted bytes are exact sum of removed files. Disk-free before/after can include concurrent diagnostic writes. Historical manifests remain historical; removed checkpoint files are no longer reloadable.'
$deletionReceipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{ deleted_files = $deletionReceipt.deleted.Count; actual_deleted_logical_bytes = $deletionReceipt.actual_deleted_logical_bytes; protected_checkpoints_verified = $preservedVerification.Count; receipt = $receiptPath }
