$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:HF_HOME = Join-Path (Get-Location).Path '.cache/hf'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME 'transformers'
$common = @('-B', 'scripts/evaluate_shapes.py', '--shapes', 'T', 'D', '--pairs', '1',
    '--seed', '100000', '--policy-seed', '42', '--steps', '64', '--seconds', '900', '--threads', '4')
$conditions = @(
    @{ Name = 'A'; Arguments = @('--planner', 'none', '--execute-steps', '8') },
    @{ Name = 'B'; Arguments = @('--planner', 'none', '--execute-steps', '1') },
    @{ Name = 'C'; Arguments = @('--planner', 'simulator', '--execute-steps', '1', '--candidates', '4',
        '--planning-horizon', '8', '--prefix-lengths', '1', '4', '8', '--settle-steps', '2') }
)
foreach ($condition in $conditions) {
    $runArgs = $common + $condition.Arguments + @('--output', "artifacts/td-abc-20261021-$($condition.Name)")
    & .\.venv\Scripts\python.exe @runArgs 2>&1 | Tee-Object "artifacts/reranking-20261021/abc-$($condition.Name)-console.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
