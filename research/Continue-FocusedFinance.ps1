$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\Parth\Documents\C68-FinanceRL'
$workspace = 'C:\Users\Parth\Documents\PolyvalentFinance'
$python = 'C:\Users\Parth\Documents\PolyvalentDemo\.venv\Scripts\python.exe'
$firstRun = Join-Path $workspace 'campaign_runs\stateful_focused_gate_seed42_20261117'
$nextRun = Join-Path $workspace 'campaign_runs\stateful_numeric_bridge_seed42_20261117'
$preflight = Join-Path $workspace 'campaign_runs\stateful_numeric_bridge_preflight_20261117'
$receiptPath = Join-Path $workspace 'reports\focused_continuation_status.json'
function Save-Receipt($state, $detail) {
    @{ state=$state; detail=$detail; supervisor_pid=$PID; utc=[DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath
}
try {
    Save-Receipt 'waiting_for_focused_learning_gate' 'No second GPU worker until run4 has exited. No deadline or training-time cap.'
    while ($true) {
        $status = Get-Content -LiteralPath (Join-Path $firstRun 'status.json') -Raw | ConvertFrom-Json
        $worker = Get-Process -Id $status.pid -ErrorAction SilentlyContinue
        if (-not $worker) { break }
        Start-Sleep -Seconds 15
    }
    $status = Get-Content -LiteralPath (Join-Path $firstRun 'status.json') -Raw | ConvertFrom-Json
    if ($status.state -ne 'stopped_learning_gate_failed') {
        if ($status.state -eq 'completed') {
            & $python (Join-Path $repo 'research\summarize_stateful_episode.py') $firstRun
            if ($LASTEXITCODE -ne 0) { throw 'Focused completed result audit failed' }
        }
        Save-Receipt 'review_required' @{ first_run_state=$status.state; reason='Fallback is authorized only for the explicit conditional-learning gate failure, not arbitrary crashes or financial losses.' }
        exit 0
    }
    Set-Location -LiteralPath (Join-Path $repo 'research\stateful_rl_20261117_run5')
    if ((Test-Path -LiteralPath $nextRun) -or (Test-Path -LiteralPath $preflight)) { throw 'Fallback artifact directory already exists; refusing duplicate launch' }
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Numeric bridge CPU tests failed; no launch' }
    Save-Receipt 'numeric_bridge_gpu_preflight' 'Focused text policy failed its learning gate; testing numerical input bridge before training.'
    & $python train.py --mode smoke --run $preflight --base (Join-Path $workspace 'local_models\InternVL3-2B')
    if ($LASTEXITCODE -ne 0) { throw 'Numeric bridge GPU preflight failed; no training launch' }
    $check = Get-Content -LiteralPath (Join-Path $preflight 'smoke.json') -Raw | ConvertFrom-Json
    if ($check.status -ne 'passed' -or $check.numeric_bridge_parameter_change -le 0) { throw 'Numeric bridge update/reload receipt invalid' }
    Save-Receipt 'numeric_bridge_training' 'Preflight passed. Same conditional-learning gate, then RL only if the gate passes. No automatic menu deployment.'
    & $python train.py --mode train --run $nextRun --base (Join-Path $workspace 'local_models\InternVL3-2B') --iterations 96 --seed 42
    if ($LASTEXITCODE -ne 0) { throw 'Numeric bridge worker failed; inspect its status and traceback' }
    $final = Get-Content -LiteralPath (Join-Path $nextRun 'status.json') -Raw | ConvertFrom-Json
    if ($final.state -eq 'completed') {
        & $python (Join-Path $repo 'research\summarize_stateful_episode.py') $nextRun
        if ($LASTEXITCODE -ne 0) { throw 'Completed result audit failed' }
    }
    Save-Receipt 'review_required' @{ final_state=$final.state; run=$nextRun; reason='Evaluate learned state response, SFT-vs-RL, and all matched controls before promotion.' }
} catch {
    Save-Receipt 'failed' $_.Exception.Message
    throw
}
