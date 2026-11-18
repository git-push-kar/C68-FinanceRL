param([string]$BaseModelPath = '', [int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$demoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $demoRoot
if (-not $BaseModelPath) { $BaseModelPath = $env:FINAL_BASE_DIR }
if (-not $BaseModelPath) {
    $candidates = @(
        (Join-Path $demoRoot '..\..\PolyvalentFinance\local_models\InternVL3-2B'),
        (Join-Path $demoRoot '..\..\models\InternVL3-2B'),
        (Join-Path $demoRoot '..\models\InternVL3-2B'),
        (Join-Path $demoRoot '..\PolyvalentFinance\local_models\InternVL3-2B')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { $BaseModelPath = $candidate; break }
    }
}
if (-not $BaseModelPath -or -not (Test-Path -LiteralPath $BaseModelPath)) {
    throw 'Supply -BaseModelPath pointing to the pinned full InternVL3-2B snapshot. See PAPER_TRADING.md.'
}
$env:FINAL_BASE_DIR = (Resolve-Path -LiteralPath $BaseModelPath).Path

# Check for internvl3_state_rl or fallback
$stateRLDir = Join-Path $demoRoot 'adapters\internvl3_state_rl'
$legacyDir = Join-Path $demoRoot 'adapters\finance_policy\finance_lora\finance'
if (Test-Path -LiteralPath (Join-Path $stateRLDir 'adapter_model.safetensors')) {
    $env:FINAL_FINANCE_POLICY_DIR = $stateRLDir
} elseif (Test-Path -LiteralPath (Join-Path $legacyDir 'adapter_model.safetensors')) {
    $env:FINAL_FINANCE_POLICY_DIR = $legacyDir
} else {
    $env:FINAL_FINANCE_POLICY_DIR = $stateRLDir
}

$env:FINAL_FINANCE_TEXT_DTYPE = 'fp32'
$env:FINAL_STUB_GENERATION = '0'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

$py = Join-Path $demoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    $py = (Get-Command python).Source
}
& $py -m uvicorn api.server:app --host 127.0.0.1 --port $Port
