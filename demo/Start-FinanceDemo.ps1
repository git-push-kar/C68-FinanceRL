param([string]$BaseModelPath = '', [int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$demoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $demoRoot
if (-not $BaseModelPath) { $BaseModelPath = $env:FINAL_BASE_DIR }
if (-not $BaseModelPath) {
    $candidates = @(
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
$env:FINAL_FINANCE_POLICY_DIR = Join-Path $demoRoot 'adapters\finance_policy\finance_lora\finance'
$env:FINAL_FINANCE_TEXT_DTYPE = 'fp32'
$env:FINAL_STUB_GENERATION = '0'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
if (-not (Test-Path -LiteralPath (Join-Path $env:FINAL_FINANCE_POLICY_DIR 'adapter_model.safetensors'))) {
    throw 'Selected trading weights are missing. Run git lfs pull from the repository root.'
}
& (Join-Path $demoRoot '.venv\Scripts\python.exe') -m uvicorn api.server:app --host 127.0.0.1 --port $Port
