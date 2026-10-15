param([int]$Epochs=300)
$ErrorActionPreference='Stop'
$projectPath=Split-Path $PSScriptRoot -Parent
$pythonPath=Join-Path $projectPath '.venv\Scripts\python.exe'
$resultPath=Join-Path $projectPath 'results\exp1_nc_physical_risk_20261015'
$env:TEE_EPOCHS="$Epochs"
$env:TEE_PATIENCE='60'
$env:TEE_BATCH_SIZE='128'
$env:TEE_TIME_STEPS='10'
$env:TEE_SPLIT_MODE='phase_blocks'
$env:TEE_RANDOM_SEED='2031'
$env:TEE_TEMPORAL_BACKBONE='gru_attention'
$env:TEE_SEQUENCE_GROUP_COLS='source_file_name,mapped_file_index'
$env:TEE_USE_VIB_FEATURES='true'
$env:TEE_SEQ_DENSE_UNITS='24'
$env:TEE_FUSED_DENSE_UNITS='160,80'
$env:TEE_Z_EXTREME_WEIGHT='1.0'
$env:TEE_SEQ_DROPOUT_RATE='0.35'
$env:TEE_REDUCE_LR_PATIENCE='12'
$env:TEE_CONSOLE_OUTPUT='full'
$env:TEE_RISK_TEACHER_EPOCHS='100'
$env:TEE_RISK_ALPHA='1.5'
$env:TEE_RISK_BETA='0.5'
$env:TEE_USE_MOE_OUTPUT_HEAD='false'
$env:TEE_USE_RESIDUAL_CORRECTION='false'
$env:TEE_USE_NEAR_THRESHOLD_REWEIGHTING='false'
$env:MPLBACKEND='Agg'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
New-Item -ItemType Directory -Path $resultPath -Force | Out-Null
& $pythonPath (Join-Path $PSScriptRoot 'physical_risk.py') --run-mode train --data-dir (Join-Path $projectPath 'data\hybrid_json') --output-dir $resultPath --model-file 'nc_physical_risk_model.keras' --artifact-file 'motion_weighted_preprocessing.pkl' --no-open-plots --no-tensorboard *> (Join-Path $resultPath 'training.log')
if ($LASTEXITCODE -ne 0) { throw "Training failed. See $resultPath\training.log" }
& $pythonPath (Join-Path $PSScriptRoot 'evaluate_physical.py') --result-dir $resultPath
if ($LASTEXITCODE -ne 0) { throw 'Physical risk evaluation failed' }
& $pythonPath (Join-Path $PSScriptRoot 'audit_physical_triggers.py')
if ($LASTEXITCODE -ne 0) { throw 'Physical risk trigger audit failed' }
