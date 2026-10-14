# GRU-Attention experiment: 81.16% position pass rate

This snapshot contains the saved model, matching preprocessing artifact,
per-point test predictions, source-level summary, and key plots for
`exp1_gru_highfreq_300`. The model was trained on the 17 original JSON files.
The exact training source is preserved as
`lstm_keras/exp1_gru_highfreq_300.py`. The committed `lstm_keras/main.py`
uses the same source. An existing working tree may contain newer,
uncommitted experiments, so use the preserved script for reproduction.

## Saved architecture

- Static feature encoder: Dense layers 256, 160, 96, 48.
- Ten-frame sequence encoder: two GRU layers (96, 48) with self-attention
  pooling, followed by a 24-unit projection.
- Static and temporal features are concatenated and passed through Dense
  layers 160 and 80.
- Three separate 32-unit regression heads predict global X, Y, and Z errors.
  An error-classification head is trained jointly.
- The saved model has **no explicit high-frequency residual branch**. The
  experiment's high-frequency channel is the vibration feature group,
  including `band_energy_1000_3000`, enabled by `TEE_USE_VIB_FEATURES=true`.
  It should not be confused with later architecture versions.

The `.keras` layer configuration and `.pkl` feature lists were inspected to
confirm this architecture. The artifact records `TIME_STEPS=10`,
`TEMPORAL_BACKBONE=gru_attention`, `SPLIT_MODE=phase_blocks`, seed 2031,
and sequence grouping by `source_file_name,mapped_file_index`. Grouping by
source alone produces 1,288 test points and is not this experiment.

## Results

Criterion: global 3D Euclidean position error strictly below 0.04 on the
saved 1,279-point test set. Counts were recalculated from the committed
prediction CSV.

| Metric | Passing points | Rate |
| --- | ---: | ---: |
| 3D position | 1038 / 1279 | 81.16% |
| X absolute error < 0.04 | 1178 / 1279 | 92.10% |
| Y absolute error < 0.04 | 1153 / 1279 | 90.15% |
| Z absolute error < 0.04 | 1221 / 1279 | 95.47% |

Mean 3D position error: 0.030162. A fresh CPU training run with the preserved
source and the settings below produced the same 1038/1279 result. Its
prediction CSV was byte-for-byte identical to the archived CSV (SHA-256
`adc7f8c344de36aaf017837b9db2eee74343de2cbdc8e191132c56410d944f3c`),
and the saved Keras weight payload had the same SHA-256 in both runs
(`0bf79f29f918dc199b0008acc315e846af0dde11a87ecdc59e24ae723ae9283e`).
The single-source 88.16% snapshot in the
repository uses only 76 points from `模型10.json` and is not the same test.

## Evaluate the saved model

Run these PowerShell commands from the repository root. Direct invocation of
`.venv` Python avoids PowerShell's activation-script policy.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:TEE_TIME_STEPS = "10"
$env:TEE_SPLIT_MODE = "phase_blocks"
$env:TEE_RANDOM_SEED = "2031"
$env:TEE_CONSOLE_OUTPUT = "quiet"
.\.venv\Scripts\python.exe .\lstm_keras\exp1_gru_highfreq_300.py --run-mode predict_only --data-dir .\data\hybrid_json --model-path .\results\exp1_gru_highfreq_300\hybrid_spatiotemporal_model.keras --artifact-path .\results\exp1_gru_highfreq_300\hybrid_spatiotemporal_artifacts.pkl --output-dir .\results\exp1_gru_highfreq_300_recheck --no-open-plots --no-tensorboard
```

The following settings and preserved source were used for the successful
300-epoch reproduction. The output directory keeps the archived files intact.

```powershell
$env:TEE_EPOCHS = "300"
$env:TEE_PATIENCE = "60"
$env:TEE_BATCH_SIZE = "128"
$env:TEE_TIME_STEPS = "10"
$env:TEE_SPLIT_MODE = "phase_blocks"
$env:TEE_RANDOM_SEED = "2031"
$env:TEE_TEMPORAL_BACKBONE = "gru_attention"
$env:TEE_USE_VIB_FEATURES = "true"
$env:TEE_SEQ_DENSE_UNITS = "24"
$env:TEE_FUSED_DENSE_UNITS = "160,80"
$env:TEE_Z_EXTREME_WEIGHT = "1.0"
$env:TEE_SEQ_DROPOUT_RATE = "0.35"
$env:TEE_REDUCE_LR_PATIENCE = "12"
$env:TEE_CONSOLE_OUTPUT = "full"
.\.venv\Scripts\python.exe .\lstm_keras\exp1_gru_highfreq_300.py --run-mode train --data-dir .\data\hybrid_json --output-dir .\results\exp1_gru_highfreq_300_retrain --model-file hybrid_spatiotemporal_model.keras --artifact-file hybrid_spatiotemporal_artifacts.pkl --no-open-plots --no-tensorboard
```
