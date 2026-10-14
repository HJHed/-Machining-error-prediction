# GRU-Attention experiment: 81.16% position pass rate

This snapshot contains the saved model, matching preprocessing artifact,
per-point test predictions, source-level summary, and key plots for
`exp1_gru_highfreq_300`. The model was trained on the 17 original JSON files.
The code base for this model is the repository's committed
`lstm_keras/main.py` before the later local-coordinate and latent-fusion
experiments. Use a fresh checkout of this commit for the commands below;
an existing working tree may contain newer, uncommitted experiments.

## Saved architecture

- Static feature encoder: Dense layers 256, 160, 96, 48.
- Ten-frame sequence encoder: two GRU layers (96, 48) with self-attention
  pooling, followed by a 24-unit projection.
- Static and temporal features are concatenated and passed through Dense
  layers 160 and 80.
- Three separate 32-unit regression heads predict global X, Y, and Z errors.
  An error-classification head is trained jointly.
- The saved model has **no explicit high-frequency residual branch**. The
  experiment name refers to the available high-frequency vibration features;
  it should not be confused with later architecture versions.

The `.keras` layer configuration and `.pkl` feature lists were inspected to
confirm this architecture. The artifact records `TIME_STEPS=10`,
`TEMPORAL_BACKBONE=gru_attention`, `SPLIT_MODE=phase_blocks`, and seed 2031.

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

Mean 3D position error: 0.030162. The single-source 88.16% snapshot in the
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
.\.venv\Scripts\python.exe .\lstm_keras\main.py --run-mode predict_only --data-dir .\data\hybrid_json --model-path .\results\exp1_gru_highfreq_300\hybrid_spatiotemporal_model.keras --artifact-path .\results\exp1_gru_highfreq_300\hybrid_spatiotemporal_artifacts.pkl --output-dir .\results\exp1_gru_highfreq_300_recheck --no-open-plots
```

The repository's committed `lstm_keras/main.py` supports training the same
architecture. A new training run may differ due to optimizer behavior and
early stopping; the archived model and CSV are the evidence for 81.16%.

```powershell
$env:TEE_EPOCHS = "300"
$env:TEE_TIME_STEPS = "10"
$env:TEE_SPLIT_MODE = "phase_blocks"
$env:TEE_RANDOM_SEED = "2031"
$env:TEE_TEMPORAL_BACKBONE = "gru_attention"
$env:TEE_SEQ_DROPOUT_RATE = "0.35"
.\.venv\Scripts\python.exe .\lstm_keras\main.py --run-mode train --data-dir .\data\hybrid_json --output-dir .\results\exp1_gru_highfreq_300_retrain --no-open-plots
```
