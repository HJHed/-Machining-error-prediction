# Hybrid v10 model artifacts

## `best_full_dataset/`

This is the selected best recorded run on the expanded dataset: trial `612_current_loss_e4.2_p7.5_a1`.

- Evaluation set: 1,288 points
- Euclidean position-error threshold: `< 0.04`
- Pass rate: 80.67% (1,039 pass / 249 fail)
- Mean / median / maximum position distance: 0.031020 / 0.018602 / 0.374633
- Worst source pass rate: 70.93% (`1753.json`)
- Model and preprocessing artifact must be used as a pair.

The recorded tuning configuration is in `lstm_keras/best_current_dataset_config.json`.

## `deployment_snapshot/`

This is the model that produced the latest bundled single-process evaluation, dated 2026-07-23. Its matching outputs are in `results/hybrid_v10/model10_2026-07-23/`.

The single-process evaluation covers 76 points from `模型10.json`: 88.16% are below 0.04, with a mean position distance of 0.023261. This is not directly comparable with the full-dataset tuning score above.
