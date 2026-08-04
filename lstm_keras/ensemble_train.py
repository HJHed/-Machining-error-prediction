import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = Path(os.environ.get("TEE_ENSEMBLE_OUTPUT_DIR", PROJECT_ROOT / "results" / "ensemble"))
PREDICTION_FILE = "tool_error_test_predictions_v10_hybrid_spatiotemporal.csv"
FAILURE_FILE = "tool_error_test_position_failures_v10_ensemble.csv"
SOURCE_SUMMARY_FILE = "tool_error_test_source_failure_summary_v10_ensemble.csv"
ENSEMBLE_FILE = "tool_error_test_predictions_v10_ensemble.csv"


def parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def run_seed(seed: int, run_dir: Path, main_path: Path, output_dir: Path, data_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "TEE_RUN_MODE": "train",
            "TEE_OUTPUT_DIR": str(output_dir.resolve()),
            "TEE_JSON_DIR": str(data_dir.resolve()),
            "TEE_RANDOM_SEED": str(seed),
            "TEE_SPLIT_MODE": "phase_blocks",
            "TEE_USE_MOE_OUTPUT_HEAD": "false",
            "TEE_USE_CONDITION_GUIDED_MOE": "false",
            "TEE_USE_NEAR_BOUNDARY_RESIDUAL": "false",
        }
    )

    log_path = run_dir / "run.log"
    print(f"\n===== Ensemble seed {seed} =====")
    print(f"Log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            [sys.executable, str(main_path)],
            cwd=str(main_path.parents[1]),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Seed {seed} failed. See {log_path}")

    pred_path = output_dir / PREDICTION_FILE
    if not pred_path.exists():
        raise FileNotFoundError(f"Seed {seed} did not produce predictions: {pred_path}")

    seed_pred_path = run_dir / f"predictions_seed_{seed}.csv"
    shutil.copy2(pred_path, seed_pred_path)
    ratio, mean_distance, fail_count = read_distance_summary(seed_pred_path)
    print(
        f"seed={seed} ratio={ratio:.6f}, mean={mean_distance:.6f}, "
        f"fail={fail_count}"
    )
    return seed_pred_path


def read_distance_summary(path: Path) -> tuple[float, float, int]:
    df = pd.read_csv(path)
    distance = df["actual_position_distance"].to_numpy(dtype=float)
    return float(np.mean(distance < 0.04)), float(np.mean(distance)), int(np.sum(distance >= 0.04))


def assert_same_rows(base: pd.DataFrame, cur: pd.DataFrame, key_cols: list[str], path: Path) -> None:
    if len(base) != len(cur):
        raise ValueError(f"Prediction row count mismatch for {path}: {len(cur)} != {len(base)}")
    for col in key_cols:
        if col not in cur.columns or col not in base.columns:
            raise ValueError(f"Missing key column {col!r} in {path}")
        if not np.array_equal(base[col].to_numpy(), cur[col].to_numpy()):
            raise ValueError(f"Prediction row order/key mismatch at column {col!r} in {path}")


def build_ensemble(seed_prediction_paths: list[Path], output_dir: Path, run_dir: Path) -> None:
    key_cols = [
        "mapped_file_index",
        "mapped_nc_line",
        "mapped_frame_num",
        "mapped_frame_id",
        "point_index",
    ]
    pred_cols = [f"hybrid_pred_{axis}" for axis in ["x", "y", "z"]]
    true_cols = [f"true_{axis}" for axis in ["x", "y", "z"]]

    dfs = [pd.read_csv(path) for path in seed_prediction_paths]
    base = dfs[0].copy()
    for path, df in zip(seed_prediction_paths[1:], dfs[1:]):
        assert_same_rows(base, df, key_cols, path)

    stacked = np.stack([df[pred_cols].to_numpy(dtype=float) for df in dfs], axis=0)
    ensemble_pred = np.mean(stacked, axis=0)
    single_pred = base[pred_cols].to_numpy(dtype=float)
    true = base[true_cols].to_numpy(dtype=float)

    ensemble_distance = np.linalg.norm(ensemble_pred - true, axis=1)
    single_distance = base["actual_position_distance"].to_numpy(dtype=float)
    ensemble_pass = ensemble_distance < 0.04
    single_pass = single_distance < 0.04

    out = base.copy()
    for idx, axis in enumerate(["x", "y", "z"]):
        out[f"ensemble_pred_{axis}"] = ensemble_pred[:, idx]
        out[f"hybrid_pred_{axis}"] = ensemble_pred[:, idx]
        out[f"ensemble_abs_err_{axis}"] = np.abs(ensemble_pred[:, idx] - true[:, idx])
    out["single_actual_position_distance"] = single_distance
    out["actual_position_distance"] = ensemble_distance
    out["ensemble_position_pass"] = ensemble_pass
    out["single_position_pass"] = single_pass
    out["ensemble_rescued"] = (~single_pass) & ensemble_pass
    out["ensemble_hurt"] = single_pass & (~ensemble_pass)

    ensemble_path = output_dir / ENSEMBLE_FILE
    out.to_csv(ensemble_path, index=False, encoding="utf-8-sig")
    shutil.copy2(ensemble_path, run_dir / ENSEMBLE_FILE)

    failure_df = out.loc[~ensemble_pass].copy().sort_values("actual_position_distance", ascending=False)
    failure_path = output_dir / FAILURE_FILE
    failure_df.to_csv(failure_path, index=False, encoding="utf-8-sig")
    shutil.copy2(failure_path, run_dir / FAILURE_FILE)

    if "source_file_name" in out.columns:
        total_by_source = out.groupby("source_file_name").size().rename("total")
        fail_by_source = failure_df.groupby("source_file_name").size().rename("fail")
        source_summary = pd.concat([total_by_source, fail_by_source], axis=1).fillna(0)
        source_summary["fail"] = source_summary["fail"].astype(int)
        source_summary["fail_rate"] = source_summary["fail"] / source_summary["total"]
        source_summary["pass_rate"] = 1.0 - source_summary["fail_rate"]
        source_summary = source_summary.sort_values(["fail_rate", "fail"], ascending=[False, False])
        source_summary_path = output_dir / SOURCE_SUMMARY_FILE
        source_summary.to_csv(source_summary_path, encoding="utf-8-sig")
        shutil.copy2(source_summary_path, run_dir / SOURCE_SUMMARY_FILE)

    print("\n===== Ensemble summary =====")
    print(
        f"single(seed {seed_prediction_paths[0].stem}) ratio={float(np.mean(single_pass)):.6f}, "
        f"fail={int(np.sum(~single_pass))}, mean={float(np.mean(single_distance)):.6f}"
    )
    print(
        f"ensemble ratio={float(np.mean(ensemble_pass)):.6f}, "
        f"fail={int(np.sum(~ensemble_pass))}, mean={float(np.mean(ensemble_distance)):.6f}, "
        f"median={float(np.median(ensemble_distance)):.6f}, max={float(np.max(ensemble_distance)):.6f}"
    )
    print(
        f"rescued={int(np.sum(out['ensemble_rescued']))}, "
        f"hurt={int(np.sum(out['ensemble_hurt']))}, "
        f"net={int(np.sum(out['ensemble_rescued'])) - int(np.sum(out['ensemble_hurt']))}"
    )
    print(f"Saved ensemble predictions to: {ensemble_path}")
    print(f"Saved ensemble failures to: {failure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multiple seeds and average predictions.")
    parser.add_argument("--seeds", default="42,2027,2031", help="Comma-separated random seeds.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory used by main.py outputs.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "hybrid_json"), help="Directory containing source JSON files.")
    parser.add_argument("--run-dir", default="", help="Directory to store per-seed logs and copied predictions.")
    parser.add_argument("--skip-train", action="store_true", help="Only ensemble existing per-seed predictions in run-dir.")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    main_path = SCRIPT_DIR / "main.py"
    run_dir = Path(args.run_dir) if args.run_dir else output_dir / "ensemble_runs" / ("seeds_" + "_".join(map(str, seeds)))
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_prediction_paths = []
    for seed in seeds:
        seed_run_dir = run_dir / f"seed_{seed}"
        seed_pred_path = seed_run_dir / f"predictions_seed_{seed}.csv"
        if args.skip_train:
            if not seed_pred_path.exists():
                raise FileNotFoundError(f"Missing existing seed prediction: {seed_pred_path}")
            seed_prediction_paths.append(seed_pred_path)
        else:
            seed_prediction_paths.append(run_seed(seed, seed_run_dir, main_path, output_dir, data_dir))

    build_ensemble(seed_prediction_paths, output_dir, run_dir)


if __name__ == "__main__":
    main()
