import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MAIN_PATH = SCRIPT_DIR / "main.py"
OUTPUT_DIR = Path(os.environ.get("TEE_AUTO_TUNE_OUTPUT_DIR", PROJECT_ROOT / "results" / "auto_tune"))
PRED_PATH = OUTPUT_DIR / "tool_error_test_predictions_v10_hybrid_spatiotemporal.csv"
FAIL_PATH = OUTPUT_DIR / "tool_error_test_position_failures_v10_hybrid_spatiotemporal.csv"
SOURCE_SUMMARY_PATH = OUTPUT_DIR / "tool_error_test_source_failure_summary_v10.csv"
DIAG_SUMMARY_PATH = OUTPUT_DIR / "tool_error_test_source_diagnostic_summary_v10.csv"
TUNE_DIR = OUTPUT_DIR / "auto_tune_runs"
RESULTS_PATH = TUNE_DIR / "auto_tune_results.csv"


def configure_paths(output_dir: str) -> None:
    """Configure all generated-output paths for a portable tuning run."""
    global OUTPUT_DIR, PRED_PATH, FAIL_PATH, SOURCE_SUMMARY_PATH, DIAG_SUMMARY_PATH, TUNE_DIR, RESULTS_PATH
    OUTPUT_DIR = Path(output_dir).resolve()
    PRED_PATH = OUTPUT_DIR / "tool_error_test_predictions_v10_hybrid_spatiotemporal.csv"
    FAIL_PATH = OUTPUT_DIR / "tool_error_test_position_failures_v10_hybrid_spatiotemporal.csv"
    SOURCE_SUMMARY_PATH = OUTPUT_DIR / "tool_error_test_source_failure_summary_v10.csv"
    DIAG_SUMMARY_PATH = OUTPUT_DIR / "tool_error_test_source_diagnostic_summary_v10.csv"
    TUNE_DIR = OUTPUT_DIR / "auto_tune_runs"
    RESULTS_PATH = TUNE_DIR / "auto_tune_results.csv"
RESULT_FIELDS = [
    "trial_id",
    "status",
    "elapsed_sec",
    "name",
    "euclidean",
    "position",
    "axis_fine",
    "residual_max",
    "residual_scales",
    "test_total",
    "test_fail",
    "test_pass_ratio",
    "distance_mean",
    "distance_median",
    "distance_max",
    "worst_source",
    "worst_source_pass_rate",
    "static_units",
    "static_dropout",
    "temporal_units",
    "temporal_backbone",
    "seq_dropout",
    "seq_dense",
    "fused_units",
    "fused_dropout",
    "axis_head",
    "l2_reg",
    "moe_enabled",
    "moe_expert_count",
    "moe_expert_units",
    "loss_mode",
    "near_enabled",
    "near_epochs",
    "near_patience",
    "near_easy_distance",
    "near_protect_distance",
    "near_rescue_distance",
    "near_easy_weight",
    "near_stable_weight",
    "near_protect_weight",
    "near_rescue_weight",
    "near_far_weight",
    "near_fine_tune_lr",
    "distance_aux_weight",
    "risk_aux_weight",
    "ensemble_size",
    "ensemble_seeds",
    "time_steps",
    "feature_mode",
    "thermal_mode",
    "source_progress_enabled",
    "thermal_enabled",
    "long_rolling_enabled",
    "long_rolling_windows",
    "source_onehot_enabled",
]

PARAM_KEYS = ["euclidean", "position", "axis_fine", "residual_max", "residual_scales"]
DEFAULT_FINE_SCALES = "0,0.1,0.2,0.3,0.4,0.5,0.75,1.0"
DEFAULT_STRUCTURE = {
    "static_units": "128,96,48,24",
    "static_dropout": "0.18,0.15",
    "temporal_units": "48,24",
    "temporal_backbone": "gru_attention",
    "seq_dropout": 0.35,
    "seq_dense": 8,
    "fused_units": "96,48",
    "fused_dropout": 0.16,
    "axis_head": 16,
    "l2_reg": 2e-4,
    "moe_enabled": False,
    "moe_expert_count": 4,
    "moe_expert_units": 24,
}
BEST_BASELINE = {
    "name": "baseline_087",
    "euclidean": 4.1,
    "position": 7.0,
    "axis_fine": 0.8,
    "residual_max": 0.04,
    "residual_scales": "0",
    **DEFAULT_STRUCTURE,
    "loss_mode": "axis_aware",
    "near_enabled": False,
    "near_epochs": 60,
    "near_patience": 15,
    "near_easy_distance": 0.025,
    "near_protect_distance": 0.035,
    "near_rescue_distance": 0.045,
    "near_easy_weight": 0.8,
    "near_stable_weight": 1.0,
    "near_protect_weight": 1.0,
    "near_rescue_weight": 1.8,
    "near_far_weight": 1.0,
    "near_fine_tune_lr": 2e-4,
    "distance_aux_weight": 0.0,
    "risk_aux_weight": 0.0,
    "ensemble_size": 1,
    "ensemble_seeds": "42",
    "time_steps": 10,
    "feature_mode": "minimal",
    "thermal_mode": "summary",
    "source_progress_enabled": True,
    "thermal_enabled": True,
    "long_rolling_enabled": False,
    "long_rolling_windows": "8,20,40",
    "source_onehot_enabled": False,
}

BROAD_SEARCH_SPACE = {
    "euclidean": [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5],
    "position": [1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0, 12.0, 16.0],
    "axis_fine": [0.5, 0.8, 1.0, 1.4, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5],
    "residual_max": [0.0, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
    "residual_scales": [
        "0",
        "0,0.25,0.5,0.75,1.0",
        "0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
    ],
    "static_units": [
        "96,64,32,16",
        "128,96,48,24",
        "160,112,64,32",
        "192,128,64,32",
        "256,128,64,32",
    ],
    "static_dropout": [
        "0.04,0.04",
        "0.08,0.06",
        "0.12,0.10",
        "0.16,0.12",
        "0.22,0.16",
    ],
    "temporal_units": [
        "32,16",
        "48,24",
        "64,32",
        "80,40",
        "96,48",
    ],
    "seq_dropout": [0.10, 0.20, 0.30, 0.40],
    "seq_dense": [8, 16, 24, 32],
    "fused_units": [
        "48,24",
        "64,32",
        "96,48",
        "128,64",
    ],
    "fused_dropout": [0.05, 0.12, 0.20, 0.30],
    "axis_head": [8, 16, 24, 32],
    "l2_reg": [0.0, 5e-5, 1e-4, 2e-4, 5e-4],
}


TRIALS = [
    {
        "name": "baseline",
        "euclidean": 0.5,
        "position": 2.0,
        "axis_fine": 1.0,
        "residual_max": 0.06,
        "residual_scales": "0,0.25,0.5,0.75,1.0",
        **DEFAULT_STRUCTURE,
    },
    {
        "name": "loss_medium",
        "euclidean": 1.0,
        "position": 5.0,
        "axis_fine": 1.5,
        "residual_max": 0.06,
        "residual_scales": "0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
        **DEFAULT_STRUCTURE,
    },
    {
        "name": "loss_strong",
        "euclidean": 1.5,
        "position": 8.0,
        "axis_fine": 2.0,
        "residual_max": 0.06,
        "residual_scales": "0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
        **DEFAULT_STRUCTURE,
    },
    {
        "name": "distance_focus",
        "euclidean": 2.0,
        "position": 6.0,
        "axis_fine": 1.0,
        "residual_max": 0.06,
        "residual_scales": "0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
        **DEFAULT_STRUCTURE,
    },
    {
        "name": "fine_axis_focus",
        "euclidean": 0.8,
        "position": 4.0,
        "axis_fine": 2.5,
        "residual_max": 0.06,
        "residual_scales": "0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
        **DEFAULT_STRUCTURE,
    },
    {
        "name": "residual_conservative",
        "euclidean": 1.0,
        "position": 5.0,
        "axis_fine": 1.5,
        "residual_max": 0.04,
        "residual_scales": "0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
        **DEFAULT_STRUCTURE,
    },
    {
        "name": "residual_wide",
        "euclidean": 1.0,
        "position": 5.0,
        "axis_fine": 1.5,
        "residual_max": 0.08,
        "residual_scales": "0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
        **DEFAULT_STRUCTURE,
    },
    {
        "name": "balanced_high",
        "euclidean": 1.2,
        "position": 4.0,
        "axis_fine": 1.8,
        "residual_max": 0.06,
        "residual_scales": "0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
        **DEFAULT_STRUCTURE,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run automated parameter tuning for main.py.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for trial artifacts and results.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "hybrid_json"), help="Directory containing source JSON files.")
    parser.add_argument("--max-trials", type=int, default=None, help="Only run the first N trials.")
    parser.add_argument("--epochs", type=int, default=None, help="Override EPOCHS for all trials.")
    parser.add_argument("--patience", type=int, default=None, help="Override PATIENCE for all trials.")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this trial index.")
    parser.add_argument("--adaptive", action="store_true", help="Generate new trials around the best prior result.")
    parser.add_argument("--adaptive-count", type=int, default=10, help="Number of adaptive trials to generate.")
    parser.add_argument("--broad", action="store_true", help="Randomly sample a broad parameter space for long runs.")
    parser.add_argument("--near-threshold", action="store_true", help="Tune second-stage near-threshold reweighting around the best baseline.")
    parser.add_argument("--near-refine", action="store_true", help="Focused near-threshold search around the best narrow rescue result.")
    parser.add_argument("--aux-ensemble", action="store_true", help="Tune distance/risk auxiliary heads and optional near-threshold weighting around the best baseline.")
    parser.add_argument("--feature-search", action="store_true", help="Tune feature retention modes plus a small set of stable training parameters.")
    parser.add_argument("--night-search", action="store_true", help="Run an overnight search that warms up with feature ablations, then keeps sampling around current winners.")
    parser.add_argument("--thermal-history-search", action="store_true", help="Focused search for multi-scale thermal history features and temporal window length.")
    parser.add_argument("--thermal-best-refine", action="store_true", help="Deterministic local refinement around the best thermal baseline result.")
    parser.add_argument("--new-data-refine", action="store_true", help="Focused local search around the recorded 0.87 baseline for the expanded dataset.")
    parser.add_argument("--expanded-data-search", action="store_true", help="Alias of --new-data-refine; prioritizes model capacity search for the expanded dataset.")
    parser.add_argument("--continue-from-best-structure", action="store_true", help="Read the best expanded-dataset structure so far and tune regularization/loss around it.")
    parser.add_argument("--overnight-best-refine", action="store_true", help="Dense overnight refinement around the best expanded-dataset trial found so far.")
    parser.add_argument("--current-dataset-refine", action="store_true", help="Refine around the current main.py best recipe after dataset files changed.")
    parser.add_argument("--night-warmup-trials", type=int, default=12, help="Number of fixed feature-ablation trials before adaptive night search.")
    parser.add_argument("--ensemble-size", type=int, default=1, help="Train N seeds per trial and average predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for broad search.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected trials without running training.")
    return parser.parse_args()


def trial_signature(trial: dict) -> tuple:
    return (
        round(float(trial["euclidean"]), 6),
        round(float(trial["position"]), 6),
        round(float(trial["axis_fine"]), 6),
        round(float(trial["residual_max"]), 6),
        str(trial["residual_scales"]),
        str(trial.get("static_units", DEFAULT_STRUCTURE["static_units"])),
        str(trial.get("static_dropout", DEFAULT_STRUCTURE["static_dropout"])),
        str(trial.get("temporal_units", DEFAULT_STRUCTURE["temporal_units"])),
        str(trial.get("temporal_backbone", DEFAULT_STRUCTURE["temporal_backbone"])),
        round(float(trial.get("seq_dropout", DEFAULT_STRUCTURE["seq_dropout"])), 6),
        int(float(trial.get("seq_dense", DEFAULT_STRUCTURE["seq_dense"]))),
        str(trial.get("fused_units", DEFAULT_STRUCTURE["fused_units"])),
        round(float(trial.get("fused_dropout", DEFAULT_STRUCTURE["fused_dropout"])), 6),
        int(float(trial.get("axis_head", DEFAULT_STRUCTURE["axis_head"]))),
        round(float(trial.get("l2_reg", DEFAULT_STRUCTURE["l2_reg"])), 8),
        bool(trial.get("moe_enabled", DEFAULT_STRUCTURE["moe_enabled"])),
        int(float(trial.get("moe_expert_count", DEFAULT_STRUCTURE["moe_expert_count"]))),
        int(float(trial.get("moe_expert_units", DEFAULT_STRUCTURE["moe_expert_units"]))),
        str(trial.get("loss_mode", BEST_BASELINE["loss_mode"])),
        bool(trial.get("near_enabled", BEST_BASELINE["near_enabled"])),
        int(float(trial.get("near_epochs", BEST_BASELINE["near_epochs"]))),
        int(float(trial.get("near_patience", BEST_BASELINE["near_patience"]))),
        round(float(trial.get("near_easy_distance", BEST_BASELINE["near_easy_distance"])), 6),
        round(float(trial.get("near_protect_distance", BEST_BASELINE["near_protect_distance"])), 6),
        round(float(trial.get("near_rescue_distance", BEST_BASELINE["near_rescue_distance"])), 6),
        round(float(trial.get("near_easy_weight", BEST_BASELINE["near_easy_weight"])), 6),
        round(float(trial.get("near_stable_weight", BEST_BASELINE["near_stable_weight"])), 6),
        round(float(trial.get("near_protect_weight", BEST_BASELINE["near_protect_weight"])), 6),
        round(float(trial.get("near_rescue_weight", BEST_BASELINE["near_rescue_weight"])), 6),
        round(float(trial.get("near_far_weight", BEST_BASELINE["near_far_weight"])), 6),
        round(float(trial.get("near_fine_tune_lr", BEST_BASELINE["near_fine_tune_lr"])), 8),
        round(float(trial.get("distance_aux_weight", BEST_BASELINE["distance_aux_weight"])), 8),
        round(float(trial.get("risk_aux_weight", BEST_BASELINE["risk_aux_weight"])), 8),
        int(float(trial.get("ensemble_size", BEST_BASELINE["ensemble_size"]))),
        str(trial.get("ensemble_seeds", BEST_BASELINE["ensemble_seeds"])),
        int(float(trial.get("time_steps", BEST_BASELINE["time_steps"]))),
        str(trial.get("feature_mode", BEST_BASELINE["feature_mode"])),
        str(trial.get("thermal_mode", BEST_BASELINE["thermal_mode"])),
        bool(trial.get("source_progress_enabled", BEST_BASELINE["source_progress_enabled"])),
        bool(trial.get("thermal_enabled", BEST_BASELINE["thermal_enabled"])),
        bool(trial.get("long_rolling_enabled", BEST_BASELINE["long_rolling_enabled"])),
        str(trial.get("long_rolling_windows", BEST_BASELINE["long_rolling_windows"])),
        bool(trial.get("source_onehot_enabled", BEST_BASELINE["source_onehot_enabled"])),
    )


def load_completed_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_PATH)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return df


def completed_signatures() -> set:
    completed = load_completed_results()
    tried = set()
    for _, row in completed.iterrows():
        tried.add(trial_signature(row_to_trial(row)))
    return tried


def successful_trial_count() -> int:
    return len(load_completed_results())


def row_to_trial(row: pd.Series) -> dict:
    trial = {
        "name": str(row.get("name", "prior_best")),
        "euclidean": float(row["euclidean"]),
        "position": float(row["position"]),
        "axis_fine": float(row["axis_fine"]),
        "residual_max": float(row["residual_max"]),
        "residual_scales": str(row["residual_scales"]),
    }
    for key, value in DEFAULT_STRUCTURE.items():
        trial[key] = row.get(key, value)
        if pd.isna(trial[key]):
            trial[key] = value
    for key in ["seq_dense", "axis_head"]:
        trial[key] = int(float(trial[key]))
    for key in ["seq_dropout", "fused_dropout", "l2_reg"]:
        trial[key] = float(trial[key])
    for key, value in BEST_BASELINE.items():
        if key in trial:
            continue
        trial[key] = row.get(key, value)
        if pd.isna(trial[key]):
            trial[key] = value
    for key in ["near_epochs", "near_patience"]:
        trial[key] = int(float(trial[key]))
    for key in [
        "near_easy_distance",
        "near_protect_distance",
        "near_rescue_distance",
        "near_easy_weight",
        "near_stable_weight",
        "near_protect_weight",
        "near_rescue_weight",
        "near_far_weight",
        "near_fine_tune_lr",
        "distance_aux_weight",
        "risk_aux_weight",
    ]:
        trial[key] = float(trial[key])
    trial["ensemble_size"] = int(float(trial["ensemble_size"]))
    trial["ensemble_seeds"] = str(trial["ensemble_seeds"])
    trial["time_steps"] = int(float(trial["time_steps"]))
    for key in ["moe_expert_count", "moe_expert_units"]:
        trial[key] = int(float(trial[key]))
    for key in [
        "moe_enabled",
        "near_enabled",
        "source_progress_enabled",
        "thermal_enabled",
        "long_rolling_enabled",
        "source_onehot_enabled",
    ]:
        if isinstance(trial[key], str):
            trial[key] = trial[key].strip().lower() in {"1", "true", "yes", "on"}
        else:
            trial[key] = bool(trial[key])
    return trial


def build_adaptive_trials(count: int) -> list:
    completed = load_completed_results()
    if completed.empty:
        return TRIALS[:count]

    completed = completed.sort_values(
        ["test_pass_ratio", "distance_mean"],
        ascending=[False, True],
    )
    best = row_to_trial(completed.iloc[0])
    best_ratio = float(completed.iloc[0]["test_pass_ratio"])
    print(
        "Adaptive base:",
        f"name={best['name']}",
        f"ratio={best_ratio:.6f}",
        f"euclidean={best['euclidean']}",
        f"position={best['position']}",
        f"axis_fine={best['axis_fine']}",
        f"residual_max={best['residual_max']}",
    )

    tried = set()
    for _, row in completed.iterrows():
        tried.add(trial_signature(row_to_trial(row)))

    e = best["euclidean"]
    p = best["position"]
    a = best["axis_fine"]
    r = best["residual_max"]
    best_with_defaults = best.copy()
    for key, value in DEFAULT_STRUCTURE.items():
        best_with_defaults.setdefault(key, value)

    top = completed.head(min(5, len(completed)))
    top_e = sorted(set(float(v) for v in top["euclidean"].dropna().tolist()))
    top_p = sorted(set(float(v) for v in top["position"].dropna().tolist()))
    top_a = sorted(set(float(v) for v in top["axis_fine"].dropna().tolist()))
    top_r = sorted(set(float(v) for v in top["residual_max"].dropna().tolist()))

    def extend_from_top(values, step, min_value=0.0):
        out = set(values)
        for value in values:
            out.add(max(min_value, value - step))
            out.add(value + step)
            out.add(value + 2 * step)
        return sorted(out)

    candidate_values = {
        "euclidean": sorted(set(extend_from_top(top_e or [e], 0.2, 0.1) + [
            max(0.1, e - 0.4),
            max(0.1, e - 0.2),
            e,
            e + 0.2,
            e + 0.4,
            e + 0.8,
        ])),
        "position": sorted(set(extend_from_top(top_p or [p], 1.0, 0.5) + [
            max(0.5, p - 2.0),
            max(0.5, p - 1.0),
            p,
            p + 1.0,
            p + 2.0,
            p + 4.0,
        ])),
        "axis_fine": sorted(set(extend_from_top(top_a or [a], 0.3, 0.0) + [
            max(0.0, a - 0.6),
            max(0.0, a - 0.3),
            a,
            a + 0.3,
            a + 0.6,
            a + 0.9,
        ])),
        "residual_max": sorted(set(top_r + [0.04, 0.05, 0.06, 0.08, r])),
    }

    proposals = []

    # Exploit the neighborhood of the best and the best few prior winners.
    for euclidean in candidate_values["euclidean"]:
        for position in candidate_values["position"]:
            for axis_fine in candidate_values["axis_fine"]:
                for residual_max in candidate_values["residual_max"]:
                    distance = (
                        abs(euclidean - e) / max(e, 1e-6)
                        + abs(position - p) / max(p, 1e-6)
                        + abs(axis_fine - a) / max(a, 1e-6)
                        + 0.5 * abs(residual_max - r) / max(r, 1e-6)
                    )
                    if distance > 0.55:
                        continue
                    trial = best_with_defaults.copy()
                    trial.update({
                        "name": f"adaptive_local_e{euclidean:g}_p{position:g}_a{axis_fine:g}_r{residual_max:g}",
                        "euclidean": euclidean,
                        "position": position,
                        "axis_fine": axis_fine,
                        "residual_max": residual_max,
                        "residual_scales": best["residual_scales"],
                    })
                    proposals.append(trial)

    # One-dimensional probes around the best point.
    for key in ["euclidean", "position", "axis_fine", "residual_max"]:
        for value in candidate_values[key]:
            trial = best_with_defaults.copy()
            trial[key] = value
            trial["residual_scales"] = best["residual_scales"]
            trial["name"] = f"adaptive_{key}_{value:g}"
            proposals.append(trial)

    # Fine structure probes around the current best structure.
    best_seq_dropout = float(best_with_defaults["seq_dropout"])
    best_fused_dropout = float(best_with_defaults["fused_dropout"])
    best_l2 = float(best_with_defaults["l2_reg"])
    best_seq_dense = int(float(best_with_defaults["seq_dense"]))
    best_axis_head = int(float(best_with_defaults["axis_head"]))
    best_fused_units = str(best_with_defaults["fused_units"])
    best_temporal_units = str(best_with_defaults["temporal_units"])

    local_structure_variants = []
    for value in sorted(set([
        max(0.0, best_seq_dropout - 0.05),
        best_seq_dropout,
        min(0.6, best_seq_dropout + 0.05),
        min(0.6, best_seq_dropout + 0.10),
    ])):
        local_structure_variants.append({"name": f"best_seq_dropout_{value:g}", "seq_dropout": value})
    for value in sorted(set([
        max(0.0, best_fused_dropout - 0.03),
        best_fused_dropout,
        min(0.5, best_fused_dropout + 0.03),
        min(0.5, best_fused_dropout + 0.07),
    ])):
        local_structure_variants.append({"name": f"best_fused_dropout_{value:g}", "fused_dropout": value})
    for value in sorted(set([
        max(0.0, best_l2 * 0.5),
        best_l2,
        best_l2 * 1.5,
        best_l2 * 2.0,
    ])):
        local_structure_variants.append({"name": f"best_l2_{value:g}", "l2_reg": value})
    for value in sorted(set([max(4, best_seq_dense - 4), best_seq_dense, best_seq_dense + 4, best_seq_dense + 8])):
        local_structure_variants.append({"name": f"best_seq_dense_{value:g}", "seq_dense": value})
    for value in sorted(set([max(4, best_axis_head - 8), best_axis_head, best_axis_head + 8, best_axis_head + 16])):
        local_structure_variants.append({"name": f"best_axis_head_{value:g}", "axis_head": value})
    for value in sorted(set([best_fused_units, "64,32", "96,48", "128,64"])):
        local_structure_variants.append({"name": f"best_fused_{value.replace(',', '_')}", "fused_units": value})
    for value in sorted(set([best_temporal_units, "32,16", "48,24", "64,32", "80,40"])):
        local_structure_variants.append({"name": f"best_temporal_{value.replace(',', '_')}", "temporal_units": value})

    for variant in local_structure_variants:
        trial = best_with_defaults.copy()
        trial.update(variant)
        trial["name"] = f"adaptive_{variant['name']}"
        proposals.append(trial)

    # Wider structural probes are kept as a fallback, but now they start from
    # the best loss settings instead of the original defaults.
    structure_variants = [
        {
            "name": "structure_wider_static",
            "static_units": "160,112,64,32",
            "static_dropout": "0.10,0.08",
        },
        {
            "name": "structure_wider_fused",
            "fused_units": "96,48",
            "fused_dropout": 0.15,
        },
        {
            "name": "structure_lower_dropout",
            "static_dropout": "0.08,0.06",
            "seq_dropout": 0.20,
            "fused_dropout": 0.12,
        },
        {
            "name": "structure_higher_dropout",
            "static_dropout": "0.16,0.12",
            "seq_dropout": 0.35,
            "fused_dropout": 0.25,
        },
        {
            "name": "structure_bigger_temporal",
            "temporal_units": "64,32",
            "seq_dense": 24,
        },
        {
            "name": "structure_bigger_head",
            "axis_head": 24,
        },
        {
            "name": "structure_lower_l2",
            "l2_reg": 5e-5,
        },
        {
            "name": "structure_higher_l2",
            "l2_reg": 2e-4,
        },
        {
            "name": "structure_wide_low_reg",
            "static_units": "160,112,64,32",
            "fused_units": "96,48",
            "static_dropout": "0.08,0.06",
            "seq_dropout": 0.20,
            "fused_dropout": 0.12,
            "l2_reg": 5e-5,
        },
    ]
    for variant in structure_variants:
        trial = best_with_defaults.copy()
        trial.update(variant)
        trial["name"] = f"adaptive_{variant['name']}"
        proposals.append(trial)

    # Local combinations biased around the current best loss-weight region.
    for euclidean in candidate_values["euclidean"]:
        for position in candidate_values["position"]:
            for axis_fine in candidate_values["axis_fine"]:
                distance = (
                    abs(euclidean - e) / max(e, 1e-6)
                    + abs(position - p) / max(p, 1e-6)
                    + abs(axis_fine - a) / max(a, 1e-6)
                )
                if distance > 0.85:
                    continue
                trial = best_with_defaults.copy()
                trial.update({
                    "name": f"adaptive_e{euclidean:g}_p{position:g}_a{axis_fine:g}",
                    "euclidean": euclidean,
                    "position": position,
                    "axis_fine": axis_fine,
                    "residual_max": r,
                    "residual_scales": best["residual_scales"],
                })
                proposals.append(trial)

    # Prefer closer points first, then progressively wider probes.
    def proposal_score(trial: dict) -> float:
        structure_penalty = 0.0
        for key in [
            "static_units",
            "static_dropout",
            "temporal_units",
            "fused_units",
        ]:
            if str(trial.get(key, best_with_defaults.get(key))) != str(best_with_defaults.get(key)):
                structure_penalty += 0.15
        structure_penalty += 0.5 * abs(float(trial.get("seq_dropout", best_seq_dropout)) - best_seq_dropout)
        structure_penalty += 0.5 * abs(float(trial.get("fused_dropout", best_fused_dropout)) - best_fused_dropout)
        structure_penalty += 20.0 * abs(float(trial.get("l2_reg", best_l2)) - best_l2)
        structure_penalty += 0.01 * abs(float(trial.get("seq_dense", best_seq_dense)) - best_seq_dense)
        structure_penalty += 0.01 * abs(float(trial.get("axis_head", best_axis_head)) - best_axis_head)
        return (
            abs(trial["euclidean"] - e) / max(e, 1e-6)
            + abs(trial["position"] - p) / max(p, 1e-6)
            + abs(trial["axis_fine"] - a) / max(a, 1e-6)
            + 0.5 * abs(trial["residual_max"] - r) / max(r, 1e-6)
            + structure_penalty
        )

    unique_trials = []
    seen = set()
    for trial in sorted(proposals, key=proposal_score):
        signature = trial_signature(trial)
        if signature in tried or signature in seen:
            continue
        seen.add(signature)
        unique_trials.append(trial)
        if len(unique_trials) >= count:
            break

    return unique_trials


def build_broad_trials(count: int, seed: int) -> list:
    rng = random.Random(seed)
    tried = completed_signatures()
    trials = []
    seen = set()
    attempts = 0
    max_attempts = max(1000, count * 200)

    while len(trials) < count and attempts < max_attempts:
        attempts += 1
        trial = {
            "name": f"broad_{seed}_{len(trials):03d}",
            "euclidean": rng.choice(BROAD_SEARCH_SPACE["euclidean"]),
            "position": rng.choice(BROAD_SEARCH_SPACE["position"]),
            "axis_fine": rng.choice(BROAD_SEARCH_SPACE["axis_fine"]),
            "residual_max": rng.choice(BROAD_SEARCH_SPACE["residual_max"]),
            "residual_scales": rng.choice(BROAD_SEARCH_SPACE["residual_scales"]),
            "static_units": rng.choice(BROAD_SEARCH_SPACE["static_units"]),
            "static_dropout": rng.choice(BROAD_SEARCH_SPACE["static_dropout"]),
            "temporal_units": rng.choice(BROAD_SEARCH_SPACE["temporal_units"]),
            "seq_dropout": rng.choice(BROAD_SEARCH_SPACE["seq_dropout"]),
            "seq_dense": rng.choice(BROAD_SEARCH_SPACE["seq_dense"]),
            "fused_units": rng.choice(BROAD_SEARCH_SPACE["fused_units"]),
            "fused_dropout": rng.choice(BROAD_SEARCH_SPACE["fused_dropout"]),
            "axis_head": rng.choice(BROAD_SEARCH_SPACE["axis_head"]),
            "l2_reg": rng.choice(BROAD_SEARCH_SPACE["l2_reg"]),
        }

        signature = trial_signature(trial)
        if signature in tried or signature in seen:
            continue
        seen.add(signature)
        trials.append(trial)

    return trials


def build_near_threshold_trials(count: int, seed: int) -> list:
    rng = random.Random(seed)
    tried = completed_signatures()
    trials = []
    seen = set()

    rescue_ranges = [
        (0.040, 0.045),
        (0.040, 0.050),
        (0.040, 0.055),
        (0.040, 0.060),
        (0.042, 0.055),
    ]
    rescue_weights = [1.3, 1.5, 1.8, 2.0, 2.3]
    protect_weights = [1.0, 1.1, 1.2]
    easy_weights = [0.7, 0.8, 0.9]
    far_weights = [0.8, 1.0]
    fine_lrs = [5e-5, 1e-4, 2e-4, 3e-4]
    epochs = [30, 45, 60]
    patience_by_epochs = {30: 10, 45: 12, 60: 15}

    proposals = []
    for rescue_low, rescue_high in rescue_ranges:
        for rescue_weight in rescue_weights:
            for protect_weight in protect_weights:
                for fine_lr in fine_lrs:
                    trial = BEST_BASELINE.copy()
                    trial.update({
                        "name": (
                            f"near_r{rescue_low:.3f}_{rescue_high:.3f}"
                            f"_rw{rescue_weight:g}_pw{protect_weight:g}_lr{fine_lr:g}"
                        ),
                        "near_enabled": True,
                        "near_protect_distance": rescue_low - 0.005,
                        "near_rescue_distance": rescue_high,
                        "near_rescue_weight": rescue_weight,
                        "near_protect_weight": protect_weight,
                        "near_fine_tune_lr": fine_lr,
                    })
                    proposals.append(trial)

    while len(proposals) < count * 3:
        ep = rng.choice(epochs)
        rescue_low, rescue_high = rng.choice(rescue_ranges)
        trial = BEST_BASELINE.copy()
        trial.update({
            "name": f"near_random_{seed}_{len(proposals):03d}",
            "near_enabled": True,
            "near_epochs": ep,
            "near_patience": patience_by_epochs[ep],
            "near_easy_distance": rng.choice([0.020, 0.025, 0.030]),
            "near_protect_distance": rescue_low - rng.choice([0.003, 0.005, 0.008]),
            "near_rescue_distance": rescue_high,
            "near_easy_weight": rng.choice(easy_weights),
            "near_stable_weight": 1.0,
            "near_protect_weight": rng.choice(protect_weights),
            "near_rescue_weight": rng.choice(rescue_weights),
            "near_far_weight": rng.choice(far_weights),
            "near_fine_tune_lr": rng.choice(fine_lrs),
        })
        proposals.append(trial)

    def proposal_score(trial: dict) -> float:
        rescue_high = float(trial["near_rescue_distance"])
        rescue_weight = float(trial["near_rescue_weight"])
        fine_lr = float(trial["near_fine_tune_lr"])
        return (
            abs(rescue_high - 0.050) * 20.0
            + abs(rescue_weight - 1.8)
            + abs(np.log10(fine_lr) - np.log10(2e-4))
            + 0.02 * int(trial["near_epochs"])
        )

    for trial in sorted(proposals, key=proposal_score):
        signature = trial_signature(trial)
        if signature in tried or signature in seen:
            continue
        seen.add(signature)
        trials.append(trial)
        if len(trials) >= count:
            break

    return trials


def build_near_refine_trials(count: int, seed: int) -> list:
    rng = random.Random(seed)
    tried = completed_signatures()
    trials = []
    seen = set()
    model_seed = int(BEST_BASELINE["ensemble_seeds"].split(",")[0])

    rescue_lows = [0.039, 0.040, 0.041, 0.042]
    rescue_highs = [0.043, 0.045, 0.047, 0.050]
    rescue_weights = [1.6, 1.8, 2.0, 2.2]
    protect_weights = [1.0, 1.05, 1.1]
    fine_lrs = [1e-4, 1.5e-4, 2e-4, 3e-4]
    epoch_options = [(30, 10), (45, 12), (60, 15)]

    proposals = []
    for rescue_low in rescue_lows:
        for rescue_high in rescue_highs:
            if rescue_high <= rescue_low:
                continue
            for rescue_weight in rescue_weights:
                for protect_weight in protect_weights:
                    for fine_lr in fine_lrs:
                        for near_epochs, near_patience in epoch_options:
                            trial = BEST_BASELINE.copy()
                            trial.update({
                                "name": (
                                    f"near_refine_l{rescue_low:.3f}_h{rescue_high:.3f}"
                                    f"_rw{rescue_weight:g}_pw{protect_weight:g}"
                                    f"_lr{fine_lr:g}_e{near_epochs}"
                                ),
                                "near_enabled": True,
                                "near_epochs": near_epochs,
                                "near_patience": near_patience,
                                "near_easy_distance": 0.025,
                                "near_protect_distance": max(0.030, rescue_low - 0.005),
                                "near_rescue_distance": rescue_high,
                                "near_easy_weight": 0.8,
                                "near_stable_weight": 1.0,
                                "near_protect_weight": protect_weight,
                                "near_rescue_weight": rescue_weight,
                                "near_far_weight": 1.0,
                                "near_fine_tune_lr": fine_lr,
                                "distance_aux_weight": 0.0,
                                "risk_aux_weight": 0.0,
                                "ensemble_size": 1,
                                "ensemble_seeds": str(model_seed),
                                "time_steps": 10,
                            })
                            proposals.append(trial)

    while len(proposals) < count * 3:
        near_epochs, near_patience = rng.choice(epoch_options)
        rescue_low = rng.choice(rescue_lows)
        rescue_high = rng.choice(rescue_highs)
        if rescue_high <= rescue_low:
            rescue_high = rescue_low + 0.004
        trial = BEST_BASELINE.copy()
        trial.update({
            "name": f"near_refine_random_{seed}_{len(proposals):03d}",
            "near_enabled": True,
            "near_epochs": near_epochs,
            "near_patience": near_patience,
            "near_easy_distance": rng.choice([0.020, 0.025, 0.030]),
            "near_protect_distance": max(0.030, rescue_low - rng.choice([0.003, 0.005, 0.007])),
            "near_rescue_distance": rescue_high,
            "near_easy_weight": rng.choice([0.7, 0.8, 0.9]),
            "near_stable_weight": 1.0,
            "near_protect_weight": rng.choice(protect_weights),
            "near_rescue_weight": rng.choice(rescue_weights),
            "near_far_weight": rng.choice([0.9, 1.0]),
            "near_fine_tune_lr": rng.choice(fine_lrs),
            "distance_aux_weight": 0.0,
            "risk_aux_weight": 0.0,
            "ensemble_size": 1,
            "ensemble_seeds": str(model_seed),
            "time_steps": 10,
        })
        proposals.append(trial)

    def proposal_score(trial: dict) -> float:
        rescue_low = float(trial["near_protect_distance"]) + 0.005
        rescue_high = float(trial["near_rescue_distance"])
        rescue_weight = float(trial["near_rescue_weight"])
        protect_weight = float(trial["near_protect_weight"])
        fine_lr = float(trial["near_fine_tune_lr"])
        epochs = int(trial["near_epochs"])
        return (
            abs(rescue_low - 0.040) * 120.0
            + abs(rescue_high - 0.045) * 120.0
            + abs(rescue_weight - 1.8) * 0.8
            + abs(protect_weight - 1.0) * 0.5
            + abs(np.log10(fine_lr) - np.log10(2e-4)) * 0.5
            + abs(epochs - 60) * 0.005
        )

    for trial in sorted(proposals, key=proposal_score):
        signature = trial_signature(trial)
        if signature in tried or signature in seen:
            continue
        seen.add(signature)
        trials.append(trial)
        if len(trials) >= count:
            break

    return trials


def build_aux_ensemble_trials(count: int, seed: int, ensemble_size: int) -> list:
    rng = random.Random(seed)
    tried = completed_signatures()
    trials = []
    seen = set()

    distance_weights = [0.01, 0.03, 0.05, 0.08, 0.12]
    risk_weights = [0.01, 0.03, 0.05, 0.08, 0.12]
    near_variants = [
        None,
        {"near_rescue_distance": 0.050, "near_rescue_weight": 1.5, "near_protect_weight": 1.0, "near_fine_tune_lr": 1e-4},
        {"near_rescue_distance": 0.050, "near_rescue_weight": 1.8, "near_protect_weight": 1.1, "near_fine_tune_lr": 2e-4},
        {"near_rescue_distance": 0.055, "near_rescue_weight": 1.8, "near_protect_weight": 1.0, "near_fine_tune_lr": 2e-4},
        {"near_rescue_distance": 0.060, "near_rescue_weight": 1.5, "near_protect_weight": 1.0, "near_fine_tune_lr": 1e-4},
    ]
    seeds = ",".join(str(seed + i * 17) for i in range(max(1, ensemble_size)))

    proposals = []
    for distance_weight in distance_weights:
        for risk_weight in risk_weights:
            for near_variant in near_variants:
                trial = BEST_BASELINE.copy()
                near_name = "none"
                if near_variant is not None:
                    trial.update(near_variant)
                    trial["near_enabled"] = True
                    trial["near_protect_distance"] = 0.035
                    near_name = f"near{trial['near_rescue_distance']:.3f}_rw{trial['near_rescue_weight']:g}"
                trial.update({
                    "name": f"aux_dw{distance_weight:g}_rw{risk_weight:g}_{near_name}_ens{ensemble_size}",
                    "distance_aux_weight": distance_weight,
                    "risk_aux_weight": risk_weight,
                    "ensemble_size": ensemble_size,
                    "ensemble_seeds": seeds,
                })
                proposals.append(trial)

    while len(proposals) < count * 3:
        trial = BEST_BASELINE.copy()
        near_enabled = rng.choice([False, True])
        if near_enabled:
            trial.update({
                "near_enabled": True,
                "near_protect_distance": rng.choice([0.032, 0.035, 0.037]),
                "near_rescue_distance": rng.choice([0.045, 0.050, 0.055, 0.060]),
                "near_rescue_weight": rng.choice([1.3, 1.5, 1.8, 2.0]),
                "near_protect_weight": rng.choice([1.0, 1.1, 1.2]),
                "near_fine_tune_lr": rng.choice([5e-5, 1e-4, 2e-4]),
                "near_epochs": rng.choice([30, 45, 60]),
            })
        distance_weight = rng.choice(distance_weights)
        risk_weight = rng.choice(risk_weights)
        trial.update({
            "name": f"aux_random_{seed}_{len(proposals):03d}_ens{ensemble_size}",
            "distance_aux_weight": distance_weight,
            "risk_aux_weight": risk_weight,
            "ensemble_size": ensemble_size,
            "ensemble_seeds": seeds,
        })
        proposals.append(trial)

    def proposal_score(trial: dict) -> float:
        score = abs(float(trial["distance_aux_weight"]) - 0.05)
        score += abs(float(trial["risk_aux_weight"]) - 0.05)
        if bool(trial.get("near_enabled", False)):
            score += 0.05
            score += abs(float(trial["near_rescue_distance"]) - 0.050)
        return score

    for trial in sorted(proposals, key=proposal_score):
        signature = trial_signature(trial)
        if signature in tried or signature in seen:
            continue
        seen.add(signature)
        trials.append(trial)
        if len(trials) >= count:
            break

    return trials


def compute_metrics(pred_path: Path, fail_path: Path, source_path: Path) -> dict:
    pred_df = pd.read_csv(pred_path)
    fail_df = pd.read_csv(fail_path) if fail_path.exists() else pred_df.iloc[0:0]
    pass_ratio = 1.0 - len(fail_df) / max(len(pred_df), 1)

    true = pred_df[["true_x", "true_y", "true_z"]].to_numpy()
    pred = pred_df[["hybrid_pred_x", "hybrid_pred_y", "hybrid_pred_z"]].to_numpy()
    dist = np.linalg.norm(pred - true, axis=1)

    source_df = pd.read_csv(source_path) if source_path.exists() else pd.DataFrame()
    worst_source = ""
    worst_source_pass_rate = np.nan
    if len(source_df) > 0 and "pass_rate" in source_df.columns:
        worst = source_df.sort_values("pass_rate", ascending=True).iloc[0]
        worst_source = str(worst["source_file_name"])
        worst_source_pass_rate = float(worst["pass_rate"])

    return {
        "test_total": int(len(pred_df)),
        "test_fail": int(len(fail_df)),
        "test_pass_ratio": float(pass_ratio),
        "distance_mean": float(np.mean(dist)),
        "distance_median": float(np.median(dist)),
        "distance_max": float(np.max(dist)),
        "worst_source": worst_source,
        "worst_source_pass_rate": worst_source_pass_rate,
    }


def compute_metrics_from_prediction_df(pred_df: pd.DataFrame) -> dict:
    true = pred_df[["true_x", "true_y", "true_z"]].to_numpy()
    pred = pred_df[["hybrid_pred_x", "hybrid_pred_y", "hybrid_pred_z"]].to_numpy()
    dist = np.linalg.norm(pred - true, axis=1)
    fail_count = int(np.sum(dist >= 0.04))

    worst_source = ""
    worst_source_pass_rate = np.nan
    if "source_file_name" in pred_df.columns:
        source_rates = pred_df.assign(_distance=dist).groupby("source_file_name")["_distance"].apply(lambda s: float((s < 0.04).mean()))
        if len(source_rates) > 0:
            worst_source = str(source_rates.idxmin())
            worst_source_pass_rate = float(source_rates.min())

    return {
        "test_total": int(len(pred_df)),
        "test_fail": fail_count,
        "test_pass_ratio": float(1.0 - fail_count / max(len(pred_df), 1)),
        "distance_mean": float(np.mean(dist)),
        "distance_median": float(np.median(dist)),
        "distance_max": float(np.max(dist)),
        "worst_source": worst_source,
        "worst_source_pass_rate": worst_source_pass_rate,
    }


def build_ensemble_prediction(pred_dfs: list) -> pd.DataFrame:
    if len(pred_dfs) == 1:
        return pred_dfs[0].copy()
    out = pred_dfs[0].copy()
    pred_cols = ["hybrid_pred_x", "hybrid_pred_y", "hybrid_pred_z"]
    stacked = np.stack([df[pred_cols].to_numpy(dtype=np.float32) for df in pred_dfs], axis=0)
    averaged = np.mean(stacked, axis=0)
    out.loc[:, pred_cols] = averaged
    true = out[["true_x", "true_y", "true_z"]].to_numpy(dtype=np.float32)
    distances = np.linalg.norm(averaged - true, axis=1)
    out["actual_position_distance"] = distances
    for axis in ["x", "y", "z"]:
        out[f"abs_err_{axis}"] = np.abs(out[f"hybrid_pred_{axis}"] - out[f"true_{axis}"])
        out[f"fine_pass_{axis}"] = out[f"abs_err_{axis}"] < 0.023
    return out


def copy_outputs(trial_dir: Path):
    trial_dir.mkdir(parents=True, exist_ok=True)
    for path in [PRED_PATH, FAIL_PATH, SOURCE_SUMMARY_PATH, DIAG_SUMMARY_PATH]:
        if path.exists():
            shutil.copy2(path, trial_dir / path.name)


def append_result(row: dict):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        existing = pd.read_csv(RESULTS_PATH)
        if list(existing.columns) != RESULT_FIELDS:
            for field in RESULT_FIELDS:
                if field not in existing.columns:
                    existing[field] = np.nan
            existing = existing[RESULT_FIELDS]
            existing.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    write_header = not RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    write_leaderboard()


def write_leaderboard():
    if not RESULTS_PATH.exists():
        return
    df = pd.read_csv(RESULTS_PATH)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if len(df) == 0:
        return
    leaderboard = df.sort_values(["test_pass_ratio", "distance_mean"], ascending=[False, True])
    leaderboard.to_csv(TUNE_DIR / "auto_tune_leaderboard.csv", index=False, encoding="utf-8-sig")


def build_feature_search_trials(count: int, seed: int) -> list:
    rng = random.Random(seed)
    base = BEST_BASELINE.copy()
    base.update({
        "moe_enabled": False,
        "near_enabled": False,
        "source_onehot_enabled": False,
        "source_progress_enabled": True,
        "thermal_enabled": True,
        "long_rolling_enabled": False,
        "time_steps": 10,
        "temporal_backbone": "gru_attention",
    })

    feature_variants = [
        {
            "name": "feat_current_full",
            "feature_mode": "current",
            "thermal_mode": "full",
            "source_progress_enabled": True,
            "thermal_enabled": True,
            "long_rolling_enabled": False,
        },
        {
            "name": "feat_minimal_minimal",
            "feature_mode": "minimal",
            "thermal_mode": "minimal",
            "source_progress_enabled": True,
            "thermal_enabled": True,
            "long_rolling_enabled": False,
        },
        {
            "name": "feat_minimal_summary",
            "feature_mode": "minimal",
            "thermal_mode": "summary",
            "source_progress_enabled": True,
            "thermal_enabled": True,
            "long_rolling_enabled": False,
        },
        {
            "name": "feat_no_rolling_full",
            "feature_mode": "no_rolling",
            "thermal_mode": "full",
            "source_progress_enabled": True,
            "thermal_enabled": True,
            "long_rolling_enabled": False,
        },
        {
            "name": "feat_no_thermal",
            "feature_mode": "no_thermal",
            "thermal_mode": "off",
            "source_progress_enabled": True,
            "thermal_enabled": False,
            "long_rolling_enabled": False,
        },
        {
            "name": "feat_no_context",
            "feature_mode": "no_context",
            "thermal_mode": "off",
            "source_progress_enabled": False,
            "thermal_enabled": False,
            "long_rolling_enabled": False,
        },
        {
            "name": "feat_thermal_only",
            "feature_mode": "thermal_only",
            "thermal_mode": "minimal",
            "source_progress_enabled": True,
            "thermal_enabled": True,
            "long_rolling_enabled": False,
        },
        {
            "name": "feat_long_rolling",
            "feature_mode": "current",
            "thermal_mode": "minimal",
            "source_progress_enabled": True,
            "thermal_enabled": True,
            "long_rolling_enabled": True,
        },
    ]

    regularization_variants = [
        {"name": "reg_base", "l2_reg": 5e-4, "static_dropout": "0.12,0.10", "seq_dropout": 0.40, "fused_dropout": 0.05},
        {"name": "reg_current", "l2_reg": 7.5e-4, "static_dropout": "0.18,0.15", "seq_dropout": 0.40, "fused_dropout": 0.05},
        {"name": "reg_light", "l2_reg": 2e-4, "static_dropout": "0.08,0.06", "seq_dropout": 0.30, "fused_dropout": 0.05},
        {"name": "reg_mid", "l2_reg": 5e-4, "static_dropout": "0.12,0.10", "seq_dropout": 0.30, "fused_dropout": 0.12},
        {"name": "reg_strong", "l2_reg": 1e-3, "static_dropout": "0.18,0.15", "seq_dropout": 0.45, "fused_dropout": 0.12},
    ]

    loss_variants = [
        {"name": "loss_best", "euclidean": 3.5, "position": 5.0, "axis_fine": 0.8},
        {"name": "loss_less_axis", "euclidean": 3.5, "position": 5.0, "axis_fine": 0.4},
        {"name": "loss_more_pos", "euclidean": 3.5, "position": 7.0, "axis_fine": 0.8},
        {"name": "loss_less_pos", "euclidean": 3.0, "position": 4.0, "axis_fine": 0.8},
        {"name": "loss_more_dist", "euclidean": 4.5, "position": 5.0, "axis_fine": 0.8},
    ]

    structure_variants = [
        {"name": "arch_base", "static_units": "128,96,48,24", "temporal_units": "48,24", "fused_units": "96,48"},
        {"name": "arch_small", "static_units": "96,64,32", "temporal_units": "32,16", "fused_units": "64,32"},
        {"name": "arch_temporal64", "static_units": "128,96,48,24", "temporal_units": "64,32", "fused_units": "96,48"},
        {"name": "arch_fused128", "static_units": "128,96,48,24", "temporal_units": "48,24", "fused_units": "128,64"},
    ]

    proposals = []

    # First block: clean feature ablation with the known-good training setup.
    for feature in feature_variants:
        trial = base.copy()
        trial.update(feature)
        trial.update(regularization_variants[0])
        trial.update(loss_variants[0])
        trial.update(structure_variants[0])
        trial["name"] = f"{feature['name']}_{regularization_variants[0]['name']}_{loss_variants[0]['name']}"
        proposals.append(trial)

    # Second block: tune regularization on the most plausible feature modes.
    promising_features = [feature_variants[0], feature_variants[1], feature_variants[2], feature_variants[3]]
    for feature in promising_features:
        for reg in regularization_variants:
            trial = base.copy()
            trial.update(feature)
            trial.update(reg)
            trial.update(loss_variants[0])
            trial.update(structure_variants[0])
            trial["name"] = f"{feature['name']}_{reg['name']}_loss_best"
            proposals.append(trial)

    # Third block: loss and width probes, still source-free and MoE-free.
    for feature in promising_features[:3]:
        for loss in loss_variants:
            trial = base.copy()
            trial.update(feature)
            trial.update(regularization_variants[0])
            trial.update(loss)
            trial.update(structure_variants[0])
            trial["name"] = f"{feature['name']}_reg_base_{loss['name']}"
            proposals.append(trial)
        for arch in structure_variants[1:]:
            trial = base.copy()
            trial.update(feature)
            trial.update(regularization_variants[0])
            trial.update(loss_variants[0])
            trial.update(arch)
            trial["name"] = f"{feature['name']}_reg_base_{arch['name']}"
            proposals.append(trial)

    # Extra random local combinations if the user asks for a longer run.
    while len(proposals) < count:
        feature = rng.choice(promising_features)
        reg = rng.choice(regularization_variants)
        loss = rng.choice(loss_variants)
        arch = rng.choice(structure_variants)
        trial = base.copy()
        trial.update(feature)
        trial.update(reg)
        trial.update(loss)
        trial.update(arch)
        trial["name"] = f"feat_rand_{len(proposals):03d}_{feature['feature_mode']}_{feature['thermal_mode']}_{reg['name']}_{loss['name']}_{arch['name']}"
        proposals.append(trial)

    return proposals[:count]


def build_thermal_history_trials(count: int, seed: int) -> list:
    rng = random.Random(seed)
    base = BEST_BASELINE.copy()
    base.update({
        "name": "thermal_history_base",
        "euclidean": 4.1,
        "position": 7.0,
        "axis_fine": 0.8,
        "residual_max": 0.04,
        "residual_scales": "0",
        "static_units": "128,96,48,24",
        "static_dropout": "0.18,0.15",
        "temporal_units": "48,24",
        "temporal_backbone": "gru_attention",
        "seq_dropout": 0.35,
        "seq_dense": 8,
        "fused_units": "96,48",
        "fused_dropout": 0.16,
        "axis_head": 16,
        "l2_reg": 2e-4,
        "moe_enabled": False,
        "near_enabled": False,
        "source_onehot_enabled": False,
        "source_progress_enabled": True,
        "thermal_enabled": True,
        "loss_mode": "axis_aware",
        "distance_aux_weight": 0.0,
        "risk_aux_weight": 0.0,
        "ensemble_size": 1,
        "ensemble_seeds": str(seed),
    })

    feature_variants = [
        {
            "name": "minimal_summary_no_roll",
            "feature_mode": "minimal",
            "thermal_mode": "summary",
            "long_rolling_enabled": False,
            "long_rolling_windows": "8,20,40",
        },
        {
            "name": "thermal_history_summary_roll",
            "feature_mode": "thermal_history",
            "thermal_mode": "summary",
            "long_rolling_enabled": True,
            "long_rolling_windows": "8,20,40",
        },
        {
            "name": "thermal_history_full_roll",
            "feature_mode": "thermal_history",
            "thermal_mode": "full",
            "long_rolling_enabled": True,
            "long_rolling_windows": "8,20,40",
        },
    ]
    time_steps_variants = [10, 15, 20, 30]
    rolling_window_variants = [
        "5,10,20",
        "8,20,40",
        "10,30,60",
        "15,30,60",
        "20,40,80",
    ]
    reg_variants = [
        {"name": "reg_best", "l2_reg": 2e-4, "static_dropout": "0.18,0.15", "seq_dropout": 0.35, "fused_dropout": 0.16},
        {"name": "reg_light", "l2_reg": 1e-4, "static_dropout": "0.12,0.10", "seq_dropout": 0.25, "fused_dropout": 0.12},
        {"name": "reg_mid", "l2_reg": 3e-4, "static_dropout": "0.18,0.15", "seq_dropout": 0.30, "fused_dropout": 0.16},
        {"name": "reg_strong", "l2_reg": 5e-4, "static_dropout": "0.22,0.16", "seq_dropout": 0.40, "fused_dropout": 0.20},
    ]
    temporal_width_variants = [
        {"name": "gru48", "temporal_units": "48,24", "seq_dense": 8},
        {"name": "gru64", "temporal_units": "64,32", "seq_dense": 16},
        {"name": "gru80", "temporal_units": "80,40", "seq_dense": 16},
    ]
    loss_variants = [
        {"name": "loss_best", "euclidean": 4.1, "position": 7.0, "axis_fine": 0.8},
        {"name": "loss_pos8", "euclidean": 4.1, "position": 8.0, "axis_fine": 0.8},
        {"name": "loss_dist45", "euclidean": 4.5, "position": 7.0, "axis_fine": 0.8},
    ]

    proposals = []

    # Controlled block: compare the feature history modes at each temporal window.
    for feature in feature_variants:
        for time_steps in time_steps_variants:
            trial = base.copy()
            trial.update(feature)
            trial.update(reg_variants[0])
            trial.update(temporal_width_variants[0])
            trial.update(loss_variants[0])
            trial["time_steps"] = time_steps
            trial["name"] = f"thermal_{feature['name']}_ts{time_steps}_{reg_variants[0]['name']}_{temporal_width_variants[0]['name']}"
            proposals.append(trial)

    # Window-scale block: only meaningful when rolling is enabled.
    for windows in rolling_window_variants:
        for time_steps in time_steps_variants:
            trial = base.copy()
            trial.update(feature_variants[1])
            trial.update(reg_variants[0])
            trial.update(temporal_width_variants[0])
            trial.update(loss_variants[0])
            trial["time_steps"] = time_steps
            trial["long_rolling_windows"] = windows
            trial["name"] = f"thermal_roll_{windows.replace(',', '_')}_ts{time_steps}"
            proposals.append(trial)

    # Random local search around the promising thermal-history setup.
    while len(proposals) < count:
        feature = rng.choice(feature_variants[1:])
        reg = rng.choice(reg_variants)
        temporal = rng.choice(temporal_width_variants)
        loss = rng.choice(loss_variants)
        windows = rng.choice(rolling_window_variants)
        time_steps = rng.choice(time_steps_variants)
        trial = base.copy()
        trial.update(feature)
        trial.update(reg)
        trial.update(temporal)
        trial.update(loss)
        trial["time_steps"] = time_steps
        trial["long_rolling_windows"] = windows
        trial["name"] = (
            f"thermal_rand_{len(proposals):03d}_"
            f"{feature['feature_mode']}_{feature['thermal_mode']}_"
            f"ts{time_steps}_roll{windows.replace(',', '_')}_{reg['name']}_{temporal['name']}_{loss['name']}"
        )
        proposals.append(trial)

    return proposals[:count]


def build_thermal_best_refine_trials(count: int, seed: int) -> list:
    base = BEST_BASELINE.copy()
    base.update({
        "name": "thermal_best_refine_base",
        "euclidean": 4.1,
        "position": 7.0,
        "axis_fine": 0.8,
        "residual_max": 0.04,
        "residual_scales": "0",
        "static_units": "128,96,48,24",
        "static_dropout": "0.18,0.15",
        "temporal_units": "48,24",
        "temporal_backbone": "gru_attention",
        "seq_dropout": 0.35,
        "seq_dense": 8,
        "fused_units": "96,48",
        "fused_dropout": 0.16,
        "axis_head": 16,
        "l2_reg": 2e-4,
        "moe_enabled": False,
        "near_enabled": False,
        "source_onehot_enabled": False,
        "source_progress_enabled": True,
        "thermal_enabled": True,
        "feature_mode": "minimal",
        "thermal_mode": "summary",
        "long_rolling_enabled": False,
        "long_rolling_windows": "8,20,40",
        "time_steps": 10,
        "loss_mode": "axis_aware",
        "distance_aux_weight": 0.0,
        "risk_aux_weight": 0.0,
        "ensemble_size": 1,
        "ensemble_seeds": str(seed),
    })

    proposals = []

    def add_trial(name: str, **updates):
        trial = base.copy()
        trial.update(updates)
        trial["name"] = name
        proposals.append(trial)

    # 0. Re-run the current best recipe once with the requested seed as a sanity anchor.
    add_trial("best_anchor")

    # 1. The prior best used TIME_STEPS=10; probe nearby windows before any other tuning.
    for time_steps in [6, 7, 8, 9, 11, 12, 14, 15]:
        add_trial(f"best_ts{time_steps}", time_steps=time_steps)

    # 2. Small loss-weight refinement around the 0.8724 recipe.
    for euclidean in [3.8, 4.0, 4.1, 4.3, 4.5]:
        for position in [6.5, 7.0, 7.5, 8.0]:
            if euclidean == 4.1 and position == 7.0:
                continue
            add_trial(
                f"best_loss_e{euclidean:g}_p{position:g}_a0.8",
                euclidean=euclidean,
                position=position,
                axis_fine=0.8,
            )

    # 3. Regularization nudges; keep architecture fixed so attribution is clean.
    reg_variants = [
        ("reg_l2_0.0001", {"l2_reg": 1e-4}),
        ("reg_l2_0.0003", {"l2_reg": 3e-4}),
        ("reg_l2_0.0004", {"l2_reg": 4e-4}),
        ("reg_seqdrop_0.30", {"seq_dropout": 0.30}),
        ("reg_seqdrop_0.40", {"seq_dropout": 0.40}),
        ("reg_fuseddrop_0.12", {"fused_dropout": 0.12}),
        ("reg_fuseddrop_0.20", {"fused_dropout": 0.20}),
        ("reg_staticdrop_0.15_0.12", {"static_dropout": "0.15,0.12"}),
        ("reg_staticdrop_0.20_0.16", {"static_dropout": "0.20,0.16"}),
    ]
    for name, updates in reg_variants:
        add_trial(f"best_{name}", **updates)

    # 4. Slightly wider/narrower temporal branch. Static/fused branch stays fixed.
    temporal_variants = [
        ("gru40", {"temporal_units": "40,20", "seq_dense": 8}),
        ("gru56", {"temporal_units": "56,28", "seq_dense": 8}),
        ("gru64", {"temporal_units": "64,32", "seq_dense": 16}),
        ("gru80", {"temporal_units": "80,40", "seq_dense": 16}),
    ]
    for name, updates in temporal_variants:
        add_trial(f"best_{name}", **updates)

    # 5. Feature-history probes kept late because the current best did not need rolling.
    feature_variants = [
        ("thermal_minimal", {"thermal_mode": "minimal"}),
        ("thermal_full", {"thermal_mode": "full"}),
        ("history_no_roll", {"feature_mode": "thermal_history", "thermal_mode": "summary", "long_rolling_enabled": False}),
        ("history_roll_5_10_20", {"feature_mode": "thermal_history", "thermal_mode": "summary", "long_rolling_enabled": True, "long_rolling_windows": "5,10,20"}),
        ("history_roll_8_20_40", {"feature_mode": "thermal_history", "thermal_mode": "summary", "long_rolling_enabled": True, "long_rolling_windows": "8,20,40"}),
        ("history_roll_10_30_60", {"feature_mode": "thermal_history", "thermal_mode": "summary", "long_rolling_enabled": True, "long_rolling_windows": "10,30,60"}),
    ]
    for name, updates in feature_variants:
        add_trial(f"best_{name}", **updates)

    # 6. Pair only the most plausible changes; no broad/random cartesian explosion.
    pair_variants = [
        ("e4.0_p7.5_l2_0.0001", {"euclidean": 4.0, "position": 7.5, "l2_reg": 1e-4}),
        ("e4.3_p7.5_l2_0.0003", {"euclidean": 4.3, "position": 7.5, "l2_reg": 3e-4}),
        ("e4.1_p8_seqdrop0.30", {"position": 8.0, "seq_dropout": 0.30}),
        ("ts11_e4.0_p7.5", {"time_steps": 11, "euclidean": 4.0, "position": 7.5}),
        ("ts12_e4.3_p7.5", {"time_steps": 12, "euclidean": 4.3, "position": 7.5}),
        ("gru64_e4.0_p7.5", {"temporal_units": "64,32", "seq_dense": 16, "euclidean": 4.0, "position": 7.5}),
        ("gru64_l2_0.0003", {"temporal_units": "64,32", "seq_dense": 16, "l2_reg": 3e-4}),
        ("history_no_roll_e4.0_p7.5", {"feature_mode": "thermal_history", "thermal_mode": "summary", "euclidean": 4.0, "position": 7.5}),
    ]
    for name, updates in pair_variants:
        add_trial(f"best_pair_{name}", **updates)

    deduped = []
    seen = set()
    for trial in proposals:
        sig = trial_signature(trial)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(trial)
        if len(deduped) >= count:
            break
    return deduped


def build_new_data_refine_trials(count: int, seed: int) -> list:
    rng = random.Random(seed)
    base = BEST_BASELINE.copy()
    base.update({
        "name": "newdata_087_anchor",
        "source_onehot_enabled": False,
        "moe_enabled": False,
        "near_enabled": False,
        "distance_aux_weight": 0.0,
        "risk_aux_weight": 0.0,
        "residual_max": 0.04,
        "residual_scales": "0",
    })

    proposals = []

    def add_trial(name: str, **updates):
        trial = base.copy()
        trial.update(updates)
        trial["name"] = name
        proposals.append(trial)

    # 0. Always rerun the recorded 0.87 recipe first on the expanded dataset.
    add_trial("newdata_087_anchor")

    # 1. More data can support different model capacity. Probe this before loss grids.
    arch_variants = [
        ("tiny", {"static_units": "64,48,24", "temporal_units": "32,16", "seq_dense": 8, "fused_units": "64,32", "axis_head": 8}),
        ("small", {"static_units": "96,64,32", "temporal_units": "40,20", "seq_dense": 8, "fused_units": "80,40", "axis_head": 12}),
        ("base", {"static_units": "128,96,48,24", "temporal_units": "48,24", "seq_dense": 8, "fused_units": "96,48", "axis_head": 16}),
        ("temporal64", {"static_units": "128,96,48,24", "temporal_units": "64,32", "seq_dense": 16, "fused_units": "96,48", "axis_head": 16}),
        ("temporal80", {"static_units": "128,96,48,24", "temporal_units": "80,40", "seq_dense": 16, "fused_units": "96,48", "axis_head": 16}),
        ("static160", {"static_units": "160,112,64,32", "temporal_units": "48,24", "seq_dense": 8, "fused_units": "112,56", "axis_head": 16}),
        ("balanced160", {"static_units": "160,112,64,32", "temporal_units": "64,32", "seq_dense": 16, "fused_units": "128,64", "axis_head": 24}),
        ("wide192", {"static_units": "192,128,64,32", "temporal_units": "80,40", "seq_dense": 16, "fused_units": "128,64", "axis_head": 24}),
        ("wide256", {"static_units": "256,160,96,48", "temporal_units": "96,48", "seq_dense": 24, "fused_units": "160,80", "axis_head": 32}),
    ]
    for name, updates in arch_variants:
        if name == "base":
            continue
        add_trial(f"newdata_arch_{name}", **updates)

    # 2. Window search. Keep the base architecture first, then test good-looking widths.
    for time_steps in [6, 7, 8, 9, 11, 12, 13, 14, 15, 18, 20, 24]:
        if time_steps == int(base["time_steps"]):
            continue
        add_trial(f"newdata_ts{time_steps}", time_steps=time_steps)

    for arch_name, arch_updates in [arch_variants[3], arch_variants[6], arch_variants[7]]:
        for time_steps in [8, 10, 12, 15, 20]:
            trial_name = f"newdata_arch_{arch_name}_ts{time_steps}"
            add_trial(trial_name, time_steps=time_steps, **arch_updates)

    # 3. Regularization is searched early because the expanded data may tolerate less dropout.
    reg_variants = [
        ("very_light", {"l2_reg": 0.0, "static_dropout": "0.04,0.04", "seq_dropout": 0.15, "fused_dropout": 0.05}),
        ("light", {"l2_reg": 5e-5, "static_dropout": "0.08,0.06", "seq_dropout": 0.25, "fused_dropout": 0.08}),
        ("base", {"l2_reg": 2e-4, "static_dropout": "0.18,0.15", "seq_dropout": 0.35, "fused_dropout": 0.16}),
        ("mid", {"l2_reg": 3e-4, "static_dropout": "0.16,0.12", "seq_dropout": 0.30, "fused_dropout": 0.12}),
        ("strong", {"l2_reg": 5e-4, "static_dropout": "0.22,0.16", "seq_dropout": 0.40, "fused_dropout": 0.20}),
        ("l2_0.00005", {"l2_reg": 5e-5}),
        ("l2_0.0001", {"l2_reg": 1e-4}),
        ("l2_0.0003", {"l2_reg": 3e-4}),
        ("l2_0.0005", {"l2_reg": 5e-4}),
        ("seqdrop_0.25", {"seq_dropout": 0.25}),
        ("seqdrop_0.30", {"seq_dropout": 0.30}),
        ("seqdrop_0.40", {"seq_dropout": 0.40}),
        ("fuseddrop_0.10", {"fused_dropout": 0.10}),
        ("fuseddrop_0.20", {"fused_dropout": 0.20}),
        ("staticdrop_0.12_0.10", {"static_dropout": "0.12,0.10"}),
        ("staticdrop_0.22_0.16", {"static_dropout": "0.22,0.16"}),
    ]
    for name, updates in reg_variants:
        if name == "base":
            continue
        add_trial(f"newdata_reg_{name}", **updates)

    for arch_name, arch_updates in [arch_variants[3], arch_variants[6], arch_variants[7], arch_variants[8]]:
        for reg_name, reg_updates in reg_variants[:5]:
            if reg_name == "base":
                continue
            add_trial(f"newdata_arch_{arch_name}_reg_{reg_name}", **arch_updates, **reg_updates)

    # 4. Loss search around the 0.87 recipe, after architecture has been probed.
    loss_variants = []
    for euclidean in [3.5, 3.8, 4.0, 4.1, 4.3, 4.6, 5.0]:
        for position in [5.5, 6.5, 7.0, 7.5, 8.5, 9.5]:
            for axis_fine in [0.6, 0.8, 1.0, 1.2]:
                if euclidean == 4.1 and position == 7.0 and axis_fine == 0.8:
                    continue
                loss_variants.append((euclidean, position, axis_fine))
    for euclidean, position, axis_fine in loss_variants:
        add_trial(
            f"newdata_loss_e{euclidean:g}_p{position:g}_a{axis_fine:g}",
            euclidean=euclidean,
            position=position,
            axis_fine=axis_fine,
        )

    # 5. A small feature block, still source-free; source id is intentionally excluded.
    feature_variants = [
        ("minimal_minimal", {"feature_mode": "minimal", "thermal_mode": "minimal", "long_rolling_enabled": False}),
        ("minimal_full", {"feature_mode": "minimal", "thermal_mode": "full", "long_rolling_enabled": False}),
        ("current_summary", {"feature_mode": "current", "thermal_mode": "summary", "long_rolling_enabled": False}),
        ("current_full", {"feature_mode": "current", "thermal_mode": "full", "long_rolling_enabled": False}),
        ("history_summary_no_roll", {"feature_mode": "thermal_history", "thermal_mode": "summary", "long_rolling_enabled": False}),
        ("history_summary_roll_5_10_20", {"feature_mode": "thermal_history", "thermal_mode": "summary", "long_rolling_enabled": True, "long_rolling_windows": "5,10,20"}),
        ("history_summary_roll_8_20_40", {"feature_mode": "thermal_history", "thermal_mode": "summary", "long_rolling_enabled": True, "long_rolling_windows": "8,20,40"}),
    ]
    for name, updates in feature_variants:
        add_trial(f"newdata_feat_{name}", **updates)

    # 6. Random local combinations after the controlled probes, for overnight runs.
    while len(proposals) < count:
        trial = base.copy()
        trial.update(rng.choice(feature_variants)[1])
        trial.update(rng.choice(reg_variants)[1])
        trial.update(rng.choice(arch_variants)[1])
        trial.update({
            "time_steps": rng.choice([6, 8, 9, 10, 11, 12, 14, 15, 18, 20, 24]),
            "euclidean": rng.choice([3.2, 3.5, 3.8, 4.0, 4.1, 4.3, 4.6, 5.0, 5.5]),
            "position": rng.choice([5.0, 5.5, 6.5, 7.0, 7.5, 8.5, 9.5, 11.0]),
            "axis_fine": rng.choice([0.6, 0.8, 1.0, 1.2]),
        })
        trial["name"] = (
            f"newdata_rand_{len(proposals):03d}_"
            f"ts{trial['time_steps']}_e{trial['euclidean']:g}_p{trial['position']:g}_a{trial['axis_fine']:g}_"
            f"{trial['feature_mode']}_{trial['thermal_mode']}"
        )
        proposals.append(trial)

    deduped = []
    seen = set()
    for trial in proposals:
        sig = trial_signature(trial)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(trial)
        if len(deduped) >= count:
            break
    return deduped


def _best_expanded_data_trial() -> dict:
    completed = load_completed_results()
    if completed.empty:
        return BEST_BASELINE.copy()

    for col in ["test_pass_ratio", "distance_mean", "test_total"]:
        if col in completed.columns:
            completed[col] = pd.to_numeric(completed[col], errors="coerce")

    mask = (
        completed["status"].eq("ok")
        if "status" in completed.columns
        else pd.Series(True, index=completed.index)
    )
    if "trial_id" in completed.columns:
        mask &= completed["trial_id"].astype(str).str.contains("newdata|expanded", case=False, na=False) | (
            completed.get("test_total", pd.Series(0, index=completed.index)) >= 1000
        )
    elif "test_total" in completed.columns:
        mask &= completed["test_total"] >= 1000

    expanded = completed[mask].copy()
    if expanded.empty:
        return BEST_BASELINE.copy()

    expanded = expanded.sort_values(["test_pass_ratio", "distance_mean"], ascending=[False, True])
    return row_to_trial(expanded.iloc[0])


def build_continue_from_best_structure_trials(count: int, seed: int) -> list:
    rng = random.Random(seed)
    best = _best_expanded_data_trial()
    base = BEST_BASELINE.copy()
    base.update(best)
    base.update({
        "name": "best_structure_anchor",
        "source_onehot_enabled": False,
        "moe_enabled": False,
        "near_enabled": False,
        "distance_aux_weight": 0.0,
        "risk_aux_weight": 0.0,
        "residual_max": 0.04,
        "residual_scales": "0",
    })

    best_static = str(base.get("static_units", DEFAULT_STRUCTURE["static_units"]))
    best_temporal = str(base.get("temporal_units", DEFAULT_STRUCTURE["temporal_units"]))
    best_fused = str(base.get("fused_units", DEFAULT_STRUCTURE["fused_units"]))
    best_time_steps = int(float(base.get("time_steps", BEST_BASELINE["time_steps"])))
    best_e = float(base.get("euclidean", BEST_BASELINE["euclidean"]))
    best_p = float(base.get("position", BEST_BASELINE["position"]))
    best_a = float(base.get("axis_fine", BEST_BASELINE["axis_fine"]))
    best_l2 = float(base.get("l2_reg", DEFAULT_STRUCTURE["l2_reg"]))
    best_seq_drop = float(base.get("seq_dropout", DEFAULT_STRUCTURE["seq_dropout"]))
    best_fused_drop = float(base.get("fused_dropout", DEFAULT_STRUCTURE["fused_dropout"]))

    proposals = []

    def add_trial(name: str, **updates):
        trial = base.copy()
        trial.update(updates)
        trial["name"] = name
        proposals.append(trial)

    # 0. Re-run the current best expanded-data structure as an anchor.
    add_trial("best_structure_anchor")

    # 1. Keep the best structure fixed and tune regularization first.
    reg_variants = [
        ("l2_zero", {"l2_reg": 0.0}),
        ("l2_half", {"l2_reg": max(0.0, best_l2 * 0.5)}),
        ("l2_base", {"l2_reg": best_l2}),
        ("l2_1p5", {"l2_reg": best_l2 * 1.5}),
        ("l2_2x", {"l2_reg": best_l2 * 2.0}),
        ("drop_light", {
            "static_dropout": "0.10,0.08",
            "seq_dropout": max(0.05, best_seq_drop - 0.12),
            "fused_dropout": max(0.0, best_fused_drop - 0.08),
        }),
        ("drop_midlight", {
            "static_dropout": "0.14,0.10",
            "seq_dropout": max(0.05, best_seq_drop - 0.06),
            "fused_dropout": max(0.0, best_fused_drop - 0.04),
        }),
        ("drop_base", {
            "static_dropout": base.get("static_dropout", DEFAULT_STRUCTURE["static_dropout"]),
            "seq_dropout": best_seq_drop,
            "fused_dropout": best_fused_drop,
        }),
        ("drop_strong", {
            "static_dropout": "0.22,0.16",
            "seq_dropout": min(0.55, best_seq_drop + 0.08),
            "fused_dropout": min(0.35, best_fused_drop + 0.06),
        }),
    ]
    for name, updates in reg_variants:
        add_trial(f"beststruct_reg_{name}", **updates)

    # 2. Fine loss search around the current best. This is smaller than the broad grid.
    loss_values = []
    for euclidean in sorted(set([
        round(max(0.5, best_e - 0.6), 4),
        round(max(0.5, best_e - 0.3), 4),
        round(best_e, 4),
        round(best_e + 0.3, 4),
        round(best_e + 0.6, 4),
        3.8,
        4.1,
        4.6,
        5.0,
    ])):
        for position in sorted(set([
            round(max(1.0, best_p - 1.5), 4),
            round(max(1.0, best_p - 0.75), 4),
            round(best_p, 4),
            round(best_p + 0.75, 4),
            round(best_p + 1.5, 4),
            6.5,
            7.0,
            8.5,
            9.5,
        ])):
            for axis_fine in sorted(set([
                round(max(0.0, best_a - 0.2), 4),
                round(best_a, 4),
                round(best_a + 0.2, 4),
                0.8,
                1.0,
            ])):
                loss_values.append((euclidean, position, axis_fine))
    for euclidean, position, axis_fine in loss_values:
        add_trial(
            f"beststruct_loss_e{euclidean:g}_p{position:g}_a{axis_fine:g}",
            euclidean=euclidean,
            position=position,
            axis_fine=axis_fine,
        )

    # 3. Time-window probes around the best structure only.
    for time_steps in sorted(set([
        max(4, best_time_steps - 2),
        max(4, best_time_steps - 1),
        best_time_steps,
        best_time_steps + 1,
        best_time_steps + 2,
        best_time_steps + 4,
        8,
        9,
        10,
        12,
    ])):
        if time_steps == best_time_steps:
            continue
        add_trial(f"beststruct_ts{time_steps}", time_steps=time_steps)

    # 4. Small neighbor structures. Keep them close to the winner.
    neighbor_structures = [
        ("winner", {"static_units": best_static, "temporal_units": best_temporal, "fused_units": best_fused}),
        ("slightly_smaller", {"static_units": "192,128,64,32", "temporal_units": "80,40", "fused_units": "128,64"}),
        ("balanced160", {"static_units": "160,112,64,32", "temporal_units": "64,32", "fused_units": "128,64"}),
        ("temporal128", {"static_units": best_static, "temporal_units": "128,64", "seq_dense": 32, "fused_units": best_fused}),
        ("fused192", {"static_units": best_static, "temporal_units": best_temporal, "fused_units": "192,96"}),
    ]
    for name, updates in neighbor_structures[1:]:
        add_trial(f"beststruct_neighbor_{name}", **updates)

    # 5. Pair the best structure with the most plausible regularization/loss changes.
    pair_regs = [reg_variants[1], reg_variants[5], reg_variants[6], reg_variants[8]]
    pair_losses = [
        {"euclidean": best_e, "position": best_p + 0.75, "axis_fine": best_a},
        {"euclidean": best_e + 0.3, "position": best_p, "axis_fine": best_a},
        {"euclidean": best_e, "position": best_p, "axis_fine": min(1.2, best_a + 0.2)},
        {"euclidean": 4.6, "position": 8.5, "axis_fine": 0.8},
    ]
    for reg_name, reg_updates in pair_regs:
        for loss_updates in pair_losses:
            add_trial(
                f"beststruct_pair_{reg_name}_e{loss_updates['euclidean']:g}_p{loss_updates['position']:g}_a{loss_updates['axis_fine']:g}",
                **reg_updates,
                **loss_updates,
            )

    # 6. Random exploitation around the winner if the run is long.
    while len(proposals) < count:
        trial = base.copy()
        _, reg_updates = rng.choice(reg_variants)
        trial.update(reg_updates)
        if rng.random() < 0.2:
            _, struct_updates = rng.choice(neighbor_structures)
            trial.update(struct_updates)
        trial.update({
            "time_steps": rng.choice([max(4, best_time_steps - 1), best_time_steps, best_time_steps + 1, best_time_steps + 2, 9, 10, 12]),
            "euclidean": max(0.5, best_e + rng.choice([-0.6, -0.3, 0.0, 0.3, 0.6])),
            "position": max(1.0, best_p + rng.choice([-1.5, -0.75, 0.0, 0.75, 1.5, 2.5])),
            "axis_fine": max(0.0, best_a + rng.choice([-0.2, 0.0, 0.2, 0.4])),
            "feature_mode": rng.choice(["minimal", "minimal", "current"]),
            "thermal_mode": rng.choice(["summary", "summary", "full"]),
            "long_rolling_enabled": False,
        })
        trial["name"] = (
            f"beststruct_rand_{len(proposals):03d}_"
            f"ts{trial['time_steps']}_e{trial['euclidean']:g}_p{trial['position']:g}_a{trial['axis_fine']:g}"
        )
        proposals.append(trial)

    deduped = []
    seen = completed_signatures()
    local_seen = set()
    for trial in proposals:
        sig = trial_signature(trial)
        if sig in seen or sig in local_seen:
            continue
        local_seen.add(sig)
        deduped.append(trial)
        if len(deduped) >= count:
            break
    return deduped


def build_overnight_best_refine_trials(count: int, seed: int) -> list:
    rng = random.Random(seed)
    best = _best_expanded_data_trial()
    base = BEST_BASELINE.copy()
    base.update(best)
    base.update({
        "name": "overnight_best_anchor",
        "source_onehot_enabled": False,
        "moe_enabled": False,
        "near_enabled": False,
        "distance_aux_weight": 0.0,
        "risk_aux_weight": 0.0,
        "residual_max": 0.04,
        "residual_scales": "0",
        "feature_mode": base.get("feature_mode", "minimal") if "base" in locals() else "minimal",
        "thermal_mode": base.get("thermal_mode", "summary") if "base" in locals() else "summary",
    })

    best_e = float(base.get("euclidean", 3.8))
    best_p = float(base.get("position", 8.5))
    best_a = float(base.get("axis_fine", 1.0))
    best_l2 = float(base.get("l2_reg", 2e-4))
    best_seq_drop = float(base.get("seq_dropout", 0.35))
    best_fused_drop = float(base.get("fused_dropout", 0.16))
    best_time_steps = int(float(base.get("time_steps", 10)))
    best_static_drop = str(base.get("static_dropout", DEFAULT_STRUCTURE["static_dropout"]))

    proposals = []

    def add_trial(name: str, **updates):
        trial = base.copy()
        trial.update(updates)
        trial["name"] = name
        proposals.append(trial)

    add_trial("overnight_best_anchor")

    # 1. Very dense loss refinement around the current winner.
    e_values = sorted(set([
        round(best_e + delta, 3)
        for delta in [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        if best_e + delta > 0
    ]))
    p_values = sorted(set([
        round(best_p + delta, 3)
        for delta in [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]
        if best_p + delta > 0
    ]))
    a_values = sorted(set([
        round(best_a + delta, 3)
        for delta in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        if best_a + delta >= 0
    ]))
    loss_candidates = []
    for euclidean in e_values:
        for position in p_values:
            for axis_fine in a_values:
                distance = (
                    abs(euclidean - best_e) / max(best_e, 1e-6)
                    + abs(position - best_p) / max(best_p, 1e-6)
                    + abs(axis_fine - best_a) / max(best_a, 1e-6)
                )
                loss_candidates.append((distance, euclidean, position, axis_fine))
    loss_candidates.sort(key=lambda item: item[0])
    for _, euclidean, position, axis_fine in loss_candidates[:90]:
        add_trial(
            f"overnight_loss_e{euclidean:g}_p{position:g}_a{axis_fine:g}",
            euclidean=euclidean,
            position=position,
            axis_fine=axis_fine,
        )

    # 2. Dropout/l2 local grid around the winner.
    reg_candidates = [
        ("l2_0", {"l2_reg": 0.0}),
        ("l2_0.00005", {"l2_reg": 5e-5}),
        ("l2_0.0001", {"l2_reg": 1e-4}),
        ("l2_best", {"l2_reg": best_l2}),
        ("l2_0.0003", {"l2_reg": 3e-4}),
        ("l2_0.0004", {"l2_reg": 4e-4}),
        ("drop_0.10_0.22_0.08", {"static_dropout": "0.10,0.08", "seq_dropout": 0.22, "fused_dropout": 0.08}),
        ("drop_0.12_0.26_0.10", {"static_dropout": "0.12,0.10", "seq_dropout": 0.26, "fused_dropout": 0.10}),
        ("drop_0.14_0.30_0.12", {"static_dropout": "0.14,0.10", "seq_dropout": 0.30, "fused_dropout": 0.12}),
        ("drop_best", {"static_dropout": best_static_drop, "seq_dropout": best_seq_drop, "fused_dropout": best_fused_drop}),
        ("drop_0.20_0.40_0.20", {"static_dropout": "0.20,0.16", "seq_dropout": 0.40, "fused_dropout": 0.20}),
    ]
    top_loss_settings = [
        {"euclidean": best_e, "position": best_p, "axis_fine": best_a},
        {"euclidean": max(0.1, best_e - 0.2), "position": best_p, "axis_fine": best_a},
        {"euclidean": best_e, "position": best_p + 0.5, "axis_fine": best_a},
        {"euclidean": best_e, "position": best_p, "axis_fine": best_a + 0.1},
    ]
    for reg_name, reg_updates in reg_candidates:
        for loss_updates in top_loss_settings:
            add_trial(
                f"overnight_reg_{reg_name}_e{loss_updates['euclidean']:g}_p{loss_updates['position']:g}_a{loss_updates['axis_fine']:g}",
                **reg_updates,
                **loss_updates,
            )

    # 3. Window probes near the winner, with the top loss and dropout candidates.
    for time_steps in sorted(set([best_time_steps - 2, best_time_steps - 1, best_time_steps, best_time_steps + 1, best_time_steps + 2, 8, 9, 10, 11, 12, 14])):
        if time_steps < 4:
            continue
        for reg_name, reg_updates in reg_candidates[6:10]:
            add_trial(
                f"overnight_ts{time_steps}_{reg_name}",
                time_steps=time_steps,
                **reg_updates,
            )

    # 4. Small structure neighbors. Only a few, because wide256 is currently clearly best.
    structure_neighbors = [
        ("wide224", {"static_units": "224,144,80,40", "temporal_units": "96,48", "fused_units": "160,80", "seq_dense": 24, "axis_head": 32}),
        ("wide256_temporal112", {"static_units": "256,160,96,48", "temporal_units": "112,56", "fused_units": "160,80", "seq_dense": 24, "axis_head": 32}),
        ("wide256_fused192", {"static_units": "256,160,96,48", "temporal_units": "96,48", "fused_units": "192,96", "seq_dense": 24, "axis_head": 32}),
        ("wide288", {"static_units": "288,192,112,56", "temporal_units": "112,56", "fused_units": "192,96", "seq_dense": 32, "axis_head": 32}),
    ]
    for struct_name, struct_updates in structure_neighbors:
        for loss_updates in top_loss_settings[:3]:
            add_trial(
                f"overnight_struct_{struct_name}_e{loss_updates['euclidean']:g}_p{loss_updates['position']:g}_a{loss_updates['axis_fine']:g}",
                **struct_updates,
                **loss_updates,
            )

    # 5. Random exploitation for the rest of a long overnight run.
    while len(proposals) < count:
        trial = base.copy()
        if rng.random() < 0.18:
            _, struct_updates = rng.choice(structure_neighbors)
            trial.update(struct_updates)
        _, reg_updates = rng.choice(reg_candidates)
        trial.update(reg_updates)
        trial.update({
            "time_steps": rng.choice([8, 9, 10, 10, 10, 11, 12]),
            "euclidean": round(max(0.1, best_e + rng.uniform(-0.35, 0.35)), 3),
            "position": round(max(0.5, best_p + rng.uniform(-1.0, 1.0)), 3),
            "axis_fine": round(max(0.0, best_a + rng.uniform(-0.25, 0.25)), 3),
            "feature_mode": rng.choice(["minimal", "minimal", "minimal", "current"]),
            "thermal_mode": rng.choice(["summary", "summary", "summary", "full"]),
            "long_rolling_enabled": False,
        })
        trial["name"] = (
            f"overnight_rand_{len(proposals):03d}_"
            f"ts{trial['time_steps']}_e{trial['euclidean']:g}_p{trial['position']:g}_a{trial['axis_fine']:g}"
        )
        proposals.append(trial)

    deduped = []
    seen = completed_signatures()
    local_seen = set()
    for trial in proposals:
        sig = trial_signature(trial)
        if sig in seen or sig in local_seen:
            continue
        local_seen.add(sig)
        deduped.append(trial)
        if len(deduped) >= count:
            break
    return deduped


def build_current_dataset_refine_trials(count: int, seed: int) -> list:
    rng = random.Random(seed)
    base = BEST_BASELINE.copy()
    base.update({
        "name": "current_dataset_anchor",
        "euclidean": 3.9,
        "position": 8.0,
        "axis_fine": 1.0,
        "residual_max": 0.04,
        "residual_scales": "0",
        "static_units": "256,160,96,48",
        "static_dropout": "0.18,0.15",
        "temporal_units": "96,48",
        "temporal_backbone": "gru_attention",
        "seq_dropout": 0.35,
        "seq_dense": 24,
        "fused_units": "160,80",
        "fused_dropout": 0.16,
        "axis_head": 32,
        "l2_reg": 2e-4,
        "moe_enabled": False,
        "near_enabled": False,
        "source_onehot_enabled": False,
        "source_progress_enabled": True,
        "thermal_enabled": True,
        "feature_mode": "minimal",
        "thermal_mode": "summary",
        "long_rolling_enabled": False,
        "long_rolling_windows": "8,20,40",
        "time_steps": 10,
        "loss_mode": "axis_aware",
        "distance_aux_weight": 0.0,
        "risk_aux_weight": 0.0,
        "ensemble_size": 1,
        "ensemble_seeds": str(seed),
    })

    proposals = []

    def add_trial(name: str, **updates):
        trial = base.copy()
        trial.update(updates)
        trial["name"] = name
        proposals.append(trial)

    add_trial("current_dataset_anchor")

    # 1. Dense loss refinement for the dataset after removing 0002.json.
    e_values = [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3]
    p_values = [7.0, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.5]
    a_values = [0.8, 0.9, 1.0, 1.1, 1.2]
    loss_candidates = []
    for euclidean in e_values:
        for position in p_values:
            for axis_fine in a_values:
                distance = abs(euclidean - 3.9) / 3.9 + abs(position - 8.0) / 8.0 + abs(axis_fine - 1.0)
                loss_candidates.append((distance, euclidean, position, axis_fine))
    loss_candidates.sort(key=lambda item: item[0])
    for _, euclidean, position, axis_fine in loss_candidates[:90]:
        add_trial(
            f"current_loss_e{euclidean:g}_p{position:g}_a{axis_fine:g}",
            euclidean=euclidean,
            position=position,
            axis_fine=axis_fine,
        )

    # 2. Current data is slightly smaller now; try less/more regularization around wide256.
    reg_variants = [
        ("l2_0", {"l2_reg": 0.0}),
        ("l2_0.00005", {"l2_reg": 5e-5}),
        ("l2_0.0001", {"l2_reg": 1e-4}),
        ("l2_0.0003", {"l2_reg": 3e-4}),
        ("l2_0.0005", {"l2_reg": 5e-4}),
        ("drop_light", {"static_dropout": "0.10,0.08", "seq_dropout": 0.22, "fused_dropout": 0.08}),
        ("drop_midlight", {"static_dropout": "0.14,0.10", "seq_dropout": 0.28, "fused_dropout": 0.12}),
        ("drop_base", {"static_dropout": "0.18,0.15", "seq_dropout": 0.35, "fused_dropout": 0.16}),
        ("drop_strong", {"static_dropout": "0.22,0.16", "seq_dropout": 0.42, "fused_dropout": 0.22}),
    ]
    top_losses = [
        {"euclidean": 3.9, "position": 8.0, "axis_fine": 1.0},
        {"euclidean": 3.8, "position": 8.5, "axis_fine": 1.0},
        {"euclidean": 4.0, "position": 8.25, "axis_fine": 1.0},
        {"euclidean": 3.9, "position": 7.75, "axis_fine": 1.1},
    ]
    for reg_name, reg_updates in reg_variants:
        for loss_updates in top_losses:
            add_trial(
                f"current_reg_{reg_name}_e{loss_updates['euclidean']:g}_p{loss_updates['position']:g}_a{loss_updates['axis_fine']:g}",
                **reg_updates,
                **loss_updates,
            )

    # 3. Window and near-wide structure checks. Keep this limited to avoid wasting the night.
    for time_steps in [8, 9, 10, 11, 12, 14]:
        for loss_updates in top_losses[:3]:
            add_trial(
                f"current_ts{time_steps}_e{loss_updates['euclidean']:g}_p{loss_updates['position']:g}_a{loss_updates['axis_fine']:g}",
                time_steps=time_steps,
                **loss_updates,
            )

    structure_variants = [
        ("wide224", {"static_units": "224,144,80,40", "temporal_units": "96,48", "fused_units": "160,80", "seq_dense": 24, "axis_head": 32}),
        ("wide256_fused192", {"static_units": "256,160,96,48", "temporal_units": "96,48", "fused_units": "192,96", "seq_dense": 24, "axis_head": 32}),
        ("wide256_temporal112", {"static_units": "256,160,96,48", "temporal_units": "112,56", "fused_units": "160,80", "seq_dense": 24, "axis_head": 32}),
        ("wide288", {"static_units": "288,192,112,56", "temporal_units": "112,56", "fused_units": "192,96", "seq_dense": 32, "axis_head": 32}),
    ]
    for struct_name, struct_updates in structure_variants:
        for loss_updates in top_losses[:3]:
            add_trial(
                f"current_struct_{struct_name}_e{loss_updates['euclidean']:g}_p{loss_updates['position']:g}_a{loss_updates['axis_fine']:g}",
                **struct_updates,
                **loss_updates,
            )

    # 4. Random local exploitation for long overnight runs.
    while len(proposals) < count:
        trial = base.copy()
        if rng.random() < 0.18:
            _, struct_updates = rng.choice(structure_variants)
            trial.update(struct_updates)
        _, reg_updates = rng.choice(reg_variants)
        trial.update(reg_updates)
        trial.update({
            "time_steps": rng.choice([8, 9, 10, 10, 10, 11, 12]),
            "euclidean": round(rng.uniform(3.55, 4.25), 3),
            "position": round(rng.uniform(7.25, 9.25), 3),
            "axis_fine": round(rng.uniform(0.85, 1.15), 3),
            "feature_mode": rng.choice(["minimal", "minimal", "current"]),
            "thermal_mode": rng.choice(["summary", "summary", "full"]),
            "long_rolling_enabled": False,
        })
        trial["name"] = (
            f"current_rand_{len(proposals):03d}_"
            f"ts{trial['time_steps']}_e{trial['euclidean']:g}_p{trial['position']:g}_a{trial['axis_fine']:g}"
        )
        proposals.append(trial)

    deduped = []
    local_seen = set()
    for trial in proposals:
        sig = trial_signature(trial)
        if sig in local_seen:
            continue
        local_seen.add(sig)
        deduped.append(trial)
        if len(deduped) >= count:
            break
    return deduped


NIGHT_FEATURE_VARIANTS = [
    {
        "name": "feat_current_full",
        "feature_mode": "current",
        "thermal_mode": "full",
        "source_progress_enabled": True,
        "thermal_enabled": True,
        "long_rolling_enabled": False,
    },
    {
        "name": "feat_minimal_minimal",
        "feature_mode": "minimal",
        "thermal_mode": "minimal",
        "source_progress_enabled": True,
        "thermal_enabled": True,
        "long_rolling_enabled": False,
    },
    {
        "name": "feat_minimal_summary",
        "feature_mode": "minimal",
        "thermal_mode": "summary",
        "source_progress_enabled": True,
        "thermal_enabled": True,
        "long_rolling_enabled": False,
    },
    {
        "name": "feat_no_rolling_full",
        "feature_mode": "no_rolling",
        "thermal_mode": "full",
        "source_progress_enabled": True,
        "thermal_enabled": True,
        "long_rolling_enabled": False,
    },
    {
        "name": "feat_no_thermal",
        "feature_mode": "no_thermal",
        "thermal_mode": "off",
        "source_progress_enabled": True,
        "thermal_enabled": False,
        "long_rolling_enabled": False,
    },
    {
        "name": "feat_no_context",
        "feature_mode": "no_context",
        "thermal_mode": "off",
        "source_progress_enabled": False,
        "thermal_enabled": False,
        "long_rolling_enabled": False,
    },
    {
        "name": "feat_thermal_only",
        "feature_mode": "thermal_only",
        "thermal_mode": "minimal",
        "source_progress_enabled": True,
        "thermal_enabled": True,
        "long_rolling_enabled": False,
    },
]


def make_night_base_trial() -> dict:
    trial = BEST_BASELINE.copy()
    trial.update({
        "moe_enabled": False,
        "near_enabled": False,
        "source_onehot_enabled": False,
        "source_progress_enabled": True,
        "thermal_enabled": True,
        "long_rolling_enabled": False,
        "time_steps": 10,
        "temporal_backbone": "gru_attention",
        "static_units": "128,96,48,24",
        "static_dropout": "0.12,0.10",
        "temporal_units": "48,24",
        "seq_dropout": 0.40,
        "seq_dense": 8,
        "fused_units": "96,48",
        "fused_dropout": 0.05,
        "axis_head": 16,
        "l2_reg": 5e-4,
        "euclidean": 3.5,
        "position": 5.0,
        "axis_fine": 0.8,
        "residual_max": 0.04,
        "residual_scales": "0",
    })
    return trial


def build_night_warmup_trials(count: int, seed: int) -> list:
    return build_feature_search_trials(max(count, 1), seed)[:count]


def _as_bool(value, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _top_night_trials(limit: int = 6) -> list:
    completed = load_completed_results()
    if completed.empty:
        return []

    for field, default in BEST_BASELINE.items():
        if field not in completed.columns:
            completed[field] = default
    mask = (
        completed["status"].eq("ok")
        if "status" in completed.columns
        else pd.Series(True, index=completed.index)
    )
    mask &= completed["source_onehot_enabled"].apply(lambda v: not _as_bool(v, False))
    mask &= completed["moe_enabled"].apply(lambda v: not _as_bool(v, False))
    completed = completed[mask].copy()
    if completed.empty:
        return []
    completed = completed.sort_values(["test_pass_ratio", "distance_mean"], ascending=[False, True])
    return [row_to_trial(row) for _, row in completed.head(limit).iterrows()]


def build_night_adaptive_candidates(seed: int, count: int, tried_signatures: set) -> list:
    rng = random.Random(seed)
    winners = _top_night_trials(limit=6)
    if not winners:
        winners = [make_night_base_trial()]

    feature_pool = []
    for winner in winners[:4]:
        feature_pool.append({
            "name": f"feat_{winner.get('feature_mode', 'current')}_{winner.get('thermal_mode', 'full')}",
            "feature_mode": winner.get("feature_mode", "current"),
            "thermal_mode": winner.get("thermal_mode", "full"),
            "source_progress_enabled": winner.get("source_progress_enabled", True),
            "thermal_enabled": winner.get("thermal_enabled", True),
            "long_rolling_enabled": winner.get("long_rolling_enabled", False),
        })
    feature_pool.extend(NIGHT_FEATURE_VARIANTS[:4])

    proposals = []
    for rank, winner in enumerate(winners):
        base = make_night_base_trial()
        base.update(winner)
        base.update({
            "moe_enabled": False,
            "near_enabled": False,
            "source_onehot_enabled": False,
            "temporal_backbone": "gru_attention",
            "residual_max": 0.04,
            "residual_scales": "0",
        })

        e = float(base.get("euclidean", 3.5))
        p = float(base.get("position", 5.0))
        a = float(base.get("axis_fine", 0.8))
        l2 = float(base.get("l2_reg", 5e-4))
        seq_drop = float(base.get("seq_dropout", 0.40))
        fused_drop = float(base.get("fused_dropout", 0.05))

        local_numeric = [
            {"euclidean": max(0.8, e - 0.4)},
            {"euclidean": e + 0.4},
            {"position": max(2.0, p - 1.0)},
            {"position": p + 1.0},
            {"axis_fine": max(0.0, a - 0.2)},
            {"axis_fine": a + 0.2},
            {"l2_reg": max(0.0, l2 * 0.5)},
            {"l2_reg": l2 * 1.5},
            {"seq_dropout": max(0.15, seq_drop - 0.08)},
            {"seq_dropout": min(0.55, seq_drop + 0.08)},
            {"fused_dropout": max(0.0, fused_drop - 0.04)},
            {"fused_dropout": min(0.30, fused_drop + 0.07)},
            {"temporal_units": "64,32"},
            {"temporal_units": "80,40"},
            {"fused_units": "128,64"},
            {"static_units": "160,112,64,32"},
            {"time_steps": 12},
            {"time_steps": 15},
        ]

        for idx, delta in enumerate(local_numeric):
            trial = base.copy()
            trial.update(delta)
            trial["name"] = f"night_r{rank}_local_{idx}_{list(delta.keys())[0]}"
            proposals.append(trial)

        for feature in feature_pool:
            trial = base.copy()
            trial.update(feature)
            trial["name"] = f"night_r{rank}_{feature['name']}"
            proposals.append(trial)

        for _ in range(18):
            feature = rng.choice(feature_pool)
            trial = base.copy()
            trial.update(feature)
            trial.update({
                "euclidean": max(1.2, e + rng.choice([-0.6, -0.3, 0.0, 0.3, 0.6])),
                "position": max(2.0, p + rng.choice([-1.5, -0.75, 0.0, 0.75, 1.5, 2.0])),
                "axis_fine": max(0.0, a + rng.choice([-0.3, -0.15, 0.0, 0.15, 0.3])),
                "l2_reg": rng.choice([1e-4, 2e-4, 5e-4, 7.5e-4, 1e-3]),
                "static_dropout": rng.choice(["0.08,0.06", "0.12,0.10", "0.16,0.12", "0.18,0.15"]),
                "seq_dropout": rng.choice([0.25, 0.30, 0.35, 0.40, 0.45]),
                "fused_dropout": rng.choice([0.03, 0.05, 0.08, 0.12, 0.16]),
                "temporal_units": rng.choice(["48,24", "64,32", "80,40"]),
                "fused_units": rng.choice(["96,48", "128,64"]),
                "time_steps": rng.choice([10, 12, 15]),
            })
            trial["name"] = (
                f"night_r{rank}_rand_{trial['feature_mode']}_{trial['thermal_mode']}_"
                f"e{trial['euclidean']:g}_p{trial['position']:g}_a{trial['axis_fine']:g}"
            )
            proposals.append(trial)

    # Confirm the current best with a few single-seed reruns late in long searches.
    for rank, winner in enumerate(winners[:3]):
        for seed_value in [seed + 101, seed + 211, seed + 307]:
            trial = make_night_base_trial()
            trial.update(winner)
            trial.update({
                "moe_enabled": False,
                "near_enabled": False,
                "source_onehot_enabled": False,
                "ensemble_size": 1,
                "ensemble_seeds": str(seed_value),
                "name": f"night_confirm_r{rank}_seed{seed_value}",
            })
            proposals.append(trial)

    deduped = []
    seen = set(tried_signatures)
    for trial in proposals:
        sig = trial_signature(trial)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(trial)
        if len(deduped) >= count:
            break
    return deduped


def run_trial(index: int, trial: dict, args) -> dict:
    trial_id = f"{index:03d}_{trial['name']}"
    trial_dir = TUNE_DIR / trial_id
    ensemble_size = max(1, int(trial.get("ensemble_size", args.ensemble_size)))
    seed_values = [
        int(float(seed.strip()))
        for seed in str(trial.get("ensemble_seeds", "")).split(",")
        if seed.strip()
    ]
    if len(seed_values) < ensemble_size:
        seed_values.extend(args.seed + 17 * i for i in range(len(seed_values), ensemble_size))
    seed_values = seed_values[:ensemble_size]

    base_env = os.environ.copy()
    base_env.update(
        {
            "TEE_RUN_MODE": "train",
            "TEE_OUTPUT_DIR": str(OUTPUT_DIR),
            "TEE_JSON_DIR": str(Path(args.data_dir).resolve()),
            "TEE_EUCLIDEAN_DISTANCE_LOSS_WEIGHT": str(trial["euclidean"]),
            "TEE_POSITION_THRESHOLD_LOSS_WEIGHT": str(trial["position"]),
            "TEE_AXIS_FINE_THRESHOLD_LOSS_WEIGHT": str(trial["axis_fine"]),
            "TEE_RESIDUAL_MAX_ABS_CORRECTION": str(trial["residual_max"]),
            "TEE_RESIDUAL_SCALE_CANDIDATES": trial["residual_scales"],
            "TEE_HYBRID_MODEL_FILE": f"hybrid_spatiotemporal_model_v10_{trial_id}.keras",
            "TEE_HYBRID_ARTIFACT_FILE": f"hybrid_spatiotemporal_artifacts_v10_{trial_id}.pkl",
            "TEE_STATIC_DENSE_UNITS": str(trial.get("static_units", DEFAULT_STRUCTURE["static_units"])),
            "TEE_STATIC_DROPOUT_RATES": str(trial.get("static_dropout", DEFAULT_STRUCTURE["static_dropout"])),
            "TEE_TEMPORAL_UNITS": str(trial.get("temporal_units", DEFAULT_STRUCTURE["temporal_units"])),
            "TEE_TEMPORAL_BACKBONE": str(trial.get("temporal_backbone", DEFAULT_STRUCTURE["temporal_backbone"])),
            "TEE_SEQ_DROPOUT_RATE": str(trial.get("seq_dropout", DEFAULT_STRUCTURE["seq_dropout"])),
            "TEE_SEQ_DENSE_UNITS": str(trial.get("seq_dense", DEFAULT_STRUCTURE["seq_dense"])),
            "TEE_FUSED_DENSE_UNITS": str(trial.get("fused_units", DEFAULT_STRUCTURE["fused_units"])),
            "TEE_FUSED_DROPOUT_RATE": str(trial.get("fused_dropout", DEFAULT_STRUCTURE["fused_dropout"])),
            "TEE_AXIS_HEAD_UNITS": str(trial.get("axis_head", DEFAULT_STRUCTURE["axis_head"])),
            "TEE_L2_REG": str(trial.get("l2_reg", DEFAULT_STRUCTURE["l2_reg"])),
            "TEE_USE_MOE_OUTPUT_HEAD": str(trial.get("moe_enabled", DEFAULT_STRUCTURE["moe_enabled"])),
            "TEE_MOE_EXPERT_COUNT": str(trial.get("moe_expert_count", DEFAULT_STRUCTURE["moe_expert_count"])),
            "TEE_MOE_EXPERT_UNITS": str(trial.get("moe_expert_units", DEFAULT_STRUCTURE["moe_expert_units"])),
            "TEE_TIME_STEPS": str(trial.get("time_steps", BEST_BASELINE["time_steps"])),
            "TEE_FEATURE_SET_MODE": str(trial.get("feature_mode", BEST_BASELINE["feature_mode"])),
            "TEE_THERMAL_ACCUMULATION_MODE": str(trial.get("thermal_mode", BEST_BASELINE["thermal_mode"])),
            "TEE_USE_SOURCE_PROGRESS_FEATURES": str(trial.get("source_progress_enabled", BEST_BASELINE["source_progress_enabled"])),
            "TEE_USE_THERMAL_ACCUMULATION_FEATURES": str(trial.get("thermal_enabled", BEST_BASELINE["thermal_enabled"])),
            "TEE_USE_LONG_ROLLING_FEATURES": str(trial.get("long_rolling_enabled", BEST_BASELINE["long_rolling_enabled"])),
            "TEE_LONG_ROLLING_WINDOWS": str(trial.get("long_rolling_windows", BEST_BASELINE["long_rolling_windows"])),
            "TEE_USE_SOURCE_ONEHOT_FEATURES": str(trial.get("source_onehot_enabled", BEST_BASELINE["source_onehot_enabled"])),
            "TEE_REGRESSION_LOSS_MODE": str(trial.get("loss_mode", BEST_BASELINE["loss_mode"])),
            "TEE_USE_NEAR_THRESHOLD_REWEIGHTING": str(trial.get("near_enabled", BEST_BASELINE["near_enabled"])),
            "TEE_NEAR_THRESHOLD_FINE_TUNE_EPOCHS": str(trial.get("near_epochs", BEST_BASELINE["near_epochs"])),
            "TEE_NEAR_THRESHOLD_FINE_TUNE_PATIENCE": str(trial.get("near_patience", BEST_BASELINE["near_patience"])),
            "TEE_NEAR_THRESHOLD_EASY_DISTANCE": str(trial.get("near_easy_distance", BEST_BASELINE["near_easy_distance"])),
            "TEE_NEAR_THRESHOLD_PROTECT_DISTANCE": str(trial.get("near_protect_distance", BEST_BASELINE["near_protect_distance"])),
            "TEE_NEAR_THRESHOLD_RESCUE_DISTANCE": str(trial.get("near_rescue_distance", BEST_BASELINE["near_rescue_distance"])),
            "TEE_NEAR_THRESHOLD_EASY_WEIGHT": str(trial.get("near_easy_weight", BEST_BASELINE["near_easy_weight"])),
            "TEE_NEAR_THRESHOLD_STABLE_WEIGHT": str(trial.get("near_stable_weight", BEST_BASELINE["near_stable_weight"])),
            "TEE_NEAR_THRESHOLD_PROTECT_WEIGHT": str(trial.get("near_protect_weight", BEST_BASELINE["near_protect_weight"])),
            "TEE_NEAR_THRESHOLD_RESCUE_WEIGHT": str(trial.get("near_rescue_weight", BEST_BASELINE["near_rescue_weight"])),
            "TEE_NEAR_THRESHOLD_FAR_WEIGHT": str(trial.get("near_far_weight", BEST_BASELINE["near_far_weight"])),
            "TEE_NEAR_THRESHOLD_FINE_TUNE_LR": str(trial.get("near_fine_tune_lr", BEST_BASELINE["near_fine_tune_lr"])),
            "TEE_DISTANCE_AUX_LOSS_WEIGHT": str(trial.get("distance_aux_weight", BEST_BASELINE["distance_aux_weight"])),
            "TEE_RISK_AUX_LOSS_WEIGHT": str(trial.get("risk_aux_weight", BEST_BASELINE["risk_aux_weight"])),
        }
    )
    if args.epochs is not None:
        base_env["TEE_EPOCHS"] = str(args.epochs)
    if args.patience is not None:
        base_env["TEE_PATIENCE"] = str(args.patience)

    print(f"\n===== Trial {trial_id} =====")
    print(
        "params:",
        f"euclidean={trial['euclidean']}",
        f"position={trial['position']}",
        f"axis_fine={trial['axis_fine']}",
        f"residual_max={trial['residual_max']}",
        f"scales={trial['residual_scales']}",
        f"loss={trial.get('loss_mode', BEST_BASELINE['loss_mode'])}",
        f"near={trial.get('near_enabled', BEST_BASELINE['near_enabled'])}",
        f"rescue<{trial.get('near_rescue_distance', BEST_BASELINE['near_rescue_distance'])}",
        f"rescue_w={trial.get('near_rescue_weight', BEST_BASELINE['near_rescue_weight'])}",
        f"ft_lr={trial.get('near_fine_tune_lr', BEST_BASELINE['near_fine_tune_lr'])}",
        f"dist_aux={trial.get('distance_aux_weight', BEST_BASELINE['distance_aux_weight'])}",
        f"risk_aux={trial.get('risk_aux_weight', BEST_BASELINE['risk_aux_weight'])}",
        f"ensemble={ensemble_size}",
        f"feature={trial.get('feature_mode', BEST_BASELINE['feature_mode'])}",
        f"thermal={trial.get('thermal_mode', BEST_BASELINE['thermal_mode'])}",
        f"time_steps={trial.get('time_steps', BEST_BASELINE['time_steps'])}",
        f"roll={trial.get('long_rolling_enabled', BEST_BASELINE['long_rolling_enabled'])}:{trial.get('long_rolling_windows', BEST_BASELINE['long_rolling_windows'])}",
        f"progress={trial.get('source_progress_enabled', BEST_BASELINE['source_progress_enabled'])}",
        f"source_onehot={trial.get('source_onehot_enabled', BEST_BASELINE['source_onehot_enabled'])}",
        f"static={trial.get('static_units', DEFAULT_STRUCTURE['static_units'])}",
        f"fused={trial.get('fused_units', DEFAULT_STRUCTURE['fused_units'])}",
    )
    start = time.time()
    trial_dir.mkdir(parents=True, exist_ok=True)
    pred_dfs = []
    for seed_idx, seed_value in enumerate(seed_values):
        env = base_env.copy()
        seed_suffix = f"_s{seed_idx}_seed{seed_value}" if ensemble_size > 1 else ""
        env["TEE_RANDOM_SEED"] = str(seed_value)
        env["TEE_HYBRID_MODEL_FILE"] = f"hybrid_spatiotemporal_model_v10_{trial_id}{seed_suffix}.keras"
        env["TEE_HYBRID_ARTIFACT_FILE"] = f"hybrid_spatiotemporal_artifacts_v10_{trial_id}{seed_suffix}.pkl"

        proc = subprocess.run(
            [sys.executable, str(MAIN_PATH)],
            cwd=str(SCRIPT_DIR.parent),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (trial_dir / f"run{seed_suffix or '_single'}.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            elapsed_sec = time.time() - start
            failed_log_name = f"run{seed_suffix or '_single'}.log"
            row = {
                "trial_id": trial_id,
                "status": "failed",
                "elapsed_sec": round(elapsed_sec, 2),
                **trial,
                "ensemble_size": ensemble_size,
                "ensemble_seeds": ",".join(str(v) for v in seed_values),
            }
            append_result(row)
            print(f"Trial failed. See {trial_dir / failed_log_name}")
            return row
        seed_dir = trial_dir / f"seed_{seed_value}"
        copy_outputs(seed_dir)
        pred_dfs.append(pd.read_csv(PRED_PATH))

    elapsed_sec = time.time() - start
    ensemble_pred_df = build_ensemble_prediction(pred_dfs)
    ensemble_pred_path = trial_dir / "ensemble_predictions.csv"
    ensemble_pred_df.to_csv(ensemble_pred_path, index=False, encoding="utf-8-sig")
    if ensemble_size == 1:
        metrics = compute_metrics(PRED_PATH, FAIL_PATH, SOURCE_SUMMARY_PATH)
        copy_outputs(trial_dir)
    else:
        metrics = compute_metrics_from_prediction_df(ensemble_pred_df)
    row = {
        "trial_id": trial_id,
        "status": "ok",
        "elapsed_sec": round(elapsed_sec, 2),
        **trial,
        **metrics,
        "ensemble_size": ensemble_size,
        "ensemble_seeds": ",".join(str(v) for v in seed_values),
    }
    append_result(row)
    print(
        f"ratio={metrics['test_pass_ratio']:.6f}, "
        f"mean={metrics['distance_mean']:.6f}, "
        f"worst={metrics['worst_source']}:{metrics['worst_source_pass_rate']:.6f}"
    )
    return row


def main():
    args = parse_args()
    configure_paths(args.output_dir)
    if args.night_search:
        total_count = args.max_trials if args.max_trials is not None else 100
        warmup_count = min(args.night_warmup_trials, total_count)
        start_index = len(load_completed_results())
        warmup_trials = build_night_warmup_trials(warmup_count, args.seed)
        tried = completed_signatures()
        planned = []
        for trial in warmup_trials:
            sig = trial_signature(trial)
            if sig in tried:
                continue
            tried.add(sig)
            planned.append(trial)
        planned.extend(build_night_adaptive_candidates(args.seed + 999, max(total_count - len(planned), 0), tried))
        trials = planned[:total_count]
        if args.dry_run:
            print("\n===== Planned night-search warmup/adaptive trials =====")
            for idx, trial in enumerate(trials, start=start_index):
                print(
                    f"{idx:03d}_{trial['name']}: "
                    f"ratio_source=dynamic, "
                    f"feature={trial.get('feature_mode', BEST_BASELINE['feature_mode'])}, "
                    f"thermal={trial.get('thermal_mode', BEST_BASELINE['thermal_mode'])}, "
                    f"euclidean={trial['euclidean']}, position={trial['position']}, axis_fine={trial['axis_fine']}, "
                    f"l2={trial.get('l2_reg', DEFAULT_STRUCTURE['l2_reg'])}, "
                    f"static_dropout={trial.get('static_dropout', DEFAULT_STRUCTURE['static_dropout'])}, "
                    f"seq_dropout={trial.get('seq_dropout', DEFAULT_STRUCTURE['seq_dropout'])}, "
                    f"time_steps={trial.get('time_steps', BEST_BASELINE['time_steps'])}, "
                    f"source_onehot={trial.get('source_onehot_enabled', BEST_BASELINE['source_onehot_enabled'])}"
                )
            print(f"Total planned preview: {len(trials)}")
            return

        TUNE_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        trial_index = start_index
        tried = completed_signatures()

        for trial in warmup_trials:
            if len(rows) >= total_count:
                break
            sig = trial_signature(trial)
            if sig in tried:
                continue
            tried.add(sig)
            rows.append(run_trial(trial_index, trial, args))
            trial_index += 1

        no_improve_count = 0
        best_ratio = None
        completed = load_completed_results()
        if not completed.empty and "test_pass_ratio" in completed.columns:
            best_ratio = float(completed["test_pass_ratio"].max())

        while len(rows) < total_count:
            candidates = build_night_adaptive_candidates(args.seed + 37 * len(rows), 8, tried)
            if not candidates:
                print("No new night-search candidates available; stopping.")
                break
            trial = candidates[0]
            tried.add(trial_signature(trial))
            row = run_trial(trial_index, trial, args)
            rows.append(row)
            trial_index += 1

            current_ratio = row.get("test_pass_ratio")
            if row.get("status") == "ok" and current_ratio is not None:
                current_ratio = float(current_ratio)
                if best_ratio is None or current_ratio > best_ratio + 1e-9:
                    best_ratio = current_ratio
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            if no_improve_count >= 25:
                print("Night search saw 25 consecutive non-improving/failed trials; continuing with narrower winner-based candidates.")
                no_improve_count = 0

        ok_rows = [row for row in rows if row.get("status") == "ok"]
        if ok_rows:
            best = max(ok_rows, key=lambda row: row["test_pass_ratio"])
            print("\n===== Best trial in this night run =====")
            print(best)
        print(f"\nAll results appended to: {RESULTS_PATH}")
        return

    if args.feature_search:
        count = args.max_trials if args.max_trials is not None else 32
        trials = build_feature_search_trials(count, args.seed)
        start_index = len(load_completed_results())
    elif args.thermal_history_search:
        count = args.max_trials if args.max_trials is not None else 40
        trials = build_thermal_history_trials(count, args.seed)
        start_index = len(load_completed_results())
    elif args.thermal_best_refine:
        count = args.max_trials if args.max_trials is not None else 48
        trials = build_thermal_best_refine_trials(count, args.seed)
        start_index = len(load_completed_results())
    elif args.continue_from_best_structure:
        count = args.max_trials if args.max_trials is not None else 80
        trials = build_continue_from_best_structure_trials(count, args.seed)
        start_index = len(load_completed_results())
    elif args.overnight_best_refine:
        count = args.max_trials if args.max_trials is not None else 160
        trials = build_overnight_best_refine_trials(count, args.seed)
        start_index = len(load_completed_results())
    elif args.current_dataset_refine:
        count = args.max_trials if args.max_trials is not None else 160
        trials = build_current_dataset_refine_trials(count, args.seed)
        start_index = len(load_completed_results())
    elif args.new_data_refine or args.expanded_data_search:
        count = args.max_trials if args.max_trials is not None else 100
        trials = build_new_data_refine_trials(count, args.seed)
        start_index = len(load_completed_results())
    elif args.near_refine:
        count = args.max_trials if args.max_trials is not None else 12
        trials = build_near_refine_trials(count, args.seed)
        start_index = len(load_completed_results())
    elif args.aux_ensemble:
        count = args.max_trials if args.max_trials is not None else 30
        trials = build_aux_ensemble_trials(count, args.seed, args.ensemble_size)
        start_index = len(load_completed_results())
    elif args.near_threshold:
        count = args.max_trials if args.max_trials is not None else 30
        trials = build_near_threshold_trials(count, args.seed)
        start_index = len(load_completed_results())
    elif args.broad:
        count = args.max_trials if args.max_trials is not None else 50
        trials = build_broad_trials(count, args.seed)
        start_index = len(load_completed_results())
    elif args.adaptive:
        count = args.adaptive_count if args.max_trials is None else args.max_trials
        trials = build_adaptive_trials(count)
        start_index = len(load_completed_results())
    else:
        trials = TRIALS[args.start_index:]
        start_index = args.start_index
    if args.max_trials is not None and not args.adaptive:
        trials = trials[:args.max_trials]

    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print("\n===== Planned trials =====")
        for idx, trial in enumerate(trials, start=start_index):
            print(
                f"{idx:03d}_{trial['name']}: "
                f"euclidean={trial['euclidean']}, "
                f"position={trial['position']}, "
                f"axis_fine={trial['axis_fine']}, "
                f"residual_max={trial['residual_max']}, "
                f"scales={trial['residual_scales']}, "
                f"near={trial.get('near_enabled', BEST_BASELINE['near_enabled'])}, "
                f"rescue={trial.get('near_protect_distance', BEST_BASELINE['near_protect_distance'])}-"
                f"{trial.get('near_rescue_distance', BEST_BASELINE['near_rescue_distance'])}, "
                f"rw={trial.get('near_rescue_weight', BEST_BASELINE['near_rescue_weight'])}, "
                f"lr={trial.get('near_fine_tune_lr', BEST_BASELINE['near_fine_tune_lr'])}, "
                f"dist_aux={trial.get('distance_aux_weight', BEST_BASELINE['distance_aux_weight'])}, "
                f"risk_aux={trial.get('risk_aux_weight', BEST_BASELINE['risk_aux_weight'])}, "
                f"ens={trial.get('ensemble_size', args.ensemble_size)}, "
                f"feature={trial.get('feature_mode', BEST_BASELINE['feature_mode'])}, "
                f"thermal={trial.get('thermal_mode', BEST_BASELINE['thermal_mode'])}, "
                f"time_steps={trial.get('time_steps', BEST_BASELINE['time_steps'])}, "
                f"roll={trial.get('long_rolling_enabled', BEST_BASELINE['long_rolling_enabled'])}:"
                f"{trial.get('long_rolling_windows', BEST_BASELINE['long_rolling_windows'])}, "
                f"progress={trial.get('source_progress_enabled', BEST_BASELINE['source_progress_enabled'])}, "
                f"source_onehot={trial.get('source_onehot_enabled', BEST_BASELINE['source_onehot_enabled'])}, "
                f"static={trial.get('static_units', DEFAULT_STRUCTURE['static_units'])}, "
                f"fused={trial.get('fused_units', DEFAULT_STRUCTURE['fused_units'])}, "
                f"dropout={trial.get('fused_dropout', DEFAULT_STRUCTURE['fused_dropout'])}"
            )
        print(f"Total planned: {len(trials)}")
        return

    rows = []
    for offset, trial in enumerate(trials, start=start_index):
        rows.append(run_trial(offset, trial, args))

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if ok_rows:
        best = max(ok_rows, key=lambda row: row["test_pass_ratio"])
        print("\n===== Best trial in this run =====")
        print(best)
    print(f"\nAll results appended to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
