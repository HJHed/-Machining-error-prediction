"""Pair the geometry-only predictions with the frozen 91.00% baseline."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GEOMETRY = Path(__file__).resolve().parent
PROJECT = GEOMETRY.parent
KEYS = ["source_file_name", "mapped_file_index", "mapped_nc_line",
        "mapped_frame_num", "mapped_frame_id", "point_index"]


def main():
    baseline = pd.read_csv(PROJECT / "baseline/test_weighted_predictions.csv")
    geometry = pd.read_csv(GEOMETRY / "test_weighted_predictions.csv")
    paired = baseline.merge(geometry, on=KEYS, suffixes=("_baseline", "_geometry11"), validate="one_to_one")
    if len(paired) != len(baseline) or len(paired) != len(geometry):
        raise ValueError("Test point identities differ")
    label_gap = max(float((paired[f"true_{axis}_baseline"] - paired[f"true_{axis}_geometry11"]).abs().max())
                    for axis in "xyz")
    if label_gap > 1e-5:
        raise ValueError(f"Test labels differ by {label_gap} mm")
    paired["baseline_correct"] = paired.distance_mm_baseline < 0.04
    paired["geometry11_correct"] = paired.distance_mm_geometry11 < 0.04
    paired["transition"] = np.select(
        [~paired.baseline_correct & paired.geometry11_correct,
         paired.baseline_correct & ~paired.geometry11_correct],
        ["fixed", "regressed"], default="unchanged")
    pieces = paired.groupby("source_file_name", sort=True).agg(
        test_points=("point_index", "size"),
        baseline_correct=("baseline_correct", "sum"),
        geometry11_correct=("geometry11_correct", "sum"),
        fixed=("transition", lambda values: int((values == "fixed").sum())),
        regressed=("transition", lambda values: int((values == "regressed").sum())),
    ).reset_index()
    codes = {source: item["code"] for source, item in
             json.loads((PROJECT / "source_manifest.json").read_text(encoding="utf-8"))["entries"].items()}
    pieces["workpiece_code"] = pieces.source_file_name.map(codes)
    if pieces.workpiece_code.isna().any():
        raise ValueError("Workpiece code missing from frozen manifest")
    pieces = pieces.sort_values("workpiece_code").reset_index(drop=True)
    pieces["baseline_accuracy_pct"] = 100 * pieces.baseline_correct / pieces.test_points
    pieces["geometry11_accuracy_pct"] = 100 * pieces.geometry11_correct / pieces.test_points
    pieces["change_percentage_points"] = pieces.geometry11_accuracy_pct - pieces.baseline_accuracy_pct
    pieces.to_csv(GEOMETRY / "per_workpiece_comparison.csv", index=False, encoding="utf-8-sig")
    paired[KEYS + ["baseline_correct", "geometry11_correct", "transition"]].to_csv(
        GEOMETRY / "point_transitions.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(2031)
    samples = rng.integers(0, len(pieces), size=(10000, len(pieces)))
    gains = (pieces.geometry11_correct - pieces.baseline_correct).to_numpy()
    totals = pieces.test_points.to_numpy()
    boot = 100 * gains[samples].sum(axis=1) / totals[samples].sum(axis=1)
    summary = {
        "test_points": len(paired),
        "baseline_correct": int(paired.baseline_correct.sum()),
        "geometry11_correct": int(paired.geometry11_correct.sum()),
        "baseline_accuracy_pct": float(100 * paired.baseline_correct.mean()),
        "geometry11_accuracy_pct": float(100 * paired.geometry11_correct.mean()),
        "change_percentage_points": float(100 * (paired.geometry11_correct.mean() - paired.baseline_correct.mean())),
        "baseline_failures_fixed": int((paired.transition == "fixed").sum()),
        "baseline_successes_regressed": int((paired.transition == "regressed").sum()),
        "pieces_improved": int((pieces.change_percentage_points > 0).sum()),
        "pieces_worsened": int((pieces.change_percentage_points < 0).sum()),
        "pieces_tied": int((pieces.change_percentage_points == 0).sum()),
        "workpiece_cluster_bootstrap_95pct_change_pp": [float(x) for x in np.quantile(boot, [.025, .975])],
        "max_label_difference_mm": label_gap,
        "criterion": "3D residual norm < 0.04 mm",
    }
    (GEOMETRY / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    x = np.arange(len(pieces))
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.bar(x - .19, pieces.baseline_accuracy_pct, width=.38, label="All 30 new features (91.00%)", color="#7C8A99")
    ax.bar(x + .19, pieces.geometry11_accuracy_pct, width=.38, label="11 geometry features (91.81%)", color="#167A7A")
    ax.set_xticks(x, pieces.workpiece_code, rotation=45, ha="right")
    ax.set_ylabel("Test points within 0.04 mm (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(GEOMETRY / "per_workpiece_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
