"""Train or verify the 24-piece geometry-only physical-feature version."""

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

GEOMETRY = Path(__file__).resolve().parent
PROJECT = GEOMETRY.parent
sys.path.insert(0, str(PROJECT))
import train_baseline as common


def check_geometry_split(globals_dict, data):
    import physical_point_features

    common.check_split(globals_dict, data)
    selected = [col for col in data["static_feature_cols"] if col.startswith("physical_")]
    if selected != physical_point_features.GEOMETRY_COLS or len(selected) != 11:
        raise ValueError(f"Expected exactly 11 geometry features, got {selected}")
    recorded = json.loads((GEOMETRY / "split_and_feature_audit.json").read_text(encoding="utf-8"))
    if (data["static_feature_cols"] != recorded["static_feature_columns"] or
            data["state_feature_cols"] != recorded["sequence_feature_columns"]):
        raise ValueError("Model inputs differ from the saved geometry-only experiment")


def worker(args):
    if os.name == "nt":
        tf_root = Path(sys.executable).resolve().parents[1] / "Lib/site-packages/tensorflow"
        handles = [os.add_dll_directory(str(path)) for path in
                   (tf_root, tf_root / "python", Path(r"C:\Windows\System32"))]
        ctypes.WinDLL(r"C:\Windows\System32\msvcp140.dll")
        ctypes.WinDLL(r"C:\Windows\System32\vcruntime140.dll")
        _ = handles
    sys.path.insert(0, str(common.SRC))
    import experiment
    import physical_risk

    physical_risk.PRIOR = PROJECT / "_unused_prior"
    experiment.check_baseline_and_record = check_geometry_split
    physical_risk.training.motion_risk.prepare = physical_risk.prepare
    sys.argv = [str(common.SRC / "main.py"), "--run-mode", "predict_only" if args.mode == "verify" else "train",
                "--data-dir", str(common.DATA), "--output-dir", str(args.output),
                "--no-open-plots", "--no-tensorboard"]
    if args.mode == "verify":
        sys.argv += ["--model-path", str(GEOMETRY / "physical_geometry_nc_model.keras"),
                     "--artifact-path", str(GEOMETRY / "physical_geometry_nc_preprocessing.pkl")]
    else:
        sys.argv += ["--model-file", "physical_geometry_nc_model.keras",
                     "--artifact-file", "physical_geometry_nc_preprocessing.pkl"]
    physical_risk.training.main()


def verify_predictions(output):
    import pandas as pd

    saved = pd.read_csv(GEOMETRY / "test_weighted_predictions.csv")
    fresh = pd.read_csv(output / "test_weighted_predictions.csv")
    matched = saved.merge(fresh, on=common.POINT_KEYS,
                          suffixes=("_saved", "_reloaded"), validate="one_to_one")
    if len(matched) != len(saved) or len(matched) != len(fresh):
        raise ValueError("Test point identities differ from the saved version")
    prediction_gap = max(float((matched[f"pred_{axis}_saved"] - matched[f"pred_{axis}_reloaded"]).abs().max())
                         for axis in "xyz")
    label_gap = max(float((matched[f"true_{axis}_saved"] - matched[f"true_{axis}_reloaded"]).abs().max())
                    for axis in "xyz")
    if prediction_gap > 1e-5 or label_gap > 1e-5:
        raise ValueError(f"Reload changed predictions or labels: {prediction_gap}, {label_gap}")
    correct = int((matched.distance_mm_reloaded < 0.04).sum())
    result = {"test_points": len(matched), "correct": correct,
              "accuracy_pct": 100 * correct / len(matched),
              "max_prediction_difference_mm": prediction_gap,
              "max_label_difference_mm": label_gap}
    if correct != 1816 or len(matched) != 1978:
        raise ValueError(f"Saved geometry-only accuracy changed: {result}")
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("verify", "train"), required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    runtime = common.prepare_manifests()
    parameters = json.loads((GEOMETRY / "training_parameters.json").read_text(encoding="utf-8"))
    if parameters["extra_env"]["TEE_PHYSICAL_FEATURE_SET"] != "geometry":
        raise ValueError("Geometry-only configuration was changed")
    output = GEOMETRY / ("verify" if args.mode == "verify" else "retrain")
    if args.mode == "train" and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Existing retrain output: {output}")
    output.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(parameters["env"])
    env.update(parameters["extra_env"])
    env.update({"TEE_CMM_STL_MANIFEST": str(runtime / "cmm_stl_manifest.json"),
                "TEE_NC_MANIFEST": str(runtime / "nc_manifest.json")})
    command = [sys.executable, str(Path(__file__)), "--worker", "--mode", args.mode,
               "--output", str(output)]
    log_path = output / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode
    if code:
        raise RuntimeError(f"Experiment failed (exit={code}); see {log_path}")
    if args.mode == "verify":
        verify_predictions(output)
    else:
        print(f"Training output: {output}", flush=True)


if __name__ == "__main__":
    main()
