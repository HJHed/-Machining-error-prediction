"""Reproduce or verify the frozen 24-piece baseline without changing parent code."""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SRC = PROJECT / "src"
BASELINE = PROJECT / "baseline"
DATA = PROJECT / "data/fusion"
POINT_KEYS = ["source_file_name", "mapped_file_index", "mapped_nc_line",
              "mapped_frame_num", "mapped_frame_id", "point_index"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_manifests():
    provenance = json.loads((PROJECT / "source_manifest.json").read_text(encoding="utf-8"))
    geometry, nc = {}, {}
    for source, item in provenance["entries"].items():
        files = [PROJECT / item[key] for key in ("fusion", "nominal", "stl")]
        if any(not path.is_file() or sha256(path) != item["sha256"][key]
               for path, key in zip(files, ("fusion", "nominal", "stl"))):
            raise ValueError(f"Missing or changed frozen input: {source}")
        geometry[source] = {"code": item["code"], "nominal_points": str(files[1]), "stl": str(files[2])}
        value = item["nc"]
        nc[source] = ({index: str(PROJECT / path) for index, path in value.items()}
                      if isinstance(value, dict) else str(PROJECT / value))
        for relative in (value.values() if isinstance(value, dict) else [value]):
            full = PROJECT / relative
            if not full.is_file() or sha256(full) != provenance["nc_sha256"][relative]:
                raise ValueError(f"Missing or changed NC file: {relative}")
    if len(geometry) != 24 or set(geometry) != {path.name for path in DATA.glob("*.json")}:
        raise ValueError("Frozen workpiece cohort differs")
    runtime = PROJECT / "runtime"
    runtime.mkdir(exist_ok=True)
    (runtime / "cmm_stl_manifest.json").write_text(json.dumps(geometry, ensure_ascii=False, indent=2), encoding="utf-8")
    (runtime / "nc_manifest.json").write_text(json.dumps(nc, ensure_ascii=False, indent=2), encoding="utf-8")
    return runtime


def check_split(globals_dict, data):
    import pandas as pd
    from experiment import KEYS

    expected = {path.name for path in DATA.glob("*.json")}
    if {Path(path).name for path in data["json_paths"]} != expected:
        raise ValueError("Workpiece list differs from frozen input")
    counts = {}
    for name in ("train", "val", "test"):
        current = data[name + "_meta"]
        reference = pd.read_csv(BASELINE / f"{name}_point_identities.csv")
        if current.duplicated(KEYS).any() or reference.duplicated(KEYS).any():
            raise ValueError(f"Duplicate point identity in {name}")
        if len(current) != len(reference) or set(map(tuple, current[KEYS].to_numpy())) != set(map(tuple, reference[KEYS].to_numpy())):
            raise ValueError(f"{name} points differ from saved baseline")
        if set(current.source_file_name) != expected:
            raise ValueError(f"Workpiece missing in {name}")
        counts[name] = len(current)
    if set(globals_dict["TARGET_COLS"]) & set(data["static_feature_cols"] + data["state_feature_cols"]):
        raise ValueError("Target leaked into model inputs")
    output = Path(globals_dict["OUTPUT_DIR"])
    (output / "split_and_feature_audit.json").write_text(json.dumps({
        "split_points": counts,
        "static_feature_columns": data["static_feature_cols"],
        "sequence_feature_columns": data["state_feature_cols"],
        "seed": globals_dict["RANDOM_SEED"],
        "split_mode": globals_dict["SPLIT_MODE"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Frozen split verified: {counts}", flush=True)


def run_worker(args):
    if os.name == "nt":
        tensorflow_root = Path(sys.executable).resolve().parents[1] / "Lib/site-packages/tensorflow"
        dll_handles = [os.add_dll_directory(str(path)) for path in
                       (tensorflow_root, tensorflow_root / "python", Path(r"C:\Windows\System32"))]
        ctypes.WinDLL(r"C:\Windows\System32\msvcp140.dll")
        ctypes.WinDLL(r"C:\Windows\System32\vcruntime140.dll")
        _ = dll_handles
    sys.path.insert(0, str(SRC))
    import experiment
    import physical_risk

    physical_risk.PRIOR = PROJECT / "_unused_prior"
    experiment.check_baseline_and_record = check_split
    physical_risk.training.motion_risk.prepare = physical_risk.prepare
    sys.argv = [str(SRC / "main.py"), "--run-mode", "predict_only" if args.mode == "verify" else "train",
                "--data-dir", str(DATA), "--output-dir", str(args.output),
                "--no-open-plots", "--no-tensorboard"]
    if args.mode == "verify":
        sys.argv += ["--model-path", str(BASELINE / "physical_geometry_nc_model.keras"),
                     "--artifact-path", str(BASELINE / "physical_geometry_nc_preprocessing.pkl")]
    else:
        sys.argv += ["--model-file", "physical_geometry_nc_model.keras",
                     "--artifact-file", "physical_geometry_nc_preprocessing.pkl"]
    physical_risk.training.main()


def check_predictions(output):
    import pandas as pd

    before = pd.read_csv(BASELINE / "test_weighted_predictions.csv")
    after = pd.read_csv(output / "test_weighted_predictions.csv")
    paired = before.merge(after, on=POINT_KEYS, suffixes=("_saved", "_reloaded"), validate="one_to_one")
    if len(paired) != len(before) or len(paired) != len(after):
        raise ValueError("Test identities differ")
    difference = max((paired[f"pred_{axis}_saved"] - paired[f"pred_{axis}_reloaded"]).abs().max()
                     for axis in "xyz")
    if difference > 1e-5:
        raise ValueError(f"Reload differs by {difference} mm")
    print(json.dumps({"test_points": len(paired), "max_prediction_difference_mm": float(difference)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("verify", "train"), required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return
    runtime = prepare_manifests()
    params = json.loads((PROJECT / "training_parameters.json").read_text(encoding="utf-8"))
    output = PROJECT / ("baseline/reload_from_snapshot" if args.mode == "verify" else "runs/baseline_retrain")
    if args.mode == "train" and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(params["env"])
    env.update(params["extra_env"])
    env.update({"TEE_PHYSICAL_FEATURE_SET": "both",
                "TEE_CMM_STL_MANIFEST": str(runtime / "cmm_stl_manifest.json"),
                "TEE_NC_MANIFEST": str(runtime / "nc_manifest.json")})
    command = [sys.executable, str(Path(__file__)), "--worker", "--mode", args.mode,
               "--output", str(output)]
    log_path = output / "training.log"
    with log_path.open("w", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode
    if code:
        raise RuntimeError(f"Run failed (exit={code}): {log_path}")
    if args.mode == "verify":
        check_predictions(output)
    else:
        print(f"Training finished; results: {output}")


if __name__ == "__main__":
    main()
