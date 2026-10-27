"""Package the frozen 91.00% experiment into this isolated project directory."""

import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
BATCH = REPO / "results/image_24_data_only_20261027"
SOURCE_CODE = REPO / "lstm_nc_risk"
CODE_FILES = (
    "main.py", "nc_features.py", "region_nc_features.py", "physical_point_features.py",
    "audit_physical_feature_sources.py", "motion_risk.py", "physical_risk.py",
    "experiment.py", "run_reprocessed_four.py",
)
BASELINE_FILES = (
    "physical_geometry_nc_model.keras", "physical_geometry_nc_preprocessing.pkl",
    "training_history.csv", "test_weighted_predictions.csv", "val_weighted_predictions.csv",
    "train_point_identities.csv", "val_point_identities.csv", "test_point_identities.csv",
    "data_only_split_audit.json", "physical_risk_preprocessing.json",
    "tool_error_test_source_failure_summary_v10.csv",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_exact(src, dst):
    if dst.exists():
        if sha256(src) != sha256(dst):
            raise ValueError(f"Existing snapshot differs: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha256(src) != sha256(dst):
        raise ValueError(f"Copy checksum differs: {dst}")


def main():
    manifest = json.loads((BATCH / "cmm_stl_manifest.json").read_text(encoding="utf-8"))
    nc_map = json.loads((BATCH / "nc_manifest.json").read_text(encoding="utf-8"))
    original = json.loads((BATCH / "input_manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != 24 or set(manifest) != set(nc_map):
        raise ValueError("Expected 24 matched geometry/NC inputs")

    for name in CODE_FILES:
        copy_exact(SOURCE_CODE / name, PROJECT / "src" / name)
    for path in sorted((BATCH / "frozen_data").glob("*.json")):
        copy_exact(path, PROJECT / "data/fusion" / path.name)
        if sha256(path) != original["data_sha256"][path.name]:
            raise ValueError(f"Input hash differs from trained experiment: {path.name}")
    for path in sorted((BATCH / "1673_measurement").glob("xy1673*")):
        if path.is_file():
            copy_exact(path, PROJECT / "data/measurement/1673" / path.name)
    for name in BASELINE_FILES:
        copy_exact(BATCH / "24_workpieces" / name, PROJECT / "baseline" / name)
    copy_exact(BATCH / "accuracy_summary.json", PROJECT / "baseline/accuracy_summary.json")
    copy_exact(BATCH / "per_workpiece_accuracy.csv", PROJECT / "baseline/per_workpiece_accuracy.csv")
    copy_exact(BATCH / "reload_verification/verification.json", PROJECT / "baseline/reload_verification.json")

    entries = {}
    nc_copies = {}
    for source, record in manifest.items():
        code = record["code"]
        nominal_dst = PROJECT / "data/nominal" / f"{code}.txt"
        stl_dst = PROJECT / "data/stl" / f"{code}.stl"
        copy_exact(Path(record["nominal_points"]), nominal_dst)
        copy_exact(Path(record["stl"]), stl_dst)
        nc_value = nc_map[source]
        if isinstance(nc_value, dict):
            paths = {}
            for index, src_name in nc_value.items():
                src = Path(src_name)
                dst = PROJECT / "data/nc" / f"{code}_{index}{src.suffix}"
                copy_exact(src, dst)
                paths[str(index)] = dst.relative_to(PROJECT).as_posix()
                nc_copies[dst.relative_to(PROJECT).as_posix()] = sha256(dst)
            nc_relative = paths
        else:
            src = Path(nc_value)
            dst = PROJECT / "data/nc" / f"{code}{src.suffix}"
            copy_exact(src, dst)
            nc_relative = dst.relative_to(PROJECT).as_posix()
            nc_copies[dst.relative_to(PROJECT).as_posix()] = sha256(dst)
        entries[source] = {
            "code": code,
            "fusion": f"data/fusion/{source}",
            "nominal": nominal_dst.relative_to(PROJECT).as_posix(),
            "stl": stl_dst.relative_to(PROJECT).as_posix(),
            "nc": nc_relative,
            "point_count": int(record["fusion_count"]),
            "sha256": {
                "fusion": sha256(PROJECT / "data/fusion" / source),
                "nominal": sha256(nominal_dst),
                "stl": sha256(stl_dst),
            },
        }

    known = {item["code"] for item in entries.values()}
    if len(known) != 24 or known & {"1739", "1892"}:
        raise ValueError("Incorrect cohort identity")
    pinned = {name: version for name in ("numpy", "pandas", "scipy", "scikit-learn", "tensorflow", "matplotlib")
              if (version := importlib.metadata.version(name))}
    (PROJECT / "requirements.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in pinned.items()) + "\n", encoding="utf-8")
    parameters = {
        "env": original["parameters"],
        "extra_env": {"TEE_USE_REGION_NC_BRANCH": "false", "TEE_USE_PHYSICAL_POINT_FEATURES": "true",
                      "TEE_PHYSICAL_FEATURE_SET": "both", "TEE_EXCLUDE_SOURCE_NUMBERS": "",
                      "TEE_EXPORT_PHYSICAL_SCREEN": "false"},
        "test_threshold_mm": 0.04,
    }
    (PROJECT / "training_parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {
        "source_repo_commit": "642243afc7a7cbea3ab2e6120afc320869568a65",
        "source_experiment": str(BATCH),
        "source_experiment_input_manifest_sha256": sha256(BATCH / "input_manifest.json"),
        "workpiece_count": 24,
        "excluded_from_image": ["1739", "1892"],
        "entries": entries,
        "nc_sha256": nc_copies,
        "code_sha256": {name: sha256(PROJECT / "src" / name) for name in CODE_FILES},
        "measurement_1673_sha256": {path.name: sha256(path) for path in
                                    sorted((PROJECT / "data/measurement/1673").iterdir()) if path.is_file()},
        "baseline_accuracy": json.loads((PROJECT / "baseline/accuracy_summary.json").read_text(encoding="utf-8")),
        "baseline_model_sha256": sha256(PROJECT / "baseline/physical_geometry_nc_model.keras"),
        "baseline_preprocessing_sha256": sha256(PROJECT / "baseline/physical_geometry_nc_preprocessing.pkl"),
    }
    (PROJECT / "source_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Packaged {len(entries)} workpieces; {len(nc_copies)} NC files in {PROJECT}")


if __name__ == "__main__":
    main()
