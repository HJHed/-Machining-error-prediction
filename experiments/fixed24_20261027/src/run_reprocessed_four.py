"""Freeze updated inputs and run the four within-workpiece experiments anew."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("17", "1477"), ("17", "1754"), ("14", "1477"), ("14", "1754")]
PARAMS = {
    "TEE_EPOCHS": "300", "TEE_PATIENCE": "60", "TEE_BATCH_SIZE": "128",
    "TEE_TIME_STEPS": "10", "TEE_SPLIT_MODE": "phase_blocks",
    "TEE_RANDOM_SEED": "2031", "TEE_TEMPORAL_BACKBONE": "gru_attention",
    "TEE_SEQUENCE_GROUP_COLS": "source_file_name,mapped_file_index",
    "TEE_USE_VIB_FEATURES": "true", "TEE_SEQ_DENSE_UNITS": "24",
    "TEE_FUSED_DENSE_UNITS": "160,80", "TEE_Z_EXTREME_WEIGHT": "1.0",
    "TEE_SEQ_DROPOUT_RATE": "0.35", "TEE_REDUCE_LR_PATIENCE": "12",
    "TEE_CONSOLE_OUTPUT": "full", "TEE_RISK_TEACHER_EPOCHS": "100",
    "TEE_RISK_ALPHA": "1.5", "TEE_RISK_BETA": "0.5",
    "TEE_USE_MOE_OUTPUT_HEAD": "false", "TEE_USE_RESIDUAL_CORRECTION": "false",
    "TEE_USE_NEAR_THRESHOLD_REWEIGHTING": "false",
    "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
    "TF_NUM_INTRAOP_THREADS": "4", "TF_NUM_INTEROP_THREADS": "2", "OMP_NUM_THREADS": "4",
}
SOURCE = {added: ROOT / "data" / f"hybrid_json_plus_{added}_run2_20261019" for added in ("1477", "1754")}
NC_MANIFEST = {
    "1477": ROOT / "results/add_1477_20261015/nc_manifest_plus_1477.json",
    "1754": ROOT / "results/add_1754_run2_20261019/nc_manifest_plus_1754.json",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(batch):
    frozen = batch / "frozen_data"
    frozen.mkdir()
    records = {}
    for added in ("1477", "1754"):
        src = SOURCE[added]
        dst = frozen / added
        dst.mkdir()
        paths = list(src.glob("*.json"))
        names = {p.name for p in paths}
        assert names == {f"模型{i}.json" for i in range(1, 18)} | {f"{added}.json"}, names
        files = []
        for path in sorted(paths):
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            points = payload["point_to_frame_ai_block_avg"]
            assert points and len({int(p["point_index"]) for p in points}) == len(points), path
            dest = dst / path.name
            shutil.copy2(path, dest)
            assert sha(dest) == sha(path)
            files.append({"name": path.name, "point_count": len(points), "sha256": sha(dest)})
        expected = {"模型17.json": 505, f"{added}.json": 670 if added == "1477" else 685}
        assert {p["name"]: p["point_count"] for p in files if p["name"] in expected} == expected
        manifest = json.loads(NC_MANIFEST[added].read_text(encoding="utf-8-sig"))
        nc_paths = []
        for value in manifest.values():
            nc_paths.extend(value.values() if isinstance(value, dict) else [value])
        assert all(Path(p).is_file() for p in nc_paths), "Missing NC file"
        shutil.copy2(NC_MANIFEST[added], batch / f"nc_manifest_{added}.json")
        records[added] = {"source_dir": str(src), "frozen_dir": str(dst),
                          "files": files, "nc_manifest_sha256": sha(NC_MANIFEST[added]),
                          "nc_files": [{"path": p, "sha256": sha(p)} for p in sorted(set(nc_paths))]}
    (batch / "input_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return frozen


def command(batch, frozen, cohort, added):
    name = f"{cohort}_plus_{added}"
    out = batch / name
    out.mkdir()
    env = os.environ.copy()
    env.update(PARAMS)
    env.update({"TEE_UPDATED_COHORT": cohort, "TEE_ADDED_WORKPIECE": added,
                "TEE_EXCLUDE_SOURCE_NUMBERS": "13,14,17" if cohort == "14" else "",
                "TEE_NC_MANIFEST": str(batch / f"nc_manifest_{added}.json")})
    args = [str(ROOT / ".venv/Scripts/python.exe"), str(ROOT / "lstm_nc_risk/run_reprocessed_experiment.py"),
            "--run-mode", "train", "--data-dir", str(frozen / added), "--output-dir", str(out),
            "--model-file", "nc_physical_risk_model.keras", "--artifact-file", "motion_weighted_preprocessing.pkl",
            "--no-open-plots", "--no-tensorboard"]
    return name, out, env, args


def main():
    batch = ROOT / "results" / ("reprocessed_four_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    batch.mkdir()
    frozen = preflight(batch)
    (batch / "run_parameters.json").write_text(json.dumps({"params": PARAMS, "runs": RUNS,
                                                             "script_sha256": sha(__file__),
                                                             "runner_sha256": sha(ROOT / "lstm_nc_risk/run_reprocessed_experiment.py")},
                                                            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"BATCH={batch}", flush=True)
    pending = list(RUNS)
    running = {}
    status = {f"{a}_plus_{b}": "pending" for a, b in RUNS}
    while pending or running:
        while pending and len(running) < 2:
            cohort, added = pending.pop(0)
            name, out, env, args = command(batch, frozen, cohort, added)
            log = (out / "training.log").open("w", encoding="utf-8")
            proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            running[name] = (proc, log, out)
            status[name] = {"state": "running", "pid": proc.pid, "start": datetime.now().isoformat()}
            print(f"START {name} PID={proc.pid}", flush=True)
        for name, (proc, log, out) in list(running.items()):
            code = proc.poll()
            if code is not None:
                log.close()
                status[name] = {"state": "complete" if code == 0 else "failed", "exit_code": code,
                                "end": datetime.now().isoformat()}
                print(f"END {name} exit={code}", flush=True)
                del running[name]
        (batch / "batch_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if pending or running:
            time.sleep(15)
    if any(value["state"] != "complete" for value in status.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
