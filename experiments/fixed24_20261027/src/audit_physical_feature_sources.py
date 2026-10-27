"""Audit pre-measurement design points against measured CMM points and STL."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from run_reprocessed_four import ROOT


BASE = ROOT / "results/reprocessed_both_20261020_195743"
DATA = ROOT / "results/add_1540_20261022_173557/frozen_data"
OUT = ROOT / "results/physical_features_20261022"
BIN_STL = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mesh_path(code):
    direct = Path("E:/realFab") / code / f"{code}.stl"
    if direct.is_file():
        return direct
    paths = list((ROOT.parent / "可视化/STL").glob(f"*-stl设计模型/turbine_blade_0310_{code}.stl"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one STL for {code}, got {paths}")
    return paths[0]


def mesh_vertices(path):
    raw = Path(path).read_bytes()
    if len(raw) >= 84:
        n = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
        if 84 + n * 50 == len(raw):
            return np.frombuffer(raw, dtype=BIN_STL, count=n, offset=84)["vertices"].reshape(-1, 3)
    vertices = [[float(value) for value in line.split()[1:4]]
                for line in raw.decode("utf-8", errors="ignore").splitlines()
                if line.lstrip().startswith("vertex ")]
    if not vertices or len(vertices) % 3:
        raise ValueError(f"Invalid STL: {path}")
    return np.asarray(vertices, np.float32)


def design_path(code):
    overrides = {
        "0098": "Machining-error-prediction-repro-20261014/data/design_nominal/xy0098_d.txt",
        "0002": "workflow_output/0002_zge0_20261020/measurement/xy0002_d.txt",
        "1477": "workflow_output/1477_run2_20261019/measurement/xy1477_d.txt",
        "1754": "workflow_output/1754_run2_20261019/measurement/xy1754_d.txt",
        "1540": "workflow_output/1540_zge0_20261022/measurement/xy1540_d.txt",
    }
    path = ROOT.parent / overrides.get(code, f"excel处理/xy{code}_d.txt")
    if not path.is_file():
        raise FileNotFoundError(f"No pre-measurement design points for {code}: {path}")
    return path


def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((BASE / "cmm_manifest.json").read_text(encoding="utf-8"))
    manifest["1540.json"] = {
        "code": "1540",
        "frozen_points": str(ROOT.parent / "workflow_output/1540_zge0_20261022/measurement/xy1540.txt"),
        "frozen_errors": str(ROOT.parent / "workflow_output/1540_zge0_20261022/measurement/xy1540_gosa.txt"),
    }
    if set(manifest) != {p.name for p in DATA.glob("*.json")}:
        raise ValueError("CMM and JSON workpieces differ")
    audited = {}
    for path in sorted(DATA.glob("*.json")):
        item = dict(manifest[path.name])
        coords_path = Path(item["frozen_points"])
        errors_path = Path(item["frozen_errors"])
        measured = np.atleast_2d(np.loadtxt(coords_path, encoding="utf-8-sig"))
        source_design_path = design_path(item["code"])
        nominal = np.atleast_2d(np.loadtxt(source_design_path, encoding="utf-8-sig"))
        errors = np.atleast_2d(np.loadtxt(errors_path, encoding="utf-8-sig"))
        if nominal.shape[1] != 3 or measured.shape != errors.shape:
            raise ValueError(f"CMM coordinate/label shape mismatch: {path.name}")
        if len(nominal) != len(measured):
            full_measured_path = ROOT.parent / "excel处理" / f"xy{item['code']}.txt"
            full_measured = np.atleast_2d(np.loadtxt(full_measured_path, encoding="utf-8-sig"))
            if full_measured.shape != nominal.shape:
                raise ValueError(f"Original measured/design shape mismatch: {path.name}")
            keep = full_measured[:, 2] >= 0
            if int(keep.sum()) != len(measured) or not np.allclose(full_measured[keep], measured, atol=1e-6, rtol=0):
                raise ValueError(f"Frozen CMM does not match the original positive-Z subset: {path.name}")
            nominal = nominal[keep]
        nominal_dir = OUT / "design_nominal"
        nominal_dir.mkdir(exist_ok=True)
        nominal_path = nominal_dir / f"{item['code']}.txt"
        np.savetxt(nominal_path, nominal, fmt="%.6f")
        records = json.loads(path.read_text(encoding="utf-8-sig"))["point_to_frame_ai_block_avg"]
        indices = np.asarray([row["point_index"] for row in records], int)
        labels = np.asarray([row["point_error_xyz"] for row in records], float)
        if indices.min() < 0 or indices.max() >= len(nominal) or len(np.unique(indices)) != len(indices):
            raise ValueError(f"CMM point index mismatch: {path.name}")
        max_label_delta = float(np.max(np.abs(labels - errors[indices])))
        if max_label_delta > 1e-6:
            raise ValueError(f"CMM error labels differ for {path.name}: {max_label_delta}")
        max_coordinate_residual = float(np.max(np.abs(measured[indices] - nominal[indices] - errors[indices])))
        stl = mesh_path(item["code"])
        vertex_gap = cKDTree(mesh_vertices(stl)).query(nominal[indices])[0]
        item.update({"frozen_points": str(coords_path.resolve()),
                     "nominal_points": str(nominal_path.resolve()),
                     "source_design_points": str(source_design_path.resolve()),
                     "frozen_errors": str(errors_path.resolve()),
                     "stl": str(stl.resolve()),
                     "points_sha256": sha(coords_path), "nominal_sha256": sha(nominal_path),
                     "source_design_sha256": sha(source_design_path),
                     "errors_sha256": sha(errors_path),
                     "stl_sha256": sha(stl), "json_sha256": sha(path),
                     "measurement_count": len(nominal), "fusion_count": len(indices),
                     "max_label_delta_mm": max_label_delta,
                     "max_measured_minus_nominal_minus_error_mm": max_coordinate_residual,
                     "nominal_to_stl_vertex_median_mm": float(np.median(vertex_gap)),
                     "nominal_to_stl_vertex_p95_mm": float(np.quantile(vertex_gap, .95)),
                     "nominal_z_min_mm": float(nominal[indices, 2].min())})
        if item["code"] == "0098":
            workbook = ROOT.parent / "可视化/cmm测点excel/xy0098.xlsx"
            item["nominal_source_workbook"] = str(workbook.resolve())
            item["nominal_source_workbook_sha256"] = sha(workbook)
        if item["nominal_to_stl_vertex_median_mm"] > 3.0:
            raise ValueError(f"STL/CMM registration needs review: {path.name}, {item['nominal_to_stl_vertex_median_mm']:.2f} mm")
        audited[path.name] = item
        print(f"{path.name}: {len(indices)} points, design-vs-STL median={item['nominal_to_stl_vertex_median_mm']:.3f} mm, measured-design-error max={max_coordinate_residual:.4f} mm", flush=True)
    (OUT / "cmm_stl_manifest.json").write_text(json.dumps(audited, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT / "cmm_stl_manifest.json", flush=True)


if __name__ == "__main__":
    main()
