"""Target-free nominal CMM/CAD geometry and compact NC-motion descriptors."""

import bisect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import nc_features
from audit_physical_feature_sources import mesh_vertices


GEOMETRY_COLS = [
    "physical_cmm_x_fraction", "physical_cmm_y_fraction", "physical_cmm_z_fraction",
    "physical_cmm_radial_fraction", "physical_cmm_tip_distance_fraction",
    "physical_cad_gap_log_mm", "physical_cad_surface_valid",
    "physical_cad_normal_x", "physical_cad_normal_y", "physical_cad_normal_z",
    "physical_cad_normal_variation",
]
MOTION_COLS = [
    "physical_nc_future5_max_turn", "physical_nc_future5_plunge_log",
    "physical_nc_future5_max_segment_log", "physical_nc_future5_feed_change",
    "physical_nc_future5_rotary_change_log", "physical_nc_future5_rapid",
    "physical_nc_future15_max_turn", "physical_nc_future15_plunge_log",
    "physical_nc_future15_max_segment_log", "physical_nc_future15_feed_change",
    "physical_nc_future15_rotary_change_log", "physical_nc_future15_rapid",
    "physical_nc_previous5_max_turn", "physical_nc_previous5_plunge_log",
    "physical_nc_previous5_max_segment_log", "physical_nc_future5_command_count_log",
    "physical_acc_normal_log", "physical_acc_tangent_log",
    "physical_nc_tangent_normal_abs",
]


def closest_segments(point, start, end):
    edge = end - start
    t = np.sum((point - start) * edge, axis=-1) / np.maximum(np.sum(edge * edge, axis=-1), 1e-12)
    return start + np.clip(t, 0, 1)[..., None] * edge


def surface_features(coords, stl):
    faces = mesh_vertices(stl).reshape(-1, 3, 3).astype(np.float64)
    vertices = faces.reshape(-1, 3)
    centroids = faces.mean(axis=1)
    k_centroid = min(32, len(faces))
    k_vertex = min(12, len(vertices))
    near_centroid = np.atleast_2d(cKDTree(centroids).query(coords, k=k_centroid)[1])
    near_vertex = np.atleast_2d(cKDTree(vertices).query(coords, k=k_vertex)[1]) // 3
    candidate = np.concatenate([near_centroid, near_vertex], axis=1)
    triangles = faces[candidate]
    a, b, c = (triangles[:, :, index] for index in range(3))
    e0, e1 = b - a, c - a
    normal = np.cross(e0, e1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    point = coords[:, None, :]
    projected = point - np.sum((point - a) * normal, axis=-1)[..., None] * normal
    v2 = projected - a
    d00 = np.sum(e0 * e0, axis=-1)
    d01 = np.sum(e0 * e1, axis=-1)
    d11 = np.sum(e1 * e1, axis=-1)
    d20 = np.sum(v2 * e0, axis=-1)
    d21 = np.sum(v2 * e1, axis=-1)
    denominator = d00 * d11 - d01 * d01
    beta = (d11 * d20 - d01 * d21) / np.where(np.abs(denominator) > 1e-12, denominator, 1)
    gamma = (d00 * d21 - d01 * d20) / np.where(np.abs(denominator) > 1e-12, denominator, 1)
    inside = (beta >= 0) & (gamma >= 0) & (beta + gamma <= 1) & (np.abs(denominator) > 1e-12)
    segment_points = np.stack([closest_segments(point, a, b), closest_segments(point, b, c),
                               closest_segments(point, c, a)], axis=2)
    segment_dist_sq = np.sum((segment_points - point[:, :, None, :]) ** 2, axis=-1)
    nearest_segment = np.take_along_axis(segment_points, np.argmin(segment_dist_sq, axis=2)[..., None, None], axis=2)[:, :, 0]
    nearest = np.where(inside[..., None], projected, nearest_segment)
    gap = np.linalg.norm(nearest - point, axis=-1)
    best = np.argmin(gap, axis=1)
    rows = np.arange(len(coords))
    best_gap = gap[rows, best]
    best_normal = normal[rows, best]
    local_weight = np.exp(-np.maximum(gap - best_gap[:, None], 0) ** 2 / 2.0)
    alignment = np.abs(np.sum(normal * best_normal[:, None, :], axis=-1))
    variation = np.sum(local_weight * (1 - alignment), axis=1) / np.maximum(local_weight.sum(axis=1), 1e-9)
    return best_gap, best_normal, variation


def geometry_features(points, manifest):
    out = np.zeros((len(points), len(GEOMETRY_COLS)), np.float32)
    audit = {}
    for source, indexes in points.groupby("source_file_name", sort=False).indices.items():
        if source not in manifest:
            raise ValueError(f"No nominal CMM/STL mapping: {source}")
        mapping = manifest[source]
        nominal = np.atleast_2d(np.loadtxt(mapping["nominal_points"], encoding="utf-8-sig"))
        point_index = points.iloc[indexes].point_index.to_numpy(int)
        if point_index.min() < 0 or point_index.max() >= len(nominal):
            raise ValueError(f"Nominal CMM index outside file: {source}")
        coords = nominal[point_index].astype(np.float64)
        low, high = coords.min(axis=0), coords.max(axis=0)
        scaled = np.clip((coords - low) / np.maximum(high - low, 1e-6), 0, 1)
        radial = np.linalg.norm(coords[:, :2] - np.median(coords[:, :2], axis=0), axis=1)
        radial /= max(float(np.linalg.norm(high[:2] - low[:2])), 1e-6)
        gap, normal, variation = surface_features(coords, mapping["stl"])
        valid = gap < 2.0
        normal[~valid] = 0
        variation[~valid] = 0
        out[indexes] = np.column_stack([
            scaled, radial, 1 - scaled[:, 2], np.log1p(np.minimum(gap, 20)), valid.astype(float),
            normal, variation,
        ]).astype(np.float32)
        audit[source] = {"points": len(indexes), "surface_valid": int(valid.sum()),
                         "surface_gap_median_mm": float(np.median(gap)),
                         "surface_gap_p95_mm": float(np.quantile(gap, .95)),
                         "nominal_file": mapping["nominal_points"], "stl_file": mapping["stl"]}
    if not np.isfinite(out).all():
        raise ValueError("Non-finite CMM/CAD features")
    return out, audit


def command_arrays(moves):
    xyz = np.stack([m[1] for m in moves])
    ac = np.stack([m[2] for m in moves])
    feed = np.asarray([m[4] for m in moves], float)
    mode = np.asarray([m[3] for m in moves], int)
    delta = np.diff(xyz, axis=0, prepend=xyz[:1])
    length = np.linalg.norm(delta, axis=1)
    tangent = delta / np.maximum(length[:, None], 1e-9)
    previous = np.roll(tangent, 1, axis=0)
    previous[0] = 0
    turn = np.arccos(np.clip(np.sum(tangent * previous, axis=1), -1, 1)) / np.pi
    turn[(length < 1e-9) | (np.roll(length, 1) < 1e-9)] = 0
    feed_change = np.abs(np.diff(feed, prepend=feed[0])) / np.maximum(np.abs(np.roll(feed, 1)), 1)
    rotary_change = np.linalg.norm(np.diff(ac, axis=0, prepend=ac[:1]), axis=1)
    return {"line": np.asarray([m[0] for m in moves], int), "length": length,
            "plunge": np.maximum(-delta[:, 2], 0), "turn": turn,
            "feed_change": np.minimum(feed_change, 5), "rotary_change": rotary_change,
            "rapid": mode == 0, "tangent": tangent}


def window_stats(commands, center, direction, distance_limit):
    seen = 0.0
    collected = []
    index = center + direction
    while 0 <= index < len(commands["line"]) and seen < distance_limit and len(collected) < 80:
        collected.append(index)
        seen += max(float(commands["length"][index]), 1e-6)
        index += direction
    if not collected:
        return np.zeros(7, np.float32)
    idx = np.asarray(collected, int)
    return np.asarray([
        float(commands["turn"][idx].max()),
        np.log1p(float(commands["plunge"][idx].sum())),
        np.log1p(float(commands["length"][idx].max())),
        float(commands["feed_change"][idx].max()),
        np.log1p(float(commands["rotary_change"][idx].max())),
        float(commands["rapid"][idx].any()),
        np.log1p(len(idx)),
    ], np.float32)


def motion_features(points, geometry):
    out = np.zeros((len(points), len(MOTION_COLS)), np.float32)
    manifest = nc_features.read_manifest()
    cache = {}
    required = {"source_file_name", "original_mapped_file_index", "mapped_nc_line"}
    if not required.issubset(points):
        raise ValueError(f"Missing NC keys: {required - set(points)}")
    for (source, original), indexes in points.groupby(["source_file_name", "original_mapped_file_index"], sort=False).indices.items():
        mapping = manifest[source]
        path = mapping[str(int(original))] if isinstance(mapping, dict) else mapping
        if path not in cache:
            moves, _, _ = nc_features.parse_program(path)
            cache[path] = command_arrays(moves)
        commands = cache[path]
        for row, line in zip(indexes, points.iloc[indexes].mapped_nc_line):
            center = bisect.bisect_right(commands["line"], int(line)) - 1
            if center < 0:
                continue
            forward5 = window_stats(commands, center, 1, 5.0)
            forward15 = window_stats(commands, center, 1, 15.0)
            backward5 = window_stats(commands, center, -1, 5.0)
            out[row, :6] = forward5[:6]
            out[row, 6:12] = forward15[:6]
            out[row, 12:15] = backward5[:3]
            out[row, 15] = forward5[6]
            normal = geometry[row, 7:10]
            acceleration = points.iloc[row][[f"motion_acceleration_a_{a}" for a in "XYZ"]].to_numpy(np.float64)
            acceleration = np.nan_to_num(acceleration)
            normal_component = abs(float(np.dot(acceleration, normal)))
            total = float(np.linalg.norm(acceleration))
            out[row, 16] = np.log1p(normal_component)
            out[row, 17] = np.log1p(np.sqrt(max(total * total - normal_component * normal_component, 0)))
            out[row, 18] = abs(float(np.dot(commands["tangent"][center], normal)))
    if not np.isfinite(out).all():
        raise ValueError("Non-finite physical NC features")
    return out, len(cache)


def add_features(points):
    """All added inputs derive from nominal geometry, NC, and recorded motion only."""
    manifest_path = os.environ.get("TEE_CMM_STL_MANIFEST")
    if not manifest_path:
        raise ValueError("TEE_CMM_STL_MANIFEST is required for physical point features")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    geometry, audit = geometry_features(points, manifest)
    motion, programs = motion_features(points, geometry)
    added = pd.DataFrame(np.column_stack([geometry, motion]), index=points.index,
                         columns=GEOMETRY_COLS + MOTION_COLS)
    return pd.concat([points, added], axis=1), {
        "geometry_columns": GEOMETRY_COLS, "motion_columns": MOTION_COLS,
        "source_audit": audit, "nc_programs": programs, "manifest": str(Path(manifest_path).resolve()),
    }
