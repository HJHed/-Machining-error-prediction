"""Target-free geometry and local NC-command features for original training flow."""

import bisect
from pathlib import Path

import numpy as np
import pandas as pd

import nc_features


WINDOW_RADIUS = 4
COMMAND_DIM = 12
REGION_COLS = [
    "region_x_fraction", "region_y_fraction", "region_z_fraction",
    "region_radial_fraction", "region_distance_from_top",
    "region_x_edge_fraction", "region_y_edge_fraction", "region_top_quarter_strength",
]
NC_WINDOW_COLS = [
    f"nc_window_{offset:+d}_{feature:02d}"
    for offset in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1)
    for feature in range(COMMAND_DIM)
]


def command_features(moves):
    xyz = np.stack([move[1] for move in moves], axis=0)
    ac = np.stack([move[2] for move in moves], axis=0)
    feed = np.asarray([move[4] for move in moves], np.float64)
    mode = np.asarray([move[3] for move in moves], np.int32)
    delta = np.diff(xyz, axis=0, prepend=xyz[:1])
    length = np.linalg.norm(delta, axis=1)
    tangent = delta / np.maximum(length[:, None], 1e-9)
    previous_tangent = np.roll(tangent, 1, axis=0)
    previous_tangent[0] = 0
    previous_length = np.roll(length, 1)
    previous_length[0] = 0
    corner = np.arccos(np.clip(np.sum(tangent * previous_tangent, axis=1), -1, 1)) / np.pi
    corner[(length < 1e-9) | (previous_length < 1e-9)] = 0
    previous_feed = np.roll(feed, 1)
    previous_feed[0] = feed[0]
    previous_ac = np.roll(ac, 1, axis=0)
    previous_ac[0] = ac[0]
    features = np.column_stack([
        tangent,
        np.log1p(length),
        np.sign(delta[:, 2]) * np.log1p(np.abs(delta[:, 2])),
        np.log1p(np.maximum(feed, 0)),
        np.clip((feed - previous_feed) / np.maximum(np.abs(previous_feed), 1), -2, 2),
        np.log1p(np.linalg.norm(ac - previous_ac, axis=1)),
        (mode == 0).astype(float),
        np.isin(mode, (2, 3)).astype(float),
        corner,
        np.log1p(np.maximum(-delta[:, 2], 0)),
    ]).astype(np.float32)
    if features.shape[1] != COMMAND_DIM or not np.isfinite(features).all():
        raise ValueError("Invalid NC-command window features")
    return features


def geometry_features(points):
    out = np.zeros((len(points), len(REGION_COLS)), np.float32)
    for _, indexes in points.groupby("source_file_name", sort=False).indices.items():
        xyz = points.iloc[indexes][["pose_x", "pose_y", "pose_z"]].to_numpy(np.float64)
        low, high, center = xyz.min(axis=0), xyz.max(axis=0), xyz.mean(axis=0)
        scaled = np.clip((xyz - low) / np.maximum(high - low, 1e-6), 0, 1)
        radial = np.linalg.norm(xyz[:, :2] - center[None, :2], axis=1)
        radial /= max(float(np.linalg.norm(high[:2] - low[:2])), 1e-6)
        out[indexes] = np.column_stack([
            scaled, radial, 1 - scaled[:, 2],
            np.minimum(scaled[:, 0], 1 - scaled[:, 0]),
            np.minimum(scaled[:, 1], 1 - scaled[:, 1]),
            np.maximum(0, (scaled[:, 2] - .75) / .25),
        ]).astype(np.float32)
    return out


def nc_window_features(points):
    out = np.zeros((len(points), len(NC_WINDOW_COLS)), np.float32)
    manifest = nc_features.read_manifest()
    cache = {}
    columns = ["source_file_name", "original_mapped_file_index", "mapped_nc_line"]
    if not set(columns).issubset(points.columns):
        raise ValueError(f"Missing NC-window keys: {set(columns)-set(points.columns)}")
    for (source, original_index), indexes in points.groupby(columns[:2], sort=False).indices.items():
        mapping = manifest[source]
        path = mapping[str(int(original_index))] if isinstance(mapping, dict) else mapping
        path = str(Path(path).resolve())
        if path not in cache:
            moves, _, _ = nc_features.parse_program(path)
            cache[path] = ([move[0] for move in moves], command_features(moves))
        line_indexes, commands = cache[path]
        for row_index, line in zip(indexes, points.iloc[indexes].mapped_nc_line):
            center = bisect.bisect_right(line_indexes, int(line)) - 1
            if center < 0:
                # The existing NC fusion permits a small fraction of unmapped
                # lines; keep the same rows and represent their window as zero.
                continue
            for window_index, command_index in enumerate(
                    range(center - WINDOW_RADIUS, center + WINDOW_RADIUS + 1)):
                if 0 <= command_index < len(commands):
                    start = window_index * COMMAND_DIM
                    out[row_index, start:start + COMMAND_DIM] = commands[command_index]
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite NC window features")
    return out, len(cache)


def add_features(points):
    """Use only positions and NC code; no error targets or split labels."""
    region = geometry_features(points)
    window, programs = nc_window_features(points)
    added = pd.DataFrame(np.column_stack([region, window]), index=points.index,
                         columns=REGION_COLS + NC_WINDOW_COLS)
    output = pd.concat([points, added], axis=1)
    return output, {"region_columns": REGION_COLS, "nc_window_columns": NC_WINDOW_COLS,
                    "window_radius": WINDOW_RADIUS, "command_dim": COMMAND_DIM,
                    "nc_programs": programs}
