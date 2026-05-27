# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load obstacle.offline data from the PAI dataset for grounded CoC reward.

The obstacle.offline feature contains per-clip parquet files with tracked
obstacle information.  This module provides helpers to load this data and
extract scene-grounded facts that can be used to verify Chain-of-Causation
reasoning.

IMPORTANT: This module is designed to be extremely robust to schema
variations.  The obstacle.offline parquet schema is not officially
documented, so we handle multiple possible column name conventions and
nested struct types.
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("cosmos")


def load_obstacle_offline(
    clip_id: str,
    avdi: Any,
    t0_us: int,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
) -> dict[str, Any] | None:
    """Load obstacle.offline data for a specific clip and time window.

    Returns:
        Dict with obstacle data or None if not available.  Keys:
          - ``obstacles``: list of dicts with per-obstacle info
          - ``closest_obstacle``: dict with nearest obstacle info
          - ``obstacle_summary``: str description of the scene
    """
    try:
        feature_name = "obstacle.offline"
        if feature_name not in avdi.features.features_df.index:
            return None

        obstacle_data = avdi.get_clip_feature(clip_id, feature_name)
        if obstacle_data is None:
            return None

    except Exception as e:
        logger.warning(f"Failed to load obstacle.offline for {clip_id}: {e}")
        return None

    # obstacle_data from pai_utils generic zip handler is a dict:
    #   {"obstacle.offline": pd.DataFrame}
    obstacle_df = None
    if isinstance(obstacle_data, dict):
        for key, value in obstacle_data.items():
            if isinstance(value, pd.DataFrame):
                obstacle_df = value
                break
            elif isinstance(value, (bytes, io.BytesIO)):
                buf = value if isinstance(value, io.BytesIO) else io.BytesIO(value)
                obstacle_df = pd.read_parquet(buf)
                break
    elif isinstance(obstacle_data, pd.DataFrame):
        obstacle_df = obstacle_data

    if obstacle_df is None or obstacle_df.empty:
        return None

    return _parse_obstacle_df(obstacle_df, t0_us)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name (case-insensitive)."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def _extract_nested(df: pd.DataFrame, col: str, sub: str) -> pd.Series | None:
    """Extract a sub-field from a struct/nested column."""
    try:
        series = df[col]
        if hasattr(series, 'dtypes') and hasattr(series.dtypes, 'pyarrow_dtype'):
            # PyArrow struct
            return series.struct.field(sub)
        # Try dict-like access
        first = series.iloc[0]
        if isinstance(first, dict) and sub in first:
            return series.apply(lambda x: x.get(sub) if isinstance(x, dict) else None)
        if hasattr(first, '__getattr__'):
            return series.apply(lambda x: getattr(x, sub, None))
    except Exception:
        pass
    return None


def _parse_obstacle_df(
    df: pd.DataFrame,
    t0_us: int,
) -> dict[str, Any] | None:
    """Parse obstacle dataframe into structured obstacle information.

    Handles multiple possible schemas robustly.
    """
    if df is None or df.empty:
        return None

    # Log schema for debugging (only first time)
    if not hasattr(_parse_obstacle_df, '_logged_schemas'):
        _parse_obstacle_df._logged_schemas = set()
    schema_key = tuple(sorted(str(c) for c in df.columns))
    if schema_key not in _parse_obstacle_df._logged_schemas:
        _parse_obstacle_df._logged_schemas.add(schema_key)
        logger.info(f"[ObstacleParser] Schema columns: {list(df.columns)}")
        logger.info(f"[ObstacleParser] Shape: {df.shape}, dtypes: {dict(df.dtypes)}")
        if len(df) > 0:
            logger.info(f"[ObstacleParser] Sample row: {df.iloc[0].to_dict()}")

    # Normalize column names
    col_map = {}
    for c in df.columns:
        col_map[str(c).strip().lower()] = c

    # Find timestamp column
    ts_col = None
    for candidate in ["timestamp", "time", "ts", "timestamp_us", "t"]:
        if candidate in col_map:
            ts_col = col_map[candidate]
            break

    # Find track_id column
    track_col = None
    for candidate in ["track_id", "trackid", "object_id", "id", "track", "agent_id"]:
        if candidate in col_map:
            track_col = col_map[candidate]
            break

    # Find object_type column
    type_col = None
    for candidate in ["object_type", "type", "class", "category", "label",
                       "agent_type", "object_class"]:
        if candidate in col_map:
            type_col = col_map[candidate]
            break

    # Find position columns (flat)
    pos_cols = {}
    for axis in ("x", "y", "z"):
        for candidate in [axis, f"pos_{axis}", f"position_{axis}", f"center_{axis}",
                          f"translation_{axis}"]:
            if candidate in col_map:
                pos_cols[axis] = col_map[candidate]
                break

    # Find velocity columns (flat)
    vel_cols = {}
    for axis in ("x", "y", "z"):
        for candidate in [f"v{axis}", f"vel_{axis}", f"velocity_{axis}",
                          f"speed_{axis}"]:
            if candidate in col_map:
                vel_cols[axis] = col_map[candidate]
                break

    # Find heading column
    heading_col = None
    for candidate in ["heading", "yaw", "rotation_z", "orientation",
                       "heading_rad", "yaw_rad"]:
        if candidate in col_map:
            heading_col = col_map[candidate]
            break

    # Find bbox columns
    bbox_cols = {}
    for dim in ("length", "width", "height"):
        for candidate in [dim, f"bbox_{dim}", f"size_{dim}", f"lwh_{dim[0]}",
                          f"dim_{dim}"]:
            if candidate in col_map:
                bbox_cols[dim] = col_map[candidate]
                break

    # Check for nested struct columns (e.g., "position" as struct{x,y,z})
    if not pos_cols:
        for struct_name in ["position", "pos", "translation", "center", "location"]:
            if struct_name in col_map:
                for axis in ("x", "y", "z"):
                    extracted = _extract_nested(df, col_map[struct_name], axis)
                    if extracted is not None:
                        tmp_col = f"_pos_{axis}"
                        df = df.copy()
                        df[tmp_col] = extracted
                        pos_cols[axis] = tmp_col
                if pos_cols:
                    break

    if not vel_cols:
        for struct_name in ["velocity", "vel", "speed"]:
            if struct_name in col_map:
                for axis in ("x", "y", "z"):
                    extracted = _extract_nested(df, col_map[struct_name], axis)
                    if extracted is not None:
                        tmp_col = f"_vel_{axis}"
                        df = df.copy()
                        df[tmp_col] = extracted
                        vel_cols[axis] = tmp_col
                if vel_cols:
                    break

    # If we still can't find basic columns, return minimal data
    if not ts_col and not pos_cols:
        logger.warning(
            f"[ObstacleParser] Cannot find timestamp or position columns. "
            f"Available: {list(df.columns)}"
        )
        # Return minimal obstacle info — just count and types
        n_rows = len(df)
        type_counts = {}
        if type_col:
            for t in df[type_col].dropna().unique():
                type_counts[str(t)] = type_counts.get(str(t), 0) + 1

        return {
            "obstacles": [],
            "closest_obstacle": None,
            "obstacle_summary": f"{n_rows} rows, types: {type_counts}",
            "raw_row_count": n_rows,
        }

    # Build per-track obstacle entries
    t0_s = t0_us * 1e-6
    obstacles = []

    # Group by track_id if available
    if track_col:
        groups = df.groupby(track_col)
    else:
        # No track_id — treat all rows as one group
        groups = [(0, df)]

    for track_id, group in groups:
        entry: dict[str, Any] = {"track_id": int(track_id) if not isinstance(track_id, str) else hash(track_id) % 100000}

        # Object type
        if type_col is not None and len(group) > 0:
            try:
                mode_val = group[type_col].mode()
                entry["object_type"] = str(mode_val.iloc[0]) if len(mode_val) > 0 else "unknown"
            except Exception:
                entry["object_type"] = "unknown"
        else:
            entry["object_type"] = "unknown"

        # Positions
        n = len(group)
        if len(pos_cols) >= 2:
            positions = np.zeros((n, 3), dtype=np.float32)
            for i, axis in enumerate(("x", "y", "z")):
                if axis in pos_cols:
                    try:
                        positions[:, i] = group[pos_cols[axis]].values.astype(np.float32)
                    except Exception:
                        pass
            entry["positions"] = positions
        else:
            entry["positions"] = np.zeros((n, 3), dtype=np.float32)

        # Velocities
        if vel_cols:
            velocities = np.zeros((n, 3), dtype=np.float32)
            for i, axis in enumerate(("x", "y", "z")):
                if axis in vel_cols:
                    try:
                        velocities[:, i] = group[vel_cols[axis]].values.astype(np.float32)
                    except Exception:
                        pass
            entry["velocities"] = velocities
        else:
            entry["velocities"] = np.zeros((n, 3), dtype=np.float32)

        # Heading
        if heading_col is not None:
            try:
                entry["heading"] = group[heading_col].values.astype(np.float32)
            except Exception:
                entry["heading"] = np.zeros(n, dtype=np.float32)
        else:
            entry["heading"] = np.zeros(n, dtype=np.float32)

        # BBox
        if bbox_cols:
            lwh = [4.0, 2.0, 1.5]
            for i, dim in enumerate(("length", "width", "height")):
                if dim in bbox_cols:
                    try:
                        lwh[i] = float(group[bbox_cols[dim]].iloc[0])
                    except Exception:
                        pass
            entry["bbox_lwh"] = np.array(lwh, dtype=np.float32)
        else:
            entry["bbox_lwh"] = np.array([4.0, 2.0, 1.5], dtype=np.float32)

        # Distance to ego
        entry["distances"] = np.linalg.norm(entry["positions"][:, :2], axis=-1)

        # Timestamps relative to t0
        if ts_col is not None:
            try:
                ts_vals = group[ts_col].values.astype(np.float64)
                if ts_vals.max() > 1e12:
                    ts_seconds = ts_vals * 1e-6
                elif ts_vals.max() > 1e9:
                    ts_seconds = ts_vals * 1e-3
                else:
                    ts_seconds = ts_vals
                entry["timestamps_rel"] = (ts_seconds - t0_s).astype(np.float32)
            except Exception:
                entry["timestamps_rel"] = np.zeros(n, dtype=np.float32)
        else:
            entry["timestamps_rel"] = np.zeros(n, dtype=np.float32)

        obstacles.append(entry)

    # Find closest obstacle at t0
    closest_obstacle = None
    min_dist = float("inf")
    for obs in obstacles:
        t0_idx = int(np.argmin(np.abs(obs["timestamps_rel"])))
        dist_at_t0 = float(obs["distances"][t0_idx])
        if dist_at_t0 < min_dist:
            min_dist = dist_at_t0
            closest_obstacle = {
                "track_id": obs["track_id"],
                "object_type": obs["object_type"],
                "distance": dist_at_t0,
                "position": obs["positions"][t0_idx].tolist(),
                "velocity": obs["velocities"][t0_idx].tolist(),
                "heading": float(obs["heading"][t0_idx]),
            }

    # Build summary
    type_counts: dict[str, int] = {}
    for obs in obstacles:
        t = obs["object_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    summary_parts = [f"{count} {t}" for t, count in sorted(type_counts.items())]
    summary = ", ".join(summary_parts) if summary_parts else "no obstacles"

    return {
        "obstacles": obstacles,
        "closest_obstacle": closest_obstacle,
        "obstacle_summary": summary,
    }


def extract_scene_facts_from_obstacles(
    obstacle_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract verifiable scene facts from obstacle data for CoC grounding."""
    empty_facts = {
        "has_vehicle_nearby": False,
        "has_pedestrian_nearby": False,
        "has_cyclist_nearby": False,
        "closest_object_type": None,
        "closest_distance": float("inf"),
        "object_on_left": False,
        "object_on_right": False,
        "object_ahead": False,
        "approaching_object": False,
        "num_obstacles": 0,
        "high_threat_objects": [],
    }

    if obstacle_data is None or not obstacle_data.get("obstacles"):
        return empty_facts

    obstacles = obstacle_data["obstacles"]
    closest = obstacle_data.get("closest_obstacle")

    vehicle_types = {"vehicle", "car", "truck", "bus", "motorcycle", "sedan", "suv", "van", "pickup"}
    pedestrian_types = {"pedestrian", "person", "ped", "walker"}
    cyclist_types = {"cyclist", "bicycle", "bike", "motorcyclist"}

    has_vehicle = False
    has_pedestrian = False
    has_cyclist = False
    high_threat: list[dict] = []
    THREAT_DISTANCE = 30.0

    for obs in obstacles:
        obj_type_raw = obs.get("object_type", "")
        # Handle both string and integer types
        if isinstance(obj_type_raw, (int, float)):
            # Common PAI dataset encoding: 0=vehicle, 1=pedestrian, 2=cyclist, etc.
            obj_type_int = int(obj_type_raw)
            if obj_type_int == 0 or obj_type_int in [4, 5, 6, 7, 8]:  # Various vehicle types
                has_vehicle = True
                obj_type = "vehicle"
            elif obj_type_int == 1 or obj_type_int == 9:  # Pedestrian
                has_pedestrian = True
                obj_type = "pedestrian"
            elif obj_type_int == 2 or obj_type_int == 3:  # Cyclist
                has_cyclist = True
                obj_type = "cyclist"
            else:
                obj_type = f"type_{obj_type_int}"
        else:
            obj_type = str(obj_type_raw).lower().strip()
            if any(vt in obj_type for vt in vehicle_types):
                has_vehicle = True
            if any(pt in obj_type for pt in pedestrian_types):
                has_pedestrian = True
            if any(ct in obj_type for ct in cyclist_types):
                has_cyclist = True

        min_dist = float(np.min(obs.get("distances", [float("inf")])))

        if min_dist < THREAT_DISTANCE:
            positions = obs.get("positions", np.zeros((1, 3)))
            velocities = obs.get("velocities", np.zeros((1, 3)))
            ts_rel = obs.get("timestamps_rel", np.zeros(1))
            t0_idx = int(np.argmin(np.abs(ts_rel)))
            pos = positions[min(t0_idx, len(positions) - 1)]
            vel = velocities[min(t0_idx, len(velocities) - 1)]
            dist = max(float(np.linalg.norm(pos[:2])), 0.1)
            radial_vel = float(np.dot(pos[:2], vel[:2]) / dist)

            high_threat.append({
                "track_id": obs.get("track_id", 0),
                "object_type": obs.get("object_type", "unknown"),
                "distance": float(min_dist),
                "position": pos.tolist(),
                "radial_velocity": radial_vel,
                "is_approaching": radial_vel < -0.5,
            })

    obj_left = False
    obj_right = False
    obj_ahead = False
    approaching = False
    closest_dist = float("inf")
    closest_type = None

    if closest is not None:
        pos = np.array(closest.get("position", [0, 0, 0])[:2])
        closest_dist = closest.get("distance", float("inf"))
        closest_type = closest.get("object_type")

        if pos[1] > 0.5:
            obj_left = True
        elif pos[1] < -0.5:
            obj_right = True
        if pos[0] > 1.0:
            obj_ahead = True

        vel = np.array(closest.get("velocity", [0, 0, 0])[:2])
        if closest_dist > 0.1:
            radial = float(np.dot(pos, vel) / closest_dist)
            approaching = radial < -0.5

    return {
        "has_vehicle_nearby": has_vehicle,
        "has_pedestrian_nearby": has_pedestrian,
        "has_cyclist_nearby": has_cyclist,
        "closest_object_type": closest_type,
        "closest_distance": closest_dist,
        "object_on_left": obj_left,
        "object_on_right": obj_right,
        "object_ahead": obj_ahead,
        "approaching_object": approaching,
        "num_obstacles": len(obstacles),
        "high_threat_objects": high_threat,
    }
