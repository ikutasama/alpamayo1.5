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
obstacle information including 3D bounding boxes, object types, velocities,
and headings.  This module provides helpers to load this data and extract
scene-grounded facts that can be used to verify Chain-of-Causation reasoning.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import torch

from alpamayo1_5.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)


def load_obstacle_offline(
    clip_id: str,
    avdi: Any,
    t0_us: int,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
) -> dict[str, Any] | None:
    """Load obstacle.offline data for a specific clip and time window.

    Args:
        clip_id: The clip identifier.
        avdi: PhysicalAIAVDatasetLocalInterface instance.
        t0_us: Reference timestamp in microseconds.
        num_history_steps: Number of history steps (to define time window).
        num_future_steps: Number of future steps (to define time window).
        time_step: Seconds per step.

    Returns:
        Dict with obstacle data or None if not available.  Keys:
          - ``obstacles``: list of dicts, each with:
              - ``track_id``: int
              - ``object_type``: str (vehicle/pedestrian/cyclist/...)
              - ``positions``: np.ndarray of shape (T, 3) in ego frame at t0
              - ``velocities``: np.ndarray of shape (T, 3)
              - ``heading``: np.ndarray of shape (T,)
              - ``bbox_lwh``: np.ndarray of shape (3,) — length, width, height
              - ``distances``: np.ndarray of shape (T,) — distance to ego
          - ``closest_obstacle``: dict with the nearest obstacle info
          - ``obstacle_summary``: str description of the scene
    """
    try:
        feature_name = "obstacle.offline"
        if feature_name not in avdi.features.features_df.index:
            logger.warning(f"Feature '{feature_name}' not available in dataset.")
            return None

        obstacle_data = avdi.get_clip_feature(clip_id, feature_name)
        if obstacle_data is None:
            return None

        # obstacle_data is returned as a dict of {key: parquet_bytes_or_df}
        # based on pai_utils.py generic zip handling
        if isinstance(obstacle_data, dict):
            # Find the parquet data
            for key, value in obstacle_data.items():
                if isinstance(value, pd.DataFrame):
                    obstacle_df = value
                    break
                elif isinstance(value, (bytes, io.BytesIO)):
                    buf = value if isinstance(value, io.BytesIO) else io.BytesIO(value)
                    obstacle_df = pd.read_parquet(buf)
                    break
            else:
                logger.warning(f"Could not find readable obstacle data for {clip_id}")
                return None
        elif isinstance(obstacle_data, pd.DataFrame):
            obstacle_df = obstacle_data
        else:
            logger.warning(f"Unexpected obstacle data type: {type(obstacle_data)}")
            return None

    except Exception as e:
        logger.warning(f"Failed to load obstacle.offline for {clip_id}: {e}")
        return None

    # Define time window around t0
    t0_s = t0_us * 1e-6
    history_start = t0_s - num_history_steps * time_step
    future_end = t0_s + num_future_steps * time_step

    return _parse_obstacle_df(obstacle_df, t0_us, history_start, future_end)


def _parse_obstacle_df(
    df: pd.DataFrame,
    t0_us: int,
    history_start_s: float,
    future_end_s: float,
) -> dict[str, Any] | None:
    """Parse obstacle dataframe into structured obstacle information.

    The obstacle.offline parquet schema typically includes columns like:
    - timestamp (int64, microseconds)
    - track_id (int)
    - object_type (str or int category)
    - x, y, z (float, position in some reference frame)
    - vx, vy, vz (float, velocity)
    - heading / yaw (float)
    - length, width, height (float, bounding box dimensions)

    Note: The exact schema depends on the dataset version. This parser
    handles common column name variants.
    """
    if df is None or df.empty:
        return None

    # Normalize column names (lowercase, strip whitespace)
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Try to find timestamp column
    ts_col = None
    for candidate in ("timestamp", "time", "ts", "timestamp_us"):
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        logger.warning("No timestamp column found in obstacle data")
        return None

    # Convert timestamps to seconds for filtering
    timestamps = df[ts_col].values
    if timestamps.dtype in (np.int64, np.int32) and timestamps.max() > 1e12:
        # Timestamps are in microseconds
        ts_seconds = timestamps * 1e-6
    elif timestamps.dtype in (np.int64, np.int32) and timestamps.max() > 1e9:
        ts_seconds = timestamps * 1e-3  # milliseconds
    else:
        ts_seconds = timestamps.astype(np.float64)

    # Filter to time window
    mask = (ts_seconds >= history_start_s) & (ts_seconds <= future_end_s)
    df_filtered = df[mask].copy()
    if df_filtered.empty:
        return {"obstacles": [], "closest_obstacle": None, "obstacle_summary": "no obstacles detected"}

    # Find track_id column
    track_col = None
    for candidate in ("track_id", "trackid", "object_id", "id", "track"):
        if candidate in df_filtered.columns:
            track_col = candidate
            break

    if track_col is None:
        # If no track_id, treat all rows as one track
        df_filtered["_track_id"] = 0
        track_col = "_track_id"

    # Find object_type column
    type_col = None
    for candidate in ("object_type", "type", "class", "category", "label"):
        if candidate in df_filtered.columns:
            type_col = candidate
            break

    # Find position columns
    pos_cols = {}
    for axis in ("x", "y", "z"):
        for candidate in (axis, f"pos_{axis}", f"position_{axis}", f"center_{axis}"):
            if candidate in df_filtered.columns:
                pos_cols[axis] = candidate
                break

    # Find velocity columns
    vel_cols = {}
    for axis in ("x", "y", "z"):
        for candidate in (f"v{axis}", f"vel_{axis}", f"velocity_{axis}"):
            if candidate in df_filtered.columns:
                vel_cols[axis] = candidate
                break

    # Find heading column
    heading_col = None
    for candidate in ("heading", "yaw", "rotation_z", "orientation"):
        if candidate in df_filtered.columns:
            heading_col = candidate
            break

    # Find bbox columns
    bbox_cols = {}
    for dim in ("length", "width", "height"):
        for candidate in (dim, f"bbox_{dim}", f"size_{dim}", f"lwh_{dim[0]}"):
            if candidate in df_filtered.columns:
                bbox_cols[dim] = candidate
                break

    # Build per-track obstacle entries
    obstacles = []
    t0_s = t0_us * 1e-6

    for track_id, group in df_filtered.groupby(track_col):
        group = group.sort_values(ts_col)
        ts_track = ts_seconds[group.index]

        entry: dict[str, Any] = {"track_id": int(track_id)}

        # Object type
        if type_col is not None:
            entry["object_type"] = str(group[type_col].mode().iloc[0]) if len(group) > 0 else "unknown"
        else:
            entry["object_type"] = "unknown"

        # Positions
        if len(pos_cols) >= 2:
            positions = np.stack(
                [group[pos_cols.get(axis, pos_cols.get("x"))].values.astype(np.float32)
                 for axis in ("x", "y", "z") if axis in pos_cols],
                axis=-1,
            )
            if positions.shape[-1] == 2:
                positions = np.concatenate([positions, np.zeros((len(positions), 1))], axis=-1)
            entry["positions"] = positions
        else:
            entry["positions"] = np.zeros((len(group), 3), dtype=np.float32)

        # Velocities
        if vel_cols:
            velocities = np.stack(
                [group[vel_cols[axis]].values.astype(np.float32) for axis in ("x", "y", "z") if axis in vel_cols],
                axis=-1,
            )
            if velocities.shape[-1] == 2:
                velocities = np.concatenate([velocities, np.zeros((len(velocities), 1))], axis=-1)
            entry["velocities"] = velocities
        else:
            entry["velocities"] = np.zeros((len(group), 3), dtype=np.float32)

        # Heading
        if heading_col is not None:
            entry["heading"] = group[heading_col].values.astype(np.float32)
        else:
            entry["heading"] = np.zeros(len(group), dtype=np.float32)

        # BBox
        if bbox_cols:
            entry["bbox_lwh"] = np.array(
                [group[bbox_cols[dim]].values[0] for dim in ("length", "width", "height") if dim in bbox_cols],
                dtype=np.float32,
            )
        else:
            entry["bbox_lwh"] = np.array([4.0, 2.0, 1.5], dtype=np.float32)

        # Distance to ego (assuming ego is at origin in ego frame)
        entry["distances"] = np.linalg.norm(entry["positions"][:, :2], axis=-1)

        # Timestamps relative to t0
        entry["timestamps_rel"] = (ts_track - t0_s).astype(np.float32)

        obstacles.append(entry)

    # Find closest obstacle at t0 (closest timestamp to 0)
    closest_obstacle = None
    min_dist = float("inf")
    for obs in obstacles:
        # Find the frame closest to t0
        t0_idx = np.argmin(np.abs(obs["timestamps_rel"]))
        dist_at_t0 = obs["distances"][t0_idx]
        if dist_at_t0 < min_dist:
            min_dist = dist_at_t0
            closest_obstacle = {
                "track_id": obs["track_id"],
                "object_type": obs["object_type"],
                "distance": float(dist_at_t0),
                "position": obs["positions"][t0_idx].tolist(),
                "velocity": obs["velocities"][t0_idx].tolist(),
                "heading": float(obs["heading"][t0_idx]),
            }

    # Build summary string
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
    """Extract verifiable scene facts from obstacle data for CoC grounding.

    Returns a dict of ground-truth facts that can be checked against
    the CoC text:
      - ``has_vehicle_nearby``: bool
      - ``has_pedestrian_nearby``: bool
      - ``has_cyclist_nearby``: bool
      - ``closest_object_type``: str or None
      - ``closest_distance``: float or inf
      - ``object_on_left``: bool (closest object is to the left)
      - ``object_on_right``: bool
      - ``object_ahead``: bool
      - ``approaching_object``: bool (relative velocity is negative)
      - ``num_obstacles``: int
      - ``high_threat_objects``: list of dicts
    """
    if obstacle_data is None or not obstacle_data.get("obstacles"):
        return {
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

    obstacles = obstacle_data["obstacles"]
    closest = obstacle_data.get("closest_obstacle")

    # Classify object types
    vehicle_types = {"vehicle", "car", "truck", "bus", "motorcycle", "4", "5", "6"}
    pedestrian_types = {"pedestrian", "person", "ped", "1", "walker"}
    cyclist_types = {"cyclist", "bicycle", "bike", "2", "3"}

    has_vehicle = False
    has_pedestrian = False
    has_cyclist = False
    high_threat: list[dict] = []

    THREAT_DISTANCE = 30.0  # meters

    for obs in obstacles:
        obj_type = obs["object_type"].lower().strip()
        min_dist = float(obs["distances"].min())

        if any(vt in obj_type for vt in vehicle_types):
            has_vehicle = True
        if any(pt in obj_type for pt in pedestrian_types):
            has_pedestrian = True
        if any(ct in obj_type for ct in cyclist_types):
            has_cyclist = True

        if min_dist < THREAT_DISTANCE:
            # Check if approaching (velocity towards ego)
            t0_idx = np.argmin(np.abs(obs["timestamps_rel"]))
            vel = obs["velocities"][t0_idx]
            pos = obs["positions"][t0_idx]
            # Radial velocity: negative = approaching
            dist = max(np.linalg.norm(pos[:2]), 0.1)
            radial_vel = float(np.dot(pos[:2], vel[:2]) / dist)

            high_threat.append({
                "track_id": obs["track_id"],
                "object_type": obs["object_type"],
                "distance": float(min_dist),
                "position": pos.tolist(),
                "radial_velocity": radial_vel,
                "is_approaching": radial_vel < -0.5,
            })

    # Spatial relationship of closest object
    obj_left = False
    obj_right = False
    obj_ahead = False
    approaching = False
    closest_dist = float("inf")
    closest_type = None

    if closest is not None:
        pos = np.array(closest["position"][:2])
        closest_dist = closest["distance"]
        closest_type = closest["object_type"]

        # In ego frame: x=forward, y=left
        if pos[1] > 0.5:
            obj_left = True
        elif pos[1] < -0.5:
            obj_right = True
        if pos[0] > 1.0:
            obj_ahead = True

        vel = np.array(closest["velocity"][:2])
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
