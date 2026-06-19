"""Convert smoothed track JSON into fixed-size model input tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tracks.extract import load_track_cache

# Per player: x, y, vx, vy, visible
PLAYER_FEAT_DIM = 5
# Global extras appended per frame: ball_x, ball_y, ball_valid
GLOBAL_FEAT_DIM = 3


def _track_frame_map(track: dict) -> dict[int, tuple[float, float]]:
    return {int(p["frame"]): (float(p["x"]), float(p["y"])) for p in track["points"]}


def assign_track_slots(tracks: list[dict], max_players: int) -> list[dict]:
    """Assign each track a fixed slot 0..max_players-1 by descending track length."""
    ranked = sorted(tracks, key=lambda t: len(t["points"]), reverse=True)
    return ranked[:max_players]


def build_clip_feature_matrix(
    cache: dict,
    max_players: int = 22,
    num_frames: int | None = None,
) -> np.ndarray:
    """
    Build (T, D) float32 feature matrix for a full clip.

    D = max_players * PLAYER_FEAT_DIM + GLOBAL_FEAT_DIM
    """
    width = float(cache["width"])
    height = float(cache["height"])
    t_len = int(num_frames if num_frames is not None else cache["num_frames"])
    tracks = assign_track_slots(cache.get("tracks", []), max_players)

    slot_maps: list[dict[int, tuple[float, float]]] = []
    for tr in tracks:
        slot_maps.append(_track_frame_map(tr))
    while len(slot_maps) < max_players:
        slot_maps.append({})

    ball_map = {int(p["frame"]): (float(p["x"]), float(p["y"])) for p in cache.get("ball_track", [])}

    feat_dim = max_players * PLAYER_FEAT_DIM + GLOBAL_FEAT_DIM
    out = np.zeros((t_len, feat_dim), dtype=np.float32)

    prev_xy = np.zeros((max_players, 2), dtype=np.float32)
    prev_valid = np.zeros(max_players, dtype=np.bool_)

    for t in range(t_len):
        offset = 0
        for slot in range(max_players):
            if t in slot_maps[slot]:
                x, y = slot_maps[slot][t]
                x_n = x / max(width, 1.0)
                y_n = y / max(height, 1.0)
                if prev_valid[slot]:
                    vx = (x_n - prev_xy[slot, 0])
                    vy = (y_n - prev_xy[slot, 1])
                else:
                    vx, vy = 0.0, 0.0
                prev_xy[slot] = (x_n, y_n)
                prev_valid[slot] = True
                out[t, offset : offset + 5] = (x_n, y_n, vx, vy, 1.0)
            else:
                out[t, offset + 4] = 0.0
            offset += PLAYER_FEAT_DIM

        if t in ball_map:
            bx, by = ball_map[t]
            out[t, offset : offset + 3] = (bx / max(width, 1.0), by / max(height, 1.0), 1.0)
        else:
            out[t, offset : offset + 3] = (0.0, 0.0, 0.0)

    return out


def load_clip_features(
    cache_path: Path,
    max_players: int = 22,
    num_frames: int | None = None,
) -> np.ndarray:
    cache = load_track_cache(cache_path)
    return build_clip_feature_matrix(cache, max_players=max_players, num_frames=num_frames)


def window_features(full_features: np.ndarray, start: int, end: int) -> np.ndarray:
    return full_features[start:end].copy()
