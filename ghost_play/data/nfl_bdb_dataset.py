# ghost_play/data/nfl_bdb_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class PlayIndexEntry:
    gameId: int
    playId: int
    week: int
    frame_ids: List[int]
    nfl_ids: List[int]  # ordered list of 22 players


DEFAULT_ROLE_MAP: Dict[str, int] = {
    # Offense skill roles
    "QB": 0,
    "RB": 1,
    "FB": 1,
    "WR": 1,
    "TE": 1,
    # Deep coverage (safeties / corners)
    "FS": 2,
    "SS": 2,
    "CB": 2,
    "DB": 2,
    # Box defenders
    "LB": 3,
    "ILB": 3,
    "MLB": 3,
    "OLB": 3,
    # Lines
    "C": 4,
    "G": 4,
    "T": 4,
    "OG": 4,
    "OT": 4,
    "DT": 5,
    "DE": 5,
    "DL": 5,
    # Special teams and unknown
    "P": 6,
    "K": 6,
    "LS": 6,
}


class NFLPlayDataset(Dataset):
    """
    NFL Big Data Bowl tracking → PIGT-ready tensors.

    Returns per item:
        x_past:       [T_in, N, F]     (x, y, s, a, o, dir)
        pos_past:     [T_in, N, 2]
        pos_future:   [T_future, N, 2]
        role_ids:     [N]
        global_goal:  [goal_dim]       (placeholder; you can overwrite)
        meta: dict with gameId, playId, week, nfl_ids, etc.

    All coordinates are standardized so the **offense always moves left→right**,
    following standard BDB preprocessing (flip x, y, dir, o when playDirection=='left'). :contentReference[oaicite:3]{index=3}
    """

    def __init__(
        self,
        data_dir: str | Path,
        weeks: Sequence[int],
        T_in: int = 10,
        T_future: int = 20,
        fps: float = 10.0,
        role_map: Optional[Dict[str, int]] = None,
        tracking_pattern: str = "week{week}.csv",  # or "tracking_week_{week}.csv"
        goal_dim: int = 16,
        feature_keys: Sequence[str] = ("x", "y", "s", "a", "o", "dir"),
        max_plays: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.weeks = list(weeks)
        self.T_in = T_in
        self.T_future = T_future
        self.T_total = T_in + T_future
        self.fps = fps
        self.dt = 1.0 / fps
        self.goal_dim = goal_dim
        self.feature_keys = list(feature_keys)
        self.pos_dim = 2
        self.role_map = role_map or DEFAULT_ROLE_MAP

        # Load base tables
        self.plays = pd.read_csv(self.data_dir / "plays.csv")
        self.players = pd.read_csv(self.data_dir / "players.csv")
        # Optional: PFF roles if you want more detailed roles
        pff_path = self.data_dir / "pffScoutingData.csv"
        self.pff = pd.read_csv(pff_path) if pff_path.exists() else None

        # Map nflId -> position string
        self._nfl_to_position: Dict[int, str] = (
            self.players.set_index("nflId")["position"].to_dict()
        )

        # Load tracking weeks into memory (skeleton; you can memory-optimize later)
        tracking_frames = []
        for w in self.weeks:
            fname = tracking_pattern.format(week=w)
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Tracking file not found: {path}")
            df = pd.read_csv(path)
            df["week"] = w
            tracking_frames.append(df)
        self.tracking = pd.concat(tracking_frames, ignore_index=True)

        # Join playDirection from plays & standardize coordinates
        self._standardize_coordinates()

        # Build per-play index of usable sequences
        self.index: List[PlayIndexEntry] = self._build_index(max_plays=max_plays)

    # ------------------------ coordinate standardization ------------------------

    def _standardize_coordinates(self) -> None:
        """
        Flip x, y, dir, o when offense moving left so that offense always
        goes left→right (increasing x). :contentReference[oaicite:4]{index=4}
        """
        plays_small = self.plays[["gameId", "playId", "playDirection"]]
        self.tracking = self.tracking.merge(
            plays_small, on=["gameId", "playId"], how="left", validate="m:1"
        )

        left_mask = self.tracking["playDirection"] == "left"

        # Flip x: 0-120 yard field
        self.tracking.loc[left_mask, "x"] = 120.0 - self.tracking.loc[left_mask, "x"]

        # Flip y: width 160/3 (53.3 yards)
        field_width = 160.0 / 3.0
        self.tracking.loc[left_mask, "y"] = field_width - self.tracking.loc[left_mask, "y"]

        # Flip dir and o (degrees)
        for ang_col in ["dir", "o"]:
            if ang_col in self.tracking.columns:
                self.tracking.loc[left_mask, ang_col] = (
                    self.tracking.loc[left_mask, ang_col] + 180.0
                )
                self.tracking.loc[self.tracking[ang_col] > 360.0, ang_col] -= 360.0

    # ------------------------ index building ------------------------

    def _build_index(self, max_plays: Optional[int] = None) -> List[PlayIndexEntry]:
        """
        For each (gameId, playId, week), find contiguous frame windows with:
          - exactly 22 unique players (nflId)
          - at least T_total frames
        Take the earliest such window per play.

        This is a skeleton: you can refine conditions (e.g. only pass plays, etc.).
        """
        idx: List[PlayIndexEntry] = []

        # Drop ball rows (team == "football") if present
        df = self.tracking
        if "team" in df.columns:
            df = df[df["team"] != "football"]

        # Group by play
        grouped = df.groupby(["week", "gameId", "playId"], sort=False)

        for (week, gameId, playId), g in grouped:
            # Unique players on field
            nfl_ids = g["nflId"].dropna().unique().tolist()
            if len(nfl_ids) != 22:
                continue

            # Frames sorted
            frame_ids = sorted(g["frameId"].unique())
            if len(frame_ids) < self.T_total:
                continue

            # Take first T_total frames for this play
            frame_window = frame_ids[: self.T_total]

            idx.append(
                PlayIndexEntry(
                    gameId=int(gameId),
                    playId=int(playId),
                    week=int(week),
                    frame_ids=frame_window,
                    nfl_ids=[int(x) for x in nfl_ids],
                )
            )

            if max_plays is not None and len(idx) >= max_plays:
                break

        print(f"[NFLPlayDataset] Built index with {len(idx)} plays.")
        return idx

    def __len__(self) -> int:
        return len(self.index)

    # ------------------------ helpers ------------------------

    def _encode_roles(self, nfl_ids: Sequence[int]) -> np.ndarray:
        roles = np.zeros(len(nfl_ids), dtype=np.int64)
        for i, nid in enumerate(nfl_ids):
            pos = self._nfl_to_position.get(nid, "UNK")
            roles[i] = self.role_map.get(pos, 0)
        return roles

    def _make_global_goal(self, gameId: int, playId: int) -> np.ndarray:
        """
        Placeholder goal tensor.
        You can encode things like route depth, target WR, etc. here.
        For now: all zeros.
        """
        return np.zeros(self.goal_dim, dtype=np.float32)

    # ------------------------ main API ------------------------

    def __getitem__(self, idx: int):
        entry = self.index[idx]
        gameId, playId, week = entry.gameId, entry.playId, entry.week
        frame_ids = entry.frame_ids
        nfl_ids = entry.nfl_ids
        N = len(nfl_ids)

        # Filter to this play's frames
        g = self.tracking[
            (self.tracking["week"] == week)
            & (self.tracking["gameId"] == gameId)
            & (self.tracking["playId"] == playId)
            & (self.tracking["frameId"].isin(frame_ids))
            & (self.tracking["nflId"].isin(nfl_ids))
        ].copy()

        # Pivot into [T_total, N, feats]
        T_total = self.T_total
        F = len(self.feature_keys)
        x_seq = np.zeros((T_total, N, F), dtype=np.float32)
        pos_seq = np.zeros((T_total, N, 2), dtype=np.float32)

        # Sort frames to stable order
        frame_ids_sorted = sorted(frame_ids)
        frame_to_idx = {fid: t for t, fid in enumerate(frame_ids_sorted)}

        # Speed up lookups by indexing by frameId + nflId
        g = g.set_index(["frameId", "nflId"])

        for fid in frame_ids_sorted:
            t = frame_to_idx[fid]
            for j, nid in enumerate(nfl_ids):
                try:
                    row = g.loc[(fid, nid)]
                except KeyError:
                    # Missing sample, leave zeros
                    continue
                # features
                feat_vals = [row[k] for k in self.feature_keys]
                x_seq[t, j, :] = np.array(feat_vals, dtype=np.float32)
                pos_seq[t, j, 0] = float(row["x"])
                pos_seq[t, j, 1] = float(row["y"])

        # Split past / future
        x_past = x_seq[: self.T_in]               # [T_in, N, F]
        pos_past = pos_seq[: self.T_in]           # [T_in, N, 2]
        pos_future = pos_seq[self.T_in :]         # [T_future, N, 2]

        # Roles + goal
        role_ids = self._encode_roles(nfl_ids)    # [N]
        global_goal = self._make_global_goal(gameId, playId)

        sample = {
            "x_past": torch.from_numpy(x_past),          # [T_in, N, F]
            "pos_past": torch.from_numpy(pos_past),      # [T_in, N, 2]
            "pos_future": torch.from_numpy(pos_future),  # [T_future, N, 2]
            "role_ids": torch.from_numpy(role_ids),      # [N]
            "global_goal": torch.from_numpy(global_goal),# [goal_dim]
            "meta": {
                "gameId": gameId,
                "playId": playId,
                "week": week,
                "nfl_ids": nfl_ids,
                "dt": self.dt,
            },
        }
        return sample

    # Convenience: find index by gameId, playId
    def index_for_play(self, gameId: int, playId: int) -> Optional[int]:
        for i, entry in enumerate(self.index):
            if entry.gameId == gameId and entry.playId == playId:
                return i
        return None
