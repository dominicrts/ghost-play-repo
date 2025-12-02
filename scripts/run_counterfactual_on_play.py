# scripts/run_counterfactual_on_play.py
"""
Run a WR-deep counterfactual on a *real* NFL play from the Big Data Bowl dataset.

Example:
    python -m scripts.run_counterfactual_on_play \
        --data-dir data/nfl-big-data-bowl-2024 \
        --weeks 1 \
        --game-id 2021091200 \
        --play-id 1234 \
        --checkpoint checkpoints/pigt_latest.pt \
        --delta-y 20 \
        --out-json outputs/counterfactual_2021091200_1234.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ghost_play.data.nfl_bdb_dataset import NFLPlayDataset, DEFAULT_ROLE_MAP
from ghost_play.models.pigt import PIGTModel


WR_ROLE_ID = 1  # must match your role_map
SAFETY_ROLE_ID = 2


def build_pigt_model_for_cli(
    in_channels: int,
    dt: float,
    num_roles: int = len(DEFAULT_ROLE_MAP),
    d_model: int = 128,
    goal_dim: int = 16,
    pos_dim: int = 2,
    device: str = "cpu",
) -> PIGTModel:
    model = PIGTModel(
        in_channels=in_channels,
        d_model=d_model,
        num_roles=num_roles,
        goal_dim=goal_dim,
        num_gnn_layers=3,
        gnn_heads=4,
        k=5,
        num_decoder_layers=4,
        decoder_heads=4,
        decoder_ffn=256,
        dropout=0.1,
        pos_dim=pos_dim,
        dt=dt,
    ).to(device)
    return model


def find_first_index_where(mask: torch.Tensor, value: int) -> Optional[int]:
    idxs = (mask == value).nonzero(as_tuple=False).view(-1)
    if idxs.numel() == 0:
        return None
    return int(idxs[0].item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--weeks", type=str, default="1")
    parser.add_argument("--game-id", type=int, required=True)
    parser.add_argument("--play-id", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--delta-y", type=float, default=20.0,
                        help="WR deep offset encoded into goal tensor.")
    parser.add_argument("--out-json", type=str, default=None,
                        help="If set, save baseline & counterfactual trajectories to this JSON.")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    weeks = [int(w) for w in args.weeks.split(",")]

    print(f"[CLI] Loading dataset from {args.data_dir} (weeks={weeks})")
    dataset = NFLPlayDataset(
        data_dir=args.data_dir,
        weeks=weeks,
        T_in=10,
        T_future=20,
        goal_dim=16,
        max_plays=None,
    )

    # Find dataset index for the requested play
    idx = dataset.index_for_play(args.game_id, args.play_id)
    if idx is None:
        raise RuntimeError(f"Play (gameId={args.game_id}, playId={args.play_id}) "
                           "not found in dataset index.")

    sample = dataset[idx]
    meta = sample["meta"]
    print(f"[CLI] Using play idx={idx}, week={meta['week']}, "
          f"gameId={meta['gameId']}, playId={meta['playId']}")

    x_past = sample["x_past"].unsqueeze(0).to(device)        # [1, T_in, N, F]
    pos_past = sample["pos_past"].unsqueeze(0).to(device)    # [1, T_in, N, 2]
    role_ids = sample["role_ids"].unsqueeze(0).to(device)    # [1, N]
    global_goal_base = sample["global_goal"].unsqueeze(0).to(device)

    B, T_in, N, F = x_past.shape
    T_future = sample["pos_future"].shape[0]
    dt = dataset.dt

    # Build model & load weights
    print(f"[CLI] Building PIGTModel (in_channels={F}, dt={dt})")
    model = build_pigt_model_for_cli(
        in_channels=F, dt=dt, num_roles=len(DEFAULT_ROLE_MAP), device=device
    )
    print(f"[CLI] Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Adjust if your checkpoint has nested keys
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    d_model = model.decoder.d_model
    tgt_init = torch.zeros(B, T_future, N, d_model, device=device)

    # Choose WR & Safety indices
    rid = role_ids[0]  # [N]
    wr_idx = find_first_index_where(rid, WR_ROLE_ID)
    safety_idx = find_first_index_where(rid, SAFETY_ROLE_ID)

    if wr_idx is None:
        print("[CLI] No WR found in this play by role_id, defaulting to player 0.")
        wr_idx = 0
    if safety_idx is None:
        print("[CLI] No Safety found in this play by role_id, defaulting to player 1.")
        safety_idx = 1

    print(f"[CLI] WR index: {wr_idx}, Safety index: {safety_idx}")

    # Baseline vs WR-deep goals
    global_goal_deep = global_goal_base.clone()
    global_goal_deep[:, 0] = args.delta_y

    with torch.no_grad():
        pos_base, vel_base, acc_base = model(
            x_past=x_past,
            pos_past=pos_past,
            role_ids=role_ids,
            tgt_init=tgt_init,
            global_goal=global_goal_base,
        )
        pos_deep, vel_deep, acc_deep = model(
            x_past=x_past,
            pos_past=pos_past,
            role_ids=role_ids,
            tgt_init=tgt_init,
            global_goal=global_goal_deep,
        )

    # Measure Safety retreat at final timestep (y-axis)
    y_base = pos_base[0, -1, safety_idx, 1].item()
    y_deep = pos_deep[0, -1, safety_idx, 1].item()
    retreat = y_base - y_deep

    print(f"[CLI] Safety y (baseline): {y_base:.3f}")
    print(f"[CLI] Safety y (WR deep): {y_deep:.3f}")
    print(f"[CLI] Safety retreat:     {retreat:.3f} (positive = deeper)")

    # Optional: save JSON for browser overlay
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        pos_base_np = pos_base[0].cpu().numpy()  # [T_future, N, 2]
        pos_deep_np = pos_deep[0].cpu().numpy()

        data = {
            "meta": {
                "gameId": meta["gameId"],
                "playId": meta["playId"],
                "week": meta["week"],
                "nflIds": meta["nfl_ids"],
                "dt": meta["dt"],
                # you can add "video_url" and "video_offset" manually
            },
            "baseline": {
                "pos_seq": pos_base_np.tolist(),
            },
            "wr_deep": {
                "delta_y": args.delta_y,
                "pos_seq": pos_deep_np.tolist(),
            },
        }

        with out_path.open("w") as f:
            json.dump(data, f)
        print(f"[CLI] Saved counterfactual trajectories to {out_path}")

    print("[CLI] Done.")


if __name__ == "__main__":
    main()
