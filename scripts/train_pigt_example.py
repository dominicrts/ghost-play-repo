# scripts/train_pigt_example.py

import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ghost_play.data.nfl_bdb_dataset import NFLPlayDataset
from ghost_play.models.pigt import PIGTModel
from ghost_play.models.pigt.physics import compute_physics_loss

dataset = NFLPlayDataset(
    data_dir="data/nfl-big-data-bowl-2024",
    weeks=[1, 2, 3],
    T_in=10,
    T_future=20,
    goal_dim=16,
    max_plays=500,  # for quick experiments
)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PIGTModel(
    in_channels=len(dataset.feature_keys),
    d_model=128,
    num_roles=len(DEFAULT_ROLE_MAP),
    goal_dim=16,
    num_gnn_layers=3,
    gnn_heads=4,
    k=5,
    num_decoder_layers=4,
    decoder_heads=4,
    decoder_ffn=256,
    dropout=0.1,
    pos_dim=2,
    dt=dataset.dt,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in loader:
    x_past = batch["x_past"].to(device)         # [B, T_in, N, F]
    pos_past = batch["pos_past"].to(device)     # [B, T_in, N, 2]
    pos_future = batch["pos_future"].to(device) # [B, T_future, N, 2]
    role_ids = batch["role_ids"].to(device)     # [B, N]
    global_goal = batch["global_goal"].to(device)

    B, T_in, N, F = x_past.shape
    T_future = pos_future.shape[1]
    d_model = model.decoder.d_model

    # Simple decoder init
    tgt_init = torch.zeros(B, T_future, N, d_model, device=device)

    pos_pred, vel_pred, acc_pred = model(
        x_past=x_past,
        pos_past=pos_past,
        role_ids=role_ids,
        tgt_init=tgt_init,
        global_goal=global_goal,
    )

    # Data loss (L2 to future trajectory)
    loss_traj = torch.mean((pos_pred - pos_future) ** 2)

    # Physics regularizer
    L_phy, metrics_phy = compute_physics_loss(
        pos_seq=pos_pred,
        vel_seq=vel_pred,
        acc_seq=acc_pred,
    )

    loss = loss_traj + 0.1 * L_phy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


class DummyGhostPlayDataset(Dataset):
    """
    Placeholder dataset to demonstrate training-time usage of PIGT + L_phy.

    Replace this with a real dataset that loads NFL tracking data:
      - x_past:      past kinematic features per player
      - pos_past:    past (x, y) positions in yards
      - pos_future:  future (x, y) positions in yards
      - role_ids:    integer role IDs per player
      - global_goal: encoded user / play intent tensor
    """

    def __init__(
        self,
        num_samples: int = 256,
        T_in: int = 10,
        T_future: int = 20,
        N: int = 22,
        in_channels: int = 64,
        pos_dim: int = 2,
        num_roles: int = 8,
        goal_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.T_in = T_in
        self.T_future = T_future
        self.N = N
        self.in_channels = in_channels
        self.pos_dim = pos_dim
        self.num_roles = num_roles
        self.goal_dim = goal_dim

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # NOTE: These are random tensors just to show plumbing.
        # In the real system, replace this logic with:
        #   - loading a play
        #   - slicing past/future frames aligned to 10 Hz
        #   - converting coordinates to yards (if needed)
        x_past = torch.randn(self.T_in, self.N, self.in_channels)
        pos_past = torch.randn(self.T_in, self.N, self.pos_dim)
        pos_future = torch.randn(self.T_future, self.N, self.pos_dim)

        role_ids = torch.randint(0, self.num_roles, (self.N,), dtype=torch.long)
        global_goal = torch.randn(self.goal_dim)

        return {
            "x_past": x_past,
            "pos_past": pos_past,
            "pos_future": pos_future,
            "role_ids": role_ids,
            "global_goal": global_goal,
        }


def train_step(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lambda_phy: float = 0.1,
    v_max: float = 12.0,
    a_max: float = 8.0,
    d_max: float = 3.5,
) -> Dict[str, float]:
    """
    One training step: supervised trajectory loss + physics loss.

    L_total = L_data + lambda_phy * L_phy

    where:
      - L_data matches predicted future positions to ground truth.
      - L_phy enforces physically plausible trajectories.
    """
    model.train()
    optimizer.zero_grad()

    x_past = batch["x_past"]          # [B, T_in, N, F_in]
    pos_past = batch["pos_past"]      # [B, T_in, N, 2]
    pos_future = batch["pos_future"]  # [B, T_future, N, 2]
    role_ids = batch["role_ids"]      # [B, N]
    global_goal = batch["global_goal"]  # [B, goal_dim]

    device = next(model.parameters()).device
    x_past = x_past.to(device)
    pos_past = pos_past.to(device)
    pos_future = pos_future.to(device)
    role_ids = role_ids.to(device)
    global_goal = global_goal.to(device)

    B, T_future, N, pos_dim = pos_future.shape
    d_model = model.decoder.layers[0].self_attn.d_model  # use decoder config

    # Start decoder from zeros. Later you can learn a start token if desired.
    tgt_init = torch.zeros(B, T_future, N, d_model, device=device)

    # Forward pass: get predicted future positions/velocities/accelerations
    pos_pred, vel_pred, acc_pred = model(
        x_past=x_past,
        pos_past=pos_past,
        role_ids=role_ids,
        tgt_init=tgt_init,
        global_goal=global_goal,
    )

    # --- 1) Supervised data loss on positions ---
    # Smooth L1 is robust to occasional annotation noise.
    L_data = F.smooth_l1_loss(pos_pred, pos_future)

    # --- 2) Physics loss (Ghost Play) ---
    L_phy, phy_metrics = compute_physics_loss(
        pos_seq=pos_pred,
        vel_seq=vel_pred,
        acc_seq=acc_pred,
        v_max=v_max,
        a_max=a_max,
        d_max=d_max,
        # Other weights use defaults from physics.py
    )

    # --- 3) Total loss ---
    L_total = L_data + lambda_phy * L_phy
    L_total.backward()
    optimizer.step()

    metrics = {
        "L_total": float(L_total.detach().cpu().item()),
        "L_data": float(L_data.detach().cpu().item()),
        "L_phy": phy_metrics["L_phy"],
        "L_vel": phy_metrics["L_vel"],
        "L_acc_mag": phy_metrics["L_acc_mag"],
        "L_acc_smooth": phy_metrics["L_acc_smooth"],
        "L_disp": phy_metrics["L_disp"],
        "L_heading": phy_metrics["L_heading"],
    }
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Hyperparameters ---
    T_in = 10
    T_future = 20
    N = 22
    in_channels = 64
    d_model = 128
    num_roles = 8
    goal_dim = 16
    pos_dim = 2
    dt = 0.1  # 10 Hz

    batch_size = 8
    num_epochs = 3

    # Start with a mild physics weight and ramp it up
    lambda_phy_start = 0.05
    lambda_phy_end = 0.2

    # --- Model ---
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- Data ---
    dataset = DummyGhostPlayDataset(
        num_samples=256,
        T_in=T_in,
        T_future=T_future,
        N=N,
        in_channels=in_channels,
        pos_dim=pos_dim,
        num_roles=num_roles,
        goal_dim=goal_dim,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(num_epochs):
        # Linearly ramp lambda_phy over epochs
        alpha = epoch / max(num_epochs - 1, 1)
        lambda_phy = (1 - alpha) * lambda_phy_start + alpha * lambda_phy_end

        for i, batch in enumerate(loader):
            metrics = train_step(
                batch,
                model,
                optimizer,
                lambda_phy=lambda_phy,
                v_max=12.0,
                a_max=8.0,
                d_max=3.5,
            )

            if i % 10 == 0:
                print(
                    f"[epoch {epoch+1}/{num_epochs} step {i:03d}] "
                    f"L_total={metrics['L_total']:.4f} "
                    f"L_data={metrics['L_data']:.4f} "
                    f"L_phy={metrics['L_phy']:.4f} "
                    f"L_vel={metrics['L_vel']:.4f} "
                    f"L_acc_mag={metrics['L_acc_mag']:.4f} "
                    f"L_disp={metrics['L_disp']:.4f} "
                    f"L_heading={metrics['L_heading']:.4f}"
                )


if __name__ == "__main__":
    main()
