# ghost_play/models/pigt/physics.py

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class KinematicIntegrator(nn.Module):
    """
    Differentiable kinematic integrator layer.

    Given predicted accelerations a_hat over future timesteps, and initial
    position & velocity from the past frames, integrates:

        v_t = v_{t-1} + a_hat_t * dt
        p_t = p_{t-1} + v_t     * dt

    All operations are standard PyTorch ops (ONNX-friendly).
    """

    def __init__(self, dt: float = 0.1) -> None:
        """
        Args:
            dt: Timestep in seconds. NFL Next Gen Stats tracking is sampled
                at 10 Hz (every 0.1s), so dt=0.1 matches the raw data.
        """
        super().__init__()
        self.dt = float(dt)

    def forward(
        self,
        acc_seq: torch.Tensor,  # [B, T, N, P]
        pos0: torch.Tensor,     # [B, N, P]
        vel0: torch.Tensor,     # [B, N, P]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            acc_seq: Predicted accelerations for each future step.
                     Shape [B, T_future, N, P].
            pos0:    Initial positions (last past frame), [B, N, P].
            vel0:    Initial velocities (estimated from past), [B, N, P].

        Returns:
            pos_seq: Integrated positions, [B, T_future, N, P].
            vel_seq: Integrated velocities, [B, T_future, N, P].
        """
        dt = self.dt

        # Change in velocity per step
        delta_v = acc_seq * dt                     # [B, T, N, P]

        # v_t = v0 + sum_{i<=t} delta_v_i
        v_seq = vel0.unsqueeze(1) + torch.cumsum(delta_v, dim=1)  # [B, T, N, P]

        # Change in position per step; p_t = p0 + sum_{i<=t} v_i * dt
        delta_p = v_seq * dt                       # [B, T, N, P]
        pos_seq = pos0.unsqueeze(1) + torch.cumsum(delta_p, dim=1)

        return pos_seq, v_seq


def compute_physics_loss(
    pos_seq: torch.Tensor,     # [B, T, N, P]
    vel_seq: torch.Tensor,     # [B, T, N, P]
    acc_seq: torch.Tensor,     # [B, T, N, P]
    *,
    # For raw yards + dt=0.1, 22–23 mph ≈ 11–11.5 yd/s.
    # v_max≈12 yd/s lets true top-end sprints pass while penalizing spikes.
    v_max: float = 12.0,
    # Acceleration cap: DBs often hit >3.5 m/s^2 (~3.8 yd/s^2); we set a soft
    # cap above that to catch crazy spikes.
    a_max: float = 8.0,
    # Teleport threshold per 0.1s frame. At 23 mph, step ≈1.1 yd. Anything
    # above ~3–4 yd/frame is visually suspicious, so start at 3.5.
    d_max: float = 3.5,
    # Loss weights
    w_vel: float = 1.0,
    w_acc_mag: float = 0.5,
    w_acc_smooth: float = 0.1,
    w_disp: float = 0.5,
    w_heading: float = 0.1,
    # Heading-change constraint: penalize turns sharper than ~60° in a
    # single 0.1s step when moving at reasonable speed.
    heading_cos_min: float = 0.5,   # cos(60°)
    speed_for_heading: float = 3.0,  # only enforce when both speeds >= 3
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Custom physics loss L_phy to encourage realistic NFL movement.

    Terms (all differentiable & ONNX-friendly):

      1) Velocity cap:    penalize ||v|| > v_max  (top speed realism).
      2) Accel magnitude: penalize ||a|| > a_max  (no crazy bursts).
      3) Accel smoothness: penalize large Δa      (jerk / noise).
      4) Teleport guard:  penalize step > d_max   (no frame jumps).
      5) Heading-change:  penalize ultra-sharp turns at speed.

    Returns:
        L_phy: scalar tensor
        metrics: dict of floats for logging (no gradients).
    """
    eps = 1e-6

    # --- 1) Velocity cap ---
    speed_sq = (vel_seq ** 2).sum(dim=-1)          # [B, T, N]
    speed = torch.sqrt(speed_sq + eps)
    vel_excess = torch.relu(speed - v_max)
    L_vel = (vel_excess ** 2).mean()

    # --- 2) Acceleration magnitude cap ---
    acc_sq = (acc_seq ** 2).sum(dim=-1)            # [B, T, N]
    acc_mag = torch.sqrt(acc_sq + eps)
    acc_excess = torch.relu(acc_mag - a_max)
    L_acc_mag = (acc_excess ** 2).mean()

    # --- 3) Acceleration smoothness (jerk penalty) ---
    if acc_seq.size(1) > 1:
        acc_diff = acc_seq[:, 1:] - acc_seq[:, :-1]    # [B, T-1, N, P]
        L_acc_smooth = (acc_diff ** 2).mean()
    else:
        L_acc_smooth = torch.zeros((), device=acc_seq.device)

    # --- 4) Teleportation / step-size penalty ---
    if pos_seq.size(1) > 1:
        step = pos_seq[:, 1:] - pos_seq[:, :-1]        # [B, T-1, N, P]
        step_dist_sq = (step ** 2).sum(dim=-1)         # [B, T-1, N]
        step_dist = torch.sqrt(step_dist_sq + eps)
        disp_excess = torch.relu(step_dist - d_max)
        L_disp = (disp_excess ** 2).mean()
    else:
        L_disp = torch.zeros((), device=pos_seq.device)

    # --- 5) Heading-change penalty for sharp turns at speed ---
    if vel_seq.size(1) > 1:
        v_prev = vel_seq[:, :-1]                       # [B, T-1, N, P]
        v_curr = vel_seq[:, 1:]                        # [B, T-1, N, P]

        v_prev_norm = torch.sqrt((v_prev ** 2).sum(dim=-1) + eps)  # [B, T-1, N]
        v_curr_norm = torch.sqrt((v_curr ** 2).sum(dim=-1) + eps)  # [B, T-1, N]

        dot = (v_prev * v_curr).sum(dim=-1)            # [B, T-1, N]
        cos_sim = dot / (v_prev_norm * v_curr_norm + eps)

        # Only enforce heading constraints when both speeds are decent
        speed_mask = (v_prev_norm >= speed_for_heading) & (v_curr_norm >= speed_for_heading)

        # Penalty when cos_sim < heading_cos_min (turn sharper than target angle)
        heading_excess = torch.relu(heading_cos_min - cos_sim)
        heading_excess = heading_excess * speed_mask.float()
        L_heading = (heading_excess ** 2).mean()
    else:
        L_heading = torch.zeros((), device=vel_seq.device)

    # Total physics loss
    L_phy = (
        w_vel * L_vel
        + w_acc_mag * L_acc_mag
        + w_acc_smooth * L_acc_smooth
        + w_disp * L_disp
        + w_heading * L_heading
    )

    metrics = {
        "L_phy": float(L_phy.detach().cpu().item()),
        "L_vel": float(L_vel.detach().cpu().item()),
        "L_acc_mag": float(L_acc_mag.detach().cpu().item()),
        "L_acc_smooth": float(L_acc_smooth.detach().cpu().item()),
        "L_disp": float(L_disp.detach().cpu().item()),
        "L_heading": float(L_heading.detach().cpu().item()),
    }

    return L_phy, metrics
