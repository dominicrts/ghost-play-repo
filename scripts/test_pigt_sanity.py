# scripts/test_pigt_sanity.py
"""
Phase 4: Sanity checks for the Ghost Play PIGT engine.

These are *gating tests* you can run after training to answer:

  1) Logic / counterfactual:
       If the WR target goes 20 yards deeper, does the deep Safety
       retreat at least 5 yards in the simulation?

  2) Physics:
       Does any player ever move faster than 10 units/second?
       (Assuming coordinate units = yards, 10 yd/s ≈ 9.1 m/s.)

Run:
    python -m scripts.test_pigt_sanity

NOTE:
    With random/untrained weights, these assertions will likely fail.
    They are meant to validate a trained model.
"""

import math
from typing import Tuple

import torch
from torch import nn

from ghost_play.models.pigt import PIGTModel
from ghost_play.models.pigt.physics import compute_physics_loss


# ---------------------------------------------------------------------------
# Utility: build a PIGTModel consistent with your other scripts
# ---------------------------------------------------------------------------

def build_pigt_model(
    *,
    T_in: int = 10,
    T_future: int = 20,
    N: int = 22,
    in_channels: int = 64,
    d_model: int = 128,
    num_roles: int = 8,
    goal_dim: int = 16,
    pos_dim: int = 2,
    dt: float = 0.1,
    device: str = "cpu",
) -> Tuple[nn.Module, dict]:
    """
    Construct a PIGTModel and return it + a config dict.

    The config is used by the tests to shape inputs.
    """
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

    cfg = dict(
        T_in=T_in,
        T_future=T_future,
        N=N,
        in_channels=in_channels,
        d_model=d_model,
        num_roles=num_roles,
        goal_dim=goal_dim,
        pos_dim=pos_dim,
        dt=dt,
    )
    return model, cfg


# ---------------------------------------------------------------------------
# Test 1: Counterfactual logic – WR deep → Safety retreats
# ---------------------------------------------------------------------------

def test_counterfactual_safety_retreats(
    *,
    safety_role_id: int = 2,
    wr_role_id: int = 1,
    goal_deep_delta: float = 20.0,  # "20 yards deeper" encoded into goal
    retreat_threshold: float = 5.0, # Safety should retreat >= 5 yards
    min_pass_ratio: float = 0.6,    # at least 60% of safety samples clear threshold
    num_plays: int = 32,            # number of synthetic plays to sample
    device: str = "cpu",
) -> None:
    """
    Counterfactual logic benchmark:

    Over num_plays synthetic plays:

      - Identify all WRs (role == wr_role_id) and all Safeties (role == safety_role_id).
      - Run model with baseline goal.
      - Run model with "WR deep" goal (global goal modified).
      - For each Safety instance, measure retreat:

            retreat = y_base - y_deep

        (positive if Safety moved deeper / retreated).

    Metrics:
      - mean_retreat
      - fraction_retreat_pos        (retreat > 0)
      - fraction_retreat_ge_thresh  (retreat >= retreat_threshold)

    We assert:
      fraction_retreat_ge_thresh >= min_pass_ratio

    NOTE:
      With untrained weights, this is likely to fail. After training,
      this becomes a beautiful investor-facing metric.
    """
    print("\n[TEST] Counterfactual benchmark: WR deep → Safety retreats")

    model, cfg = build_pigt_model(device=device)
    model.eval()

    T_in = cfg["T_in"]
    T_future = cfg["T_future"]
    N = cfg["N"]
    in_channels = cfg["in_channels"]
    d_model = cfg["d_model"]
    num_roles = cfg["num_roles"]
    goal_dim = cfg["goal_dim"]
    pos_dim = cfg["pos_dim"]
    dt = cfg["dt"]

    B = 1  # one play per forward pass

    retreats = []

    for play_idx in range(num_plays):
        # --- Synthetic past features and positions ---
        x_past = torch.randn(B, T_in, N, in_channels, device=device)

        # Simple formation along x-axis
        pos_past = torch.zeros(B, T_in, N, pos_dim, device=device)
        for i in range(N):
            pos_past[:, :, i, 0] = -15.0 + i  # x
            pos_past[:, :, i, 1] = 0.0        # y

        # --- Roles: random, then force at least one WR & one Safety ---
        role_ids = torch.randint(0, num_roles, (B, N), device=device)

        # Ensure at least one WR and one Safety exist
        role_ids[0, 0] = wr_role_id
        role_ids[0, 1] = safety_role_id

        # Identify all WRs and Safeties in this play
        wr_mask = (role_ids[0] == wr_role_id)
        safety_mask = (role_ids[0] == safety_role_id)

        wr_indices = wr_mask.nonzero(as_tuple=False).view(-1)
        safety_indices = safety_mask.nonzero(as_tuple=False).view(-1)

        if wr_indices.numel() == 0 or safety_indices.numel() == 0:
            # Skip plays with no valid WR/Safety (should be rare because we enforced at least one)
            continue

        # --- Decoder init ---
        tgt_init = torch.zeros(B, T_future, N, d_model, device=device)

        # --- Global goals ---
        global_goal_base = torch.zeros(B, goal_dim, device=device)
        global_goal_deep = global_goal_base.clone()

        # For now, encode "WRs go deeper" as a positive scalar in dim 0.
        # Later, you can refine this to include per-player targeting.
        global_goal_deep[:, 0] = goal_deep_delta

        with torch.no_grad():
            pos_base, _, _ = model(
                x_past=x_past,
                pos_past=pos_past,
                role_ids=role_ids,
                tgt_init=tgt_init,
                global_goal=global_goal_base,
            )
            pos_deep, _, _ = model(
                x_past=x_past,
                pos_past=pos_past,
                role_ids=role_ids,
                tgt_init=tgt_init,
                global_goal=global_goal_deep,
            )

        # For each Safety in this play, measure retreat at final timestep
        for s_idx in safety_indices:
            s = int(s_idx.item())
            y_base = pos_base[0, -1, s, 1].item()
            y_deep = pos_deep[0, -1, s, 1].item()
            delta = y_base - y_deep  # positive if Safety retreated (went "deeper")
            retreats.append(delta)

    if len(retreats) == 0:
        raise RuntimeError("No Safety samples collected in counterfactual test.")

    import numpy as np

    retreats_np = np.array(retreats, dtype=float)
    mean_retreat = float(retreats_np.mean())
    frac_pos = float((retreats_np > 0.0).mean())
    frac_ge_thresh = float((retreats_np >= retreat_threshold).mean())

    print(f"num_safety_samples:         {len(retreats_np)}")
    print(f"mean_retreat (yards):       {mean_retreat:.3f}")
    print(f"fraction_retreat_pos:       {frac_pos:.3f}")
    print(f"fraction_retreat_ge_{retreat_threshold:.1f}: {frac_ge_thresh:.3f}")
    print(f"required fraction ≥ thresh: {min_pass_ratio:.3f}")

    assert frac_ge_thresh >= min_pass_ratio, (
        f"Counterfactual benchmark failed: only {frac_ge_thresh:.3f} of Safety samples "
        f"retreated ≥ {retreat_threshold:.1f} yards (required {min_pass_ratio:.3f})."
    )

    print("✅ Counterfactual benchmark passed.")

# ---------------------------------------------------------------------------
# Test 2: Physics – max speed < 10 units/second
# ---------------------------------------------------------------------------

def test_physics_max_speed(
    *,
    max_speed_units_per_s: float = 10.0,  # 10 yd/s ≈ 9.1 m/s ≈ 22 mph
    device: str = "cpu",
) -> None:
    """
    Physics sanity check:

    Compute instantaneous speeds from the predicted positions:

        v_t = ||(p_t - p_{t-1})|| / dt

    and assert that:

        max_t,i v_t(i) <= max_speed_units_per_s

    If your coordinate system is yards and dt=0.1s, then 10 yd/s
    corresponds to 1 yard per frame on average.
    """
    print("\n[TEST] Physics: max speed < "
          f"{max_speed_units_per_s:.1f} units/second")

    model, cfg = build_pigt_model(device=device)
    model.eval()

    T_in = cfg["T_in"]
    T_future = cfg["T_future"]
    N = cfg["N"]
    in_channels = cfg["in_channels"]
    d_model = cfg["d_model"]
    num_roles = cfg["num_roles"]
    goal_dim = cfg["goal_dim"]
    pos_dim = cfg["pos_dim"]
    dt = cfg["dt"]

    B = 2  # small batch

    # --- Synthetic inputs as in training demo ---
    x_past = torch.randn(B, T_in, N, in_channels, device=device)
    pos_past = torch.randn(B, T_in, N, pos_dim, device=device)
    role_ids = torch.randint(0, num_roles, (B, N), device=device)
    tgt_init = torch.zeros(B, T_future, N, d_model, device=device)
    global_goal = torch.randn(B, goal_dim, device=device)

    with torch.no_grad():
        pos_seq, vel_seq, acc_seq = model(
            x_past=x_past,
            pos_past=pos_past,
            role_ids=role_ids,
            tgt_init=tgt_init,
            global_goal=global_goal,
        )

    # Compute finite-difference velocities directly from positions
    # to double-check what integrator produced.
    # pos_seq: [B, T_future, N, 2]
    diff = pos_seq[:, 1:] - pos_seq[:, :-1]  # [B, T_future-1, N, 2]
    dist_per_step = torch.linalg.norm(diff, dim=-1)  # [B, T_future-1, N]
    speed_inst = dist_per_step / dt                  # [B, T_future-1, N]

    max_speed = speed_inst.max().item()

    print(f"Max instantaneous speed observed: {max_speed:.3f} units/s")
    print(f"Speed constraint:                {max_speed_units_per_s:.3f} units/s")

    assert max_speed <= max_speed_units_per_s + 1e-5, (
        f"Physics constraint violated: max speed {max_speed:.3f} "
        f"> {max_speed_units_per_s:.3f} units/s."
    )

    print("✅ Physics speed test passed (max speed within bound).")


# ---------------------------------------------------------------------------
# Optional: Physics loss smoke test
# ---------------------------------------------------------------------------

def smoke_test_physics_loss(device: str = "cpu") -> None:
    """
    Optional helper: run compute_physics_loss on a forward pass
    just to ensure it produces finite numbers.
    """
    print("\n[SMOKE] Physics loss finite check")

    model, cfg = build_pigt_model(device=device)
    model.eval()

    T_in = cfg["T_in"]
    T_future = cfg["T_future"]
    N = cfg["N"]
    in_channels = cfg["in_channels"]
    d_model = cfg["d_model"]
    num_roles = cfg["num_roles"]
    goal_dim = cfg["goal_dim"]
    pos_dim = cfg["pos_dim"]
    dt = cfg["dt"]

    B = 2

    x_past = torch.randn(B, T_in, N, in_channels, device=device)
    pos_past = torch.randn(B, T_in, N, pos_dim, device=device)
    role_ids = torch.randint(0, num_roles, (B, N), device=device)
    tgt_init = torch.zeros(B, T_future, N, d_model, device=device)
    global_goal = torch.randn(B, goal_dim, device=device)

    with torch.no_grad():
        pos_seq, vel_seq, acc_seq = model(
            x_past=x_past,
            pos_past=pos_past,
            role_ids=role_ids,
            tgt_init=tgt_init,
            global_goal=global_goal,
        )

    L_phy, metrics = compute_physics_loss(
        pos_seq=pos_seq,
        vel_seq=vel_seq,
        acc_seq=acc_seq,
    )
    print("L_phy:", L_phy.item())
    print("  terms:", metrics)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Ghost Play sanity checks on device={device!r}")

    # NOTE: With untrained weights, these tests may fail.
    # After training, they become your 'fool-proof' gatekeepers.
    try:
        test_counterfactual_safety_retreats(device=device)
    except AssertionError as e:
        print("❌ Counterfactual test FAILED:", e)

    try:
        test_physics_max_speed(device=device)
    except AssertionError as e:
        print("❌ Physics speed test FAILED:", e)

    # Optional: smoke test for physics loss
    smoke_test_physics_loss(device=device)
