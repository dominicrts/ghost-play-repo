# scripts/run_pigt_demo.py

import torch

from ghost_play.models.pigt import PIGTModel
from ghost_play.models.pigt.physics import compute_physics_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2           # batch size (plays)
    T_in = 10       # past timesteps
    T_future = 20   # future timesteps to predict
    N = 22          # players
    in_channels = 64
    d_model = 128
    num_roles = 8
    goal_dim = 16
    pos_dim = 2
    dt = 0.1        # 10 Hz NFL tracking

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

    # --- Fake input tensors for a smoke test ---
    x_past = torch.randn(B, T_in, N, in_channels, device=device)
    pos_past = torch.randn(B, T_in, N, pos_dim, device=device)
    role_ids = torch.randint(0, num_roles, (B, N), device=device)

    # Decoder input: zero embeddings as a starting point
    tgt_init = torch.zeros(B, T_future, N, d_model, device=device)
    global_goal = torch.randn(B, goal_dim, device=device)

    model.eval()
    with torch.no_grad():
        pos_seq, vel_seq, acc_seq = model(
            x_past, pos_past, role_ids, tgt_init, global_goal
        )

    print("pos_seq shape:", pos_seq.shape)
    print("vel_seq shape:", vel_seq.shape)
    print("acc_seq shape:", acc_seq.shape)

    # --- Physics loss sanity check (no grad) ---
    L_phy, metrics = compute_physics_loss(
        pos_seq=pos_seq,
        vel_seq=vel_seq,
        acc_seq=acc_seq,
        v_max=10.0,
        d_max=5.0,
        w_vel=1.0,
        w_acc_smooth=0.1,
        w_disp=0.1,
    )
    print("Physics loss:", L_phy.item())
    print("Physics terms:", metrics)


if __name__ == "__main__":
    main()
