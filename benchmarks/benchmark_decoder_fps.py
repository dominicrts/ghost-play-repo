import time
import torch

from ghost_play.models.pigt.decoder import AgentAwareTransformerDecoder, build_default_role_pair_allow


def benchmark_decoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 32          # batch size (number of plays)
    A = 22          # agents
    T_future = 20   # future timesteps
    L_tgt = A * T_future
    L_mem = A * 10  # e.g., 10 past timesteps

    d_model = 128
    num_heads = 4
    num_roles = 8
    goal_dim = 16

    role_pair_allow = build_default_role_pair_allow(num_roles)
    decoder = AgentAwareTransformerDecoder(
        num_layers=4,
        d_model=d_model,
        num_heads=num_heads,
        num_roles=num_roles,
        goal_dim=goal_dim,
        dropout=0.1,
        dim_feedforward=256,
        role_pair_allow=role_pair_allow,
    ).to(device).eval()

    tgt = torch.randn(B, L_tgt, d_model, device=device)
    memory = torch.randn(B, L_mem, d_model, device=device)
    tgt_roles = torch.randint(0, num_roles, (B, L_tgt), device=device)
    memory_roles = torch.randint(0, num_roles, (B, L_mem), device=device)
    global_goal = torch.randn(B, goal_dim, device=device)

    # Warmup
    for _ in range(10):
        _ = decoder(tgt, tgt_roles, memory, global_goal, memory_roles)

    iters = 50
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = decoder(tgt, tgt_roles, memory, global_goal, memory_roles)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    frames = B * iters
    fps = frames / (t1 - t0)
    print(f"Decoder effective frames/sec: {fps:.1f}")


if __name__ == "__main__":
    benchmark_decoder()
