import os
from pathlib import Path

import torch
import onnx

from ghost_play.models.pigt import PIGTModel, PIGTWebExport

# optional: only needed if you want the int8 model
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False


def export_pigt_onnx(
    out_dir: str = "onnx_models",
    opset: int = 17,
) -> None:
    device = "cpu"

    # --- Config must match training / deployment assumptions ---
    T_in = 10
    T_future = 20
    N = 22
    in_channels = 64
    d_model = 128
    num_roles = 8
    goal_dim = 16
    pos_dim = 2
    dt = 0.1

    # Full training model (not exported directly)
    base_model = PIGTModel(
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
    base_model.eval()

    # Deployment wrapper: ONNX-friendly (no PyG / GNN inside)
    export_model = PIGTWebExport(
        decoder=base_model.decoder,
        acc_head=base_model.acc_head,
        integrator=base_model.integrator,
        pos_dim=pos_dim,
        dt=dt,
        num_roles=num_roles,
        goal_dim=goal_dim,
    ).to(device)
    export_model.eval()

    B = 1
    L_mem = T_in * N  # one token per (time, agent) from encoder

    # --- Dummy inputs for tracing ---
    memory = torch.randn(B, L_mem, d_model, device=device)
    memory_roles = torch.randint(0, num_roles, (B, L_mem), device=device, dtype=torch.long)
    role_ids = torch.randint(0, num_roles, (B, N), device=device, dtype=torch.long)
    tgt_init = torch.zeros(B, T_future, N, d_model, device=device)
    global_goal = torch.randn(B, goal_dim, device=device)
    pos_last = torch.randn(B, N, pos_dim, device=device)
    pos_prev = torch.randn(B, N, pos_dim, device=device)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir_path / "pigt_web_fp32.onnx"

    input_names = [
        "memory",
        "memory_roles",
        "role_ids",
        "tgt_init",
        "global_goal",
        "pos_last",
        "pos_prev",
    ]
    output_names = ["pos_seq", "vel_seq", "acc_seq"]

    # Dynamic axes: batch, memory length, and future horizon can vary
    dynamic_axes = {
        "memory": {0: "batch", 1: "L_mem"},
        "memory_roles": {0: "batch", 1: "L_mem"},
        "role_ids": {0: "batch"},
        "tgt_init": {0: "batch", 1: "t_future"},
        "global_goal": {0: "batch"},
        "pos_last": {0: "batch"},
        "pos_prev": {0: "batch"},
        "pos_seq": {0: "batch", 1: "t_future"},
        "vel_seq": {0: "batch", 1: "t_future"},
        "acc_seq": {0: "batch", 1: "t_future"},
    }

    print(f"Exporting PIGTWebExport to {onnx_path} (opset {opset})...")
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (memory, memory_roles, role_ids, tgt_init, global_goal, pos_last, pos_prev),
            f=str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    # Validate
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX export OK:", onnx_path)

    # Optional INT8 quantization
    if HAS_QUANT:
        q_path = out_dir_path / "pigt_web_int8.onnx"
        print("Quantizing to INT8:", q_path)
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(q_path),
            weight_type=QuantType.QInt8,
        )
        print("INT8 quantized model saved:", q_path)
    else:
        print("onnxruntime.quantization not installed; skipping quantization.")


if __name__ == "__main__":
    export_pigt_onnx()
