# ghost_play/models/pigt/pigt_model.py

from typing import Tuple

import torch
from torch import nn

from .gnn_encoder import SocialGATv2Encoder
from .decoder import AgentAwareTransformerDecoder, build_default_role_pair_allow
from .physics import KinematicIntegrator


class PIGTModel(nn.Module):
    """
    Player-Intent Graph Transformer (PIGT) with a differentiable physics head.

    - Past frames: encoded with SocialGATv2Encoder per frame.
    - Agents × time are flattened into memory tokens for the decoder.
    - Future: decoder produces latent embeddings which are mapped to
      accelerations. A differentiable kinematic integrator turns these into
      velocities & positions.

    Forward API (for now):

        pos_seq, vel_seq, acc_seq = model(
            x_past, pos_past, role_ids, tgt_init, global_goal
        )

    Shapes:
        x_past:      [B, T_in, N, F_in]
        pos_past:    [B, T_in, N, 2]     (or pos_dim)
        role_ids:    [B, N]
        tgt_init:    [B, T_future, N, D] (decoder token embeddings)
        global_goal: [B, goal_dim]

        pos_seq:     [B, T_future, N, pos_dim]
        vel_seq:     [B, T_future, N, pos_dim]
        acc_seq:     [B, T_future, N, pos_dim]
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_roles: int,
        goal_dim: int,
        num_gnn_layers: int = 3,
        gnn_heads: int = 4,
        k: int = 5,
        num_decoder_layers: int = 4,
        decoder_heads: int = 4,
        decoder_ffn: int = 256,
        dropout: float = 0.1,
        pos_dim: int = 2,
        dt: float = 0.1,
    ) -> None:
        super().__init__()

        self.pos_dim = pos_dim
        self.dt = float(dt)

        role_pair_allow = build_default_role_pair_allow(num_roles)

        # --- Social GNN encoder over agents for each frame ---
        self.social_encoder = SocialGATv2Encoder(
            in_channels=in_channels,
            hidden_channels=d_model,
            out_channels=d_model,
            num_layers=num_gnn_layers,
            heads=gnn_heads,
            k=k,
            dropout=dropout,
        )

        # --- Agent-aware Transformer decoder ---
        self.decoder = AgentAwareTransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=decoder_heads,
            num_roles=num_roles,
            goal_dim=goal_dim,
            dropout=dropout,
            dim_feedforward=decoder_ffn,
            role_pair_allow=role_pair_allow,
        )

        # --- Physics head: embeddings → accelerations ---
        self.acc_head = nn.Linear(d_model, pos_dim)

        # --- Differentiable kinematic integrator ---
        self.integrator = KinematicIntegrator(dt=dt)

    def encode_past(
        self,
        x_past: torch.Tensor,     # [B, T_in, N, F_in]
        pos_past: torch.Tensor,   # [B, T_in, N, pos_dim]
        role_ids: torch.Tensor,   # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode past frames into memory tokens for the decoder.

        Returns:
            memory:       [B, T_in * N, D]
            memory_roles: [B, T_in * N]
        """
        B, T_in, N, F_in = x_past.shape
        device = x_past.device

        node_emb_list = []

        for t in range(T_in):
            x_t = x_past[:, t].reshape(B * N, F_in)          # [B*N, F_in]
            pos_t = pos_past[:, t].reshape(B * N, self.pos_dim)  # [B*N, pos_dim]

            roles_expanded = role_ids.reshape(B * N)         # [B*N]

            batch = (
                torch.arange(B, device=device)
                .unsqueeze(1)
                .expand(B, N)
                .reshape(-1)
            )

            node_emb, _ = self.social_encoder(x_t, pos_t, roles_expanded, batch)
            node_emb_list.append(node_emb.view(B, N, -1))

        # [B, T_in, N, D]
        h = torch.stack(node_emb_list, dim=1)
        B, T_in, N, D = h.shape

        # Flatten time × agents into memory tokens
        memory = h.view(B, T_in * N, D)  # [B, L_mem, D]

        # Broadcast roles over time: [B, T_in * N]
        memory_roles = (
            role_ids.unsqueeze(1)
            .expand(B, T_in, N)
            .reshape(B, T_in * N)
        )

        return memory, memory_roles

    def forward(
        self,
        x_past: torch.Tensor,       # [B, T_in, N, F_in]
        pos_past: torch.Tensor,     # [B, T_in, N, pos_dim]
        role_ids: torch.Tensor,     # [B, N]
        tgt_init: torch.Tensor,     # [B, T_future, N, D]
        global_goal: torch.Tensor,  # [B, goal_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through PIGT + physics.

        Returns:
            pos_seq: [B, T_future, N, pos_dim]
            vel_seq: [B, T_future, N, pos_dim]
            acc_seq: [B, T_future, N, pos_dim]
        """
        B, T_in, N, _ = x_past.shape
        _, T_future, N_tgt, D = tgt_init.shape
        assert N_tgt == N, "Number of agents in tgt_init must match x_past / pos_past."

        # --- Encode past frames into memory tokens ---
        memory, memory_roles = self.encode_past(x_past, pos_past, role_ids)

        # --- Prepare future tokens for decoder ---
        tgt = tgt_init.view(B, T_future * N, D)  # [B, L_tgt, D]

        # Roles for each future agent-time token
        tgt_roles = (
            role_ids.unsqueeze(1)
            .expand(B, T_future, N)
            .reshape(B, T_future * N)
        )

        # --- Agent-aware decoding conditioned on global goal ---
        dec_out_tokens = self.decoder(
            tgt=tgt,
            tgt_roles=tgt_roles,
            memory=memory,
            global_goal=global_goal,
            memory_roles=memory_roles,
        )  # [B, L_tgt, D]

        dec_out = dec_out_tokens.view(B, T_future, N, D)  # [B, T_future, N, D]

        # --- Map embeddings → accelerations ---
        acc_seq = self.acc_head(dec_out)  # [B, T_future, N, pos_dim]

        # --- Estimate initial velocity from last two past frames ---
        if T_in >= 2:
            pos_last = pos_past[:, -1]      # [B, N, pos_dim]
            pos_prev = pos_past[:, -2]      # [B, N, pos_dim]
            vel0 = (pos_last - pos_prev) / self.dt
        else:
            pos_last = pos_past[:, -1]
            vel0 = torch.zeros_like(pos_last)

        # --- Integrate to get future positions & velocities ---
        pos_seq, vel_seq = self.integrator(
            acc_seq=acc_seq,
            pos0=pos_last,
            vel0=vel0,
        )

        return pos_seq, vel_seq, acc_seq


class PIGTWebExport(nn.Module):
    """
    ONNX-friendly deployment wrapper for PIGT.

    This model:
      - assumes the social encoder has already been run offline,
      - takes precomputed memory + roles + last positions,
      - runs the agent-aware decoder + physics integrator,
      - outputs future positions, velocities, and accelerations.

    Inputs:
        memory:       [B, L_mem, D]
        memory_roles: [B, L_mem]
        role_ids:     [B, N]
        tgt_init:     [B, T_future, N, D]
        global_goal:  [B, goal_dim]
        pos_last:     [B, N, pos_dim]   (positions at last past frame)
        pos_prev:     [B, N, pos_dim]   (positions at second-last past frame)

    Outputs:
        pos_seq:      [B, T_future, N, pos_dim]
        vel_seq:      [B, T_future, N, pos_dim]
        acc_seq:      [B, T_future, N, pos_dim]
    """

    def __init__(
        self,
        decoder: AgentAwareTransformerDecoder,
        acc_head: nn.Linear,
        integrator: KinematicIntegrator,
        pos_dim: int,
        dt: float,
        num_roles: int,
        goal_dim: int,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.acc_head = acc_head
        self.integrator = integrator
        self.pos_dim = pos_dim
        self.dt = float(dt)
        self.num_roles = num_roles
        self.goal_dim = goal_dim

    def forward(
        self,
        memory: torch.Tensor,         # [B, L_mem, D]
        memory_roles: torch.Tensor,   # [B, L_mem]
        role_ids: torch.Tensor,       # [B, N]
        tgt_init: torch.Tensor,       # [B, T_future, N, D]
        global_goal: torch.Tensor,    # [B, goal_dim]
        pos_last: torch.Tensor,       # [B, N, pos_dim]
        pos_prev: torch.Tensor,       # [B, N, pos_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L_mem, D = memory.shape
        B2, T_future, N, D2 = tgt_init.shape

        # Flatten future tokens: [B, L_tgt, D]
        L_tgt = T_future * N
        tgt = tgt_init.view(B, L_tgt, D)  # [B, T_future * N, D]

        # Broadcast roles for future tokens: [B, L_tgt]
        tgt_roles = (
            role_ids.unsqueeze(1)
            .expand(B, T_future, N)
            .reshape(B, L_tgt)
        )

        # Decoder: agent-aware, goal-conditioned
        dec_out_tokens = self.decoder(
            tgt=tgt,
            tgt_roles=tgt_roles,
            memory=memory,
            global_goal=global_goal,
            memory_roles=memory_roles,
        )  # [B, L_tgt, D]

        dec_out = dec_out_tokens.view(B, T_future, N, D)  # [B, T_future, N, D]

        # Embeddings → accelerations
        acc_seq = self.acc_head(dec_out)  # [B, T_future, N, pos_dim]

        # Initial velocity from last two past frames
        vel0 = (pos_last - pos_prev) / self.dt

        # Integrate to get future positions & velocities
        pos_seq, vel_seq = self.integrator(
            acc_seq=acc_seq,
            pos0=pos_last,
            vel0=vel0,
        )

        return pos_seq, vel_seq, acc_seq
