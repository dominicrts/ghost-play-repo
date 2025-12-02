# ghost_play/models/pigt/decoder.py

from typing import Optional

import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Simple, ONNX-friendly multi-head attention on [B, L, D].
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B, H, L, Dh]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, L, H * Dh)

    def forward(
        self,
        query: torch.Tensor,   # [B, L_q, D]
        key: torch.Tensor,     # [B, L_k, D]
        value: torch.Tensor,   # [B, L_k, D]
        attn_mask: Optional[torch.Tensor] = None,  # [B, 1, L_q, L_k] broadcastable
    ) -> torch.Tensor:
        Q = self._split_heads(self.q_proj(query))
        K = self._split_heads(self.k_proj(key))
        V = self._split_heads(self.v_proj(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores + attn_mask  # additive mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        out = self._merge_heads(context)
        out = self.out_proj(out)
        return out


class RoleBasedMask(nn.Module):
    """
    Builds a [B, 1, L_q, L_k] additive attention mask from token roles and
    a role-pair allow matrix.

    role_pair_allow: [R, R] with 1 for allowed, 0 for disallowed.
    """

    def __init__(self, num_roles: int, role_pair_allow: torch.Tensor) -> None:
        super().__init__()
        assert role_pair_allow.shape == (num_roles, num_roles)
        self.num_roles = num_roles
        self.register_buffer(
            "role_pair_allow", role_pair_allow.float(), persistent=False
        )

    def forward(
        self,
        roles_q: torch.Tensor,  # [B, L_q]
        roles_k: torch.Tensor,  # [B, L_k]
        large_neg: float = -1e9,
    ) -> torch.Tensor:
        B, L_q = roles_q.shape
        _, L_k = roles_k.shape

        roles_q_oh = F.one_hot(roles_q, num_classes=self.num_roles).float()
        roles_k_oh = F.one_hot(roles_k, num_classes=self.num_roles).float()

        # roles_q_oh: [B, L_q, R]
        # role_pair_allow: [R, R]
        left = torch.matmul(roles_q_oh, self.role_pair_allow)  # [B, L_q, R]
        allowed = torch.bmm(left, roles_k_oh.transpose(1, 2))  # [B, L_q, L_k]

        mask = torch.where(
            allowed > 0.5,
            torch.zeros_like(allowed),
            torch.full_like(allowed, large_neg),
        )
        return mask.unsqueeze(1)  # [B, 1, L_q, L_k]


def build_default_role_pair_allow(num_roles: int) -> torch.Tensor:
    """
    For now: fully allowed. Later you can make this a football-specific
    matrix (e.g., Deep Zone head restrictions).
    """
    return torch.ones(num_roles, num_roles, dtype=torch.float32)


class AgentAwareTransformerDecoderLayer(nn.Module):
    """
    Single decoder layer with:
      - Role-aware self-attention over tgt.
      - Goal-conditioned cross-attention over encoder memory.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_roles: int,
        goal_dim: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        role_pair_allow: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        if role_pair_allow is None:
            role_pair_allow = build_default_role_pair_allow(num_roles)

        self.role_mask_builder = RoleBasedMask(num_roles, role_pair_allow)

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.goal_proj = nn.Linear(goal_dim, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x

    def forward(
        self,
        tgt: torch.Tensor,               # [B, L_tgt, D]
        tgt_roles: torch.Tensor,         # [B, L_tgt]
        memory: torch.Tensor,            # [B, L_mem, D]
        global_goal: torch.Tensor,       # [B, goal_dim]
        memory_roles: Optional[torch.Tensor] = None,  # [B, L_mem]
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L_tgt, D = tgt.shape
        _, L_mem, _ = memory.shape

        if memory_roles is None:
            memory_roles = tgt_roles.new_zeros(B, L_mem)

        # === 1) Role-aware self-attention ===
        role_mask = self.role_mask_builder(tgt_roles, tgt_roles)  # [B,1,L_tgt,L_tgt]

        if tgt_key_padding_mask is not None:
            pad_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(2).to(tgt.dtype)
            pad_mask = torch.where(
                pad_mask > 0.5,
                torch.full_like(pad_mask, -1e9),
                torch.zeros_like(pad_mask),
            )
            self_attn_mask = role_mask + pad_mask
        else:
            self_attn_mask = role_mask

        residual = tgt
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=self_attn_mask)
        tgt = self.norm1(residual + self.dropout1(tgt2))

        # === 2) Goal-conditioned cross-attention ===
        goal_token = self.goal_proj(global_goal).unsqueeze(1)  # [B,1,D]
        memory_aug = torch.cat([memory, goal_token], dim=1)    # [B,L_mem+1,D]

        goal_role = memory_roles.new_zeros(B, 1)
        memory_roles_aug = torch.cat([memory_roles, goal_role], dim=1)  # [B,L_mem+1]

        if memory_key_padding_mask is not None:
            pad_token = memory_key_padding_mask.new_zeros(B, 1)
            memory_kp_aug = torch.cat([memory_key_padding_mask, pad_token], dim=1)
        else:
            memory_kp_aug = None

        cross_role_mask = self.role_mask_builder(tgt_roles, memory_roles_aug)

        if memory_kp_aug is not None:
            pad_mask_mem = memory_kp_aug.unsqueeze(1).unsqueeze(2).to(tgt.dtype)
            pad_mask_mem = torch.where(
                pad_mask_mem > 0.5,
                torch.full_like(pad_mask_mem, -1e9),
                torch.zeros_like(pad_mask_mem),
            )
            cross_attn_mask = cross_role_mask + pad_mask_mem
        else:
            cross_attn_mask = cross_role_mask

        residual = tgt
        tgt2 = self.cross_attn(tgt, memory_aug, memory_aug, attn_mask=cross_attn_mask)
        tgt = self.norm2(residual + self.dropout3(tgt2))

        # === 3) FFN ===
        residual = tgt
        tgt2 = self._ffn(tgt)
        tgt = self.norm3(residual + tgt2)

        return tgt


class AgentAwareTransformerDecoder(nn.Module):
    """
    Stacked agent-aware Transformer decoder with goal-conditioned cross-attn.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        num_roles: int,
        goal_dim: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        role_pair_allow: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                AgentAwareTransformerDecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_roles=num_roles,
                    goal_dim=goal_dim,
                    dropout=dropout,
                    dim_feedforward=dim_feedforward,
                    role_pair_allow=role_pair_allow,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_roles: torch.Tensor,
        memory: torch.Tensor,
        global_goal: torch.Tensor,
        memory_roles: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = tgt
        for layer in self.layers:
            out = layer(
                tgt=out,
                tgt_roles=tgt_roles,
                memory=memory,
                global_goal=global_goal,
                memory_roles=memory_roles,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return out
