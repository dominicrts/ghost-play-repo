from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool


class SocialGATv2Encoder(nn.Module):
    """
    Spatio-social GNN encoder for a single time snapshot using GATv2.

    - Builds a dynamic k-NN graph over player positions (pure PyTorch,
      no torch-cluster dependency).
    - Enforces asymmetric interactions via role-based edge masking
      (e.g. Safety attends to WR, WR ignores Safety).
    - Produces node-level "social state" embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        k: int = 5,
        dropout: float = 0.1,
        receiver_role_id: int = 1,
        safety_role_id: int = 2,
    ) -> None:
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"
        assert hidden_channels == out_channels, (
            "hidden_channels is assumed to equal out_channels for now."
        )

        self.k = k
        self.dropout = dropout
        self.receiver_role_id = receiver_role_id
        self.safety_role_id = safety_role_id

        convs = []

        # First GATv2 layer
        convs.append(
            GATv2Conv(
                in_channels,
                hidden_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
            )
        )

        # Additional GATv2 layers
        for _ in range(num_layers - 1):
            convs.append(
                GATv2Conv(
                    hidden_channels,
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                )
            )

        self.convs = nn.ModuleList(convs)

    def _build_knn_edges_torch(
        self,
        pos: torch.Tensor,    # [N_total, 2]
        batch: torch.Tensor,  # [N_total]
    ) -> torch.Tensor:
        """
        Pure-PyTorch k-NN graph builder.

        For each graph in the batch:
          - Compute pairwise distances.
          - For each node, pick k nearest neighbors (no self-loop).
          - Build directed edges (i -> neighbors).

        Returns:
            edge_index: [2, E], where each column is (src, dst).
        """
        device = pos.device
        N_total = pos.size(0)
        assert batch.numel() == N_total

        edge_src_list = []
        edge_dst_list = []

        unique_batches = batch.unique(sorted=True)

        for b in unique_batches:
            mask = (batch == b)
            idx = mask.nonzero(as_tuple=False).view(-1)  # global indices for this graph

            if idx.numel() <= 1:
                continue

            pos_b = pos[idx]  # [N_b, 2]
            N_b = pos_b.size(0)

            # Pairwise Euclidean distances: [N_b, N_b]
            dist = torch.cdist(pos_b, pos_b, p=2)

            # Avoid self-connections by setting diagonal to +inf
            inf = float("inf")
            dist.fill_diagonal_(inf)

            # Effective k (can't be more than N_b - 1)
            k_eff = min(self.k, max(N_b - 1, 1))

            _, nn_idx = torch.topk(dist, k=k_eff, dim=-1, largest=False)

            # For each node i, connect i -> its k nearest neighbors
            src_local = torch.arange(N_b, device=device).unsqueeze(1).expand(N_b, k_eff)
            src_local = src_local.reshape(-1)         # [N_b * k_eff]
            dst_local = nn_idx.reshape(-1)            # [N_b * k_eff]

            edge_src_list.append(idx[src_local])      # map to global indices
            edge_dst_list.append(idx[dst_local])

        if len(edge_src_list) == 0:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        edge_src = torch.cat(edge_src_list, dim=0)
        edge_dst = torch.cat(edge_dst_list, dim=0)

        edge_index = torch.stack([edge_src, edge_dst], dim=0)  # [2, E]
        return edge_index

    @torch.no_grad()
    def _build_asymmetric_edge_index(
        self,
        pos: torch.Tensor,       # [N_total, 2]
        role_ids: torch.Tensor,  # [N_total]
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build dynamic k-NN graph with asymmetric interactions:

        - Build k-NN edges via pure PyTorch (per batch).
        - Make edges bidirectional (src->dst and dst->src).
        - Remove edges where src is Receiver and dst is Safety
          (so Safety->WR allowed, WR->Safety dropped).
        """
        if batch is None:
            batch = pos.new_zeros(pos.size(0), dtype=torch.long)

        # 1) k-NN edges
        edge_index = self._build_knn_edges_torch(pos, batch)  # [2, E]

        # 2) Make edges bidirectional
        edge_index_rev = edge_index.flip(0)
        edge_index = torch.cat([edge_index, edge_index_rev], dim=1)  # [2, 2E]

        # 3) Apply asymmetric role-based masking
        src, dst = edge_index

        drop_mask = (role_ids[src] == self.receiver_role_id) & (
            role_ids[dst] == self.safety_role_id
        )
        keep_mask = ~drop_mask

        edge_index = edge_index[:, keep_mask]
        return edge_index

    def forward(
        self,
        x: torch.Tensor,         # [N, in_channels]
        pos: torch.Tensor,       # [N, 2]
        role_ids: torch.Tensor,  # [N]
        batch: Optional[torch.Tensor] = None,
        return_graph_emb: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the SocialGATv2Encoder.

        Returns:
            node_emb:  [N, out_channels]
            graph_emb: [B, out_channels] or None
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        edge_index = self._build_asymmetric_edge_index(pos, role_ids, batch)

        h = x
        for conv in self.convs:
            h = conv(h, edge_index)  # [N, hidden_channels]
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        node_emb = h

        graph_emb = None
        if return_graph_emb:
            graph_emb = global_mean_pool(node_emb, batch)

        return node_emb, graph_emb
