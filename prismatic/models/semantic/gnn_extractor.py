"""
gnn_extractor.py

Graph Neural Network based Structured Semantic Extractor for ActDistill framework.
Extracts structured semantic representations from hidden states at each layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeatureBasedEdgeBuilder(nn.Module):
    """Builds k-NN graph edges based on feature affinity (Eq. 1-2 in paper)."""
    def __init__(self, hidden_dim: int, k_neighbors: int = 8):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.phi = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.psi = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        Returns:
            edge_index: [2, num_edges]
            edge_weights: [num_edges]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        phi_h = self.phi(hidden_states)
        psi_h = self.psi(hidden_states)

        # Vectorized affinity computation
        # [B, L, D] @ [B, D, L] -> [B, L, L]
        scores = torch.matmul(phi_h, psi_h.transpose(1, 2))
        affinity = torch.exp(scores)

        # Top-k selection
        k = min(self.k_neighbors, seq_len)
        # [B, L, K]
        topk_values, topk_indices = torch.topk(affinity, k, dim=-1, largest=True)

        # Normalize weights
        norm_sum = topk_values.sum(dim=-1, keepdim=True) + 1e-8
        normalized_weights = topk_values / norm_sum

        # Construct global indices
        # [B, 1, 1]
        batch_offset = (torch.arange(batch_size, device=device) * seq_len).view(-1, 1, 1)
        # [1, L, 1] -> [B, L, K]
        src_local_idx = torch.arange(seq_len, device=device).view(1, -1, 1).expand(batch_size, -1, k)
        
        src_global = batch_offset + src_local_idx
        dst_global = batch_offset + topk_indices

        # Filter self-loops
        mask = topk_indices != src_local_idx
        
        src_final = src_global[mask]
        dst_final = dst_global[mask]
        weights_final = normalized_weights[mask]

        if src_final.numel() == 0:
            # Fallback for empty graph
            all_nodes = torch.arange(batch_size * seq_len, device=device)
            edge_index = torch.stack([all_nodes, all_nodes], dim=0)
            edge_weights = torch.ones(batch_size * seq_len, device=device)
        else:
            edge_index = torch.stack([src_final, dst_final], dim=0)
            edge_weights = weights_final
        
        return edge_index, edge_weights


class GATConv(nn.Module):
    """Graph Attention Network layer."""
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, concat: bool = True, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Linear transformations for multi-head attention
        self.linear = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, in_dim]
            edge_index: [2, num_edges]
        Returns:
            out: [num_nodes, out_dim * heads] if concat else [num_nodes, out_dim]
        """
        num_nodes = x.size(0)

        # Linear transformation
        h = self.linear(x).view(num_nodes, self.heads, self.out_dim)  # [num_nodes, heads, out_dim]

        # Edge-wise attention
        edge_src, edge_dst = edge_index[0], edge_index[1]
        h_src = h[edge_src]  # [num_edges, heads, out_dim]
        h_dst = h[edge_dst]

        # Concatenate source and destination features
        h_concat = torch.cat([h_src, h_dst], dim=-1)  # [num_edges, heads, 2*out_dim]

        # Compute attention scores
        e = self.leaky_relu((h_concat * self.att).sum(dim=-1))  # [num_edges, heads]

        # Combine learnable attention with k-NN edge weights (if provided)
        if edge_weights is not None:
            prior = edge_weights.clamp(min=1e-8).unsqueeze(-1).expand(-1, self.heads)
            weights = prior * torch.exp(e)
        else:
            weights = torch.exp(e)

        # Aggregate using index_add_ (scatter_add)
        # msg: [num_edges, heads, out_dim]
        msg = weights.unsqueeze(-1) * h_src
        
        out = torch.zeros((num_nodes, self.heads, self.out_dim), device=x.device, dtype=x.dtype)
        attention = torch.zeros((num_nodes, self.heads), device=x.device, dtype=x.dtype)
        
        out.index_add_(0, edge_dst, msg)
        attention.index_add_(0, edge_dst, weights)

        # Normalize
        attention = attention.unsqueeze(-1)  # [num_nodes, heads, 1]
        out = out / (attention + 1e-8)

        # Dropout
        out = F.dropout(out, p=self.dropout, training=self.training)

        if self.concat:
            return out.view(num_nodes, self.heads * self.out_dim)
        else:
            return out.mean(dim=1)


class AttentionPooling(nn.Module):
    """Attention-based pooling for graph aggregation."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_p = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, hidden_dim]
            batch_idx: [num_nodes]
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        # Assuming x is [B*L, D] and batch_idx is regular [0...0, 1...1, ...]
        # We can infer B and L from batch_idx
        batch_size = batch_idx.max().item() + 1
        num_nodes, hidden_dim = x.shape
        seq_len = num_nodes // batch_size
        
        # Reshape to [B, L, D]
        x_reshaped = x.view(batch_size, seq_len, hidden_dim)
        
        # Compute scores [B, L, 1]
        scores = self.w_p(x_reshaped) 
        
        # Softmax over L
        alpha = F.softmax(scores, dim=1) # [B, L, 1]
        
        # Weighted sum
        pooled = (alpha * x_reshaped).sum(dim=1) # [B, D]
        
        return pooled


class GraphSemanticExtractor(nn.Module):
    """
    Graph-based structured semantic extractor.
    Extracts semantic representations from transformer hidden states using GNN.
    """
    def __init__(
        self,
        hidden_dim: int = 4096,
        semantic_dim: int = 512,
        gnn_type: str = 'GAT',
        num_gnn_layers: int = 2,
        k_neighbors: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.semantic_dim = semantic_dim
        self.gnn_type = gnn_type

        self.edge_builder = FeatureBasedEdgeBuilder(hidden_dim, k_neighbors=k_neighbors)

        # GNN layers
        if gnn_type == 'GAT':
            self.gnn_layers = nn.ModuleList([
                GATConv(
                    hidden_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    heads=4,
                    concat=False,
                    dropout=dropout
                )
                for i in range(num_gnn_layers)
            ])
        else:
            raise NotImplementedError(f"GNN type {gnn_type} not implemented. Use 'GAT'.")

        self.pooling = AttentionPooling(hidden_dim)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, semantic_dim)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        Returns:
            semantic: [batch, semantic_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        edge_index, edge_weights = self.edge_builder(hidden_states)
        x = hidden_states.reshape(-1, hidden_dim)

        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, edge_index, edge_weights))

        batch_idx = torch.arange(batch_size, device=hidden_states.device).repeat_interleave(seq_len)
        pooled = self.pooling(x, batch_idx)

        semantic = self.proj(pooled)

        return semantic
