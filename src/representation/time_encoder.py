import torch
import torch.nn as nn
import numpy as np

class TimeIntervalEmbedding(nn.Module):
    """
    Encodes continuous purchase intervals: TE(dt) = [sin(w_k * dt), cos(w_k * dt)]
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable frequency weights
        self.frequencies = nn.Parameter(torch.randn(embed_dim // 2))

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        # deltas shape: [batch_size, seq_len]
        batch_size, seq_len = deltas.shape
        deltas_expanded = deltas.unsqueeze(-1) # [B, L, 1]
        freqs = self.frequencies.unsqueeze(0).unsqueeze(0) # [1, 1, D/2]

        angles = deltas_expanded * freqs
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embeddings # [B, L, D]