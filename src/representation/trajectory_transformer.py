import torch
import torch.nn as nn
from src.representation.time_encoder import TimeIntervalEmbedding

class BasketEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.item_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, basket_items, mask):
        # basket_items: [batch, max_basket_size]
        embeds = self.item_embedding(basket_items)
        scores = self.attention(embeds).masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(embeds * weights, dim=1) # [batch, embed_dim]

class CustomerTrajectoryTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.basket_encoder = BasketEncoder(vocab_size, d_model)
        self.time_encoder = TimeIntervalEmbedding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 32)
        )

    def forward(self, basket_sequences, time_deltas, seq_padding_mask=None):
        # basket_sequences: [B, L, max_basket_size]
        B, L, S = basket_sequences.shape
        flat_baskets = basket_sequences.view(B * L, S)
        flat_mask = flat_baskets != 0

        basket_embeds = self.basket_encoder(flat_baskets, flat_mask).view(B, L, -1)
        time_embeds = self.time_encoder(time_deltas)

        # Fuse basket content with continuous temporal delta
        trajectory_embeds = basket_embeds + time_embeds

        # Transformer sequential processing
        out = self.transformer(trajectory_embeds, src_key_padding_mask=seq_padding_mask)

        # Average pool across valid sequence length to obtain customer embedding z_i
        mask = (~seq_padding_mask).unsqueeze(-1).float()
        latent_z = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        projected = self.proj_head(latent_z)
        return nn.functional.normalize(latent_z, dim=-1), nn.functional.normalize(projected, dim=-1)