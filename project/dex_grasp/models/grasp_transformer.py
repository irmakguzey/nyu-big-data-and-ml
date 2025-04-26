from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class GraspTransformer(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.head_3d = nn.Linear(hidden_dim, 3)
        self.head_48d = nn.Linear(hidden_dim, 48)

    def forward(self, text_emb, image_emb):
        text = self.text_proj(text_emb)
        image = self.image_proj(image_emb)
        x = torch.stack([text, image], dim=1)  # [B, 2, hidden_dim]
        x = self.transformer(x)
        out_3d = self.head_3d(x[:, 0, :])
        out_48d = self.head_48d(x[:, 1, :])
        return out_3d, out_48d
