"""common/boundary_model.py — Forward model for boundary experiment."""
import torch
import torch.nn as nn


class ForwardModel(nn.Module):
    def __init__(self, hidden_size: int = 2048, action_vocab: int = 5,
                 action_emb: int = 16, n_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.action_emb = nn.Embedding(action_vocab, action_emb)
        self.net = nn.Sequential(
            nn.Linear(hidden_size + action_emb, n_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_idx = (a + 2).clamp(0, 4).long()
        a_emb = self.action_emb(a_idx)
        x = torch.cat([h, a_emb], dim=-1)
        return self.net(x).squeeze(-1)