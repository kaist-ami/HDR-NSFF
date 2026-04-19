import torch
import torch.nn as nn

class ColorRefiner(nn.Module):
    def __init__(self, in_dim=3*20, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, 3)
        )

    def forward(self, pe):
        return self.mlp(pe)