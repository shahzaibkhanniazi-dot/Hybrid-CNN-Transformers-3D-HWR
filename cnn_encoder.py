import torch
import torch.nn as nn

class SpatialTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.InstanceNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.act(self.norm(self.conv(x))))

class KinematicCNNEncoder(nn.Module):
    def __init__(self, in_channels=13, hidden_dims=[64, 128, 256]):
        super().__init__()
        layers = []
        curr_in = in_channels
        for dim in hidden_dims:
            layers.append(SpatialTemporalBlock(curr_in, dim))
            curr_in = dim
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
