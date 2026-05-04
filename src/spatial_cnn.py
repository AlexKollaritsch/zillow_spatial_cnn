import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialCNNEncoder(nn.Module):
    """
    Input:  X_spatial (B, F, C, T)
    Output: spatial_emb (B, C, E)
    """

    def __init__(
        self,
        num_features: int,
        spatial_channels: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.spatial_dw1 = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=num_features,
        )
        self.spatial_pw1 = nn.Conv2d(num_features, spatial_channels, kernel_size=1)

        self.spatial_dw2 = nn.Conv2d(
            spatial_channels,
            spatial_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=spatial_channels,
        )
        self.spatial_pw2 = nn.Conv2d(spatial_channels, spatial_channels, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(spatial_channels)
        self.bn2 = nn.BatchNorm2d(spatial_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, F, C, T)
        x = self.spatial_dw1(X)
        x = self.spatial_pw1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)

        x = self.spatial_dw2(x)
        x = self.spatial_pw2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)

        # x: (B, S, C, T)
        x = x.mean(dim=-1)  # (B, S, C)
        x = x.permute(0, 2, 1).contiguous()  # (B, C, S)

        return x