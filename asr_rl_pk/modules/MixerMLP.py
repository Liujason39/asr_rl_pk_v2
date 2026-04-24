import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, dim_in: int, hidden_dim: int = 256, activation=nn.ELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, dim_in),
        )

    def forward(self, x):
        return self.net(x)


class MixerLayer(nn.Module):
    """
    x: [B, T, F]
    T = time/channel dimension
    F = feature/token dimension
    """
    def __init__(self, num_tokens: int, num_channels: int, hidden_dim: int = 256):
        super().__init__()
        self.norm_token = nn.LayerNorm(num_tokens)
        self.norm_channel = nn.LayerNorm(num_channels)

        # token-mixing: mix along feature/token axis F
        self.token_mlp = MLPBlock(num_tokens, hidden_dim)

        # channel-mixing: mix along time/channel axis T
        self.channel_mlp = MLPBlock(num_channels, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]

        # 1) token/feature mixing over F
        # print("x1:", x.shape)
        y = self.norm_token(x)          # [B, T, F]
        # print("after transpose:", y.shape)
        y = self.token_mlp(y)           # 最後一維 F
        # print("after transpose_2:", y.shape)
        x = x + y

        # 2) channel/time mixing over T
        # print("x2:", x.shape)
        y = x.transpose(1, 2)           # [B, F, T]
        # print("after transpose:", y.shape)
        y = self.norm_channel(y)        # normalize over T
        y = self.channel_mlp(y)         # mix over T
        # print("after transpose_2:", y.shape)
        y = y.transpose(1, 2)           # [B, T, F]
        # print("after transpose_3:", y.shape)
        x = x + y

        return x


class MixerMLP(nn.Module):
    """
    將 [B, T, F] 的 multimodal sequence 經過 mixer 後，
    pool 成單一 latent。
    """
    def __init__(
        self,
        num_tokens: int,     # F
        num_channels: int,   # T
        hidden_dim: int = 256,
        num_layers: int = 1,
        out_dim: int = 64,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(num_tokens)
        self.layers = nn.ModuleList([
            MixerLayer(num_tokens, num_channels, hidden_dim)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(num_tokens),
            nn.Linear(num_tokens, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = self.input_norm(x)
        for layer in self.layers:
            x = layer(x)

        # temporal pooling
        x = x.mean(dim=1)   # [B, F]
        z = self.head(x)    # [B, out_dim]
        return z