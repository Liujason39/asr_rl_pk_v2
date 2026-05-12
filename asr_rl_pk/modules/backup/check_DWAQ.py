class MixerLayer(nn.Module):
    def __init__(self, num_tokens: int, num_channels: int, hidden_dim: int = 256):
        super().__init__()
        self.norm_token = nn.LayerNorm(num_tokens)      # 對 F 做 LN
        self.norm_channel = nn.LayerNorm(num_channels)  # 對 T 做 LN

        self.token_mlp = MLPBlock(num_tokens, hidden_dim)      # 吃 F
        self.channel_mlp = MLPBlock(num_channels, hidden_dim)  # 吃 T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]

        # 1) token/feature mixing over F
        y = self.norm_token(x)          # [B, T, F]
        y = self.token_mlp(y)           # 最後一維 F
        x = x + y

        # 2) channel/time mixing over T
        y = x.transpose(1, 2)           # [B, F, T]
        y = self.norm_channel(y)        # normalize over T
        y = self.channel_mlp(y)         # mix over T
        y = y.transpose(1, 2)           # [B, T, F]
        x = x + y

        return x