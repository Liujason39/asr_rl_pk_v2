import torch
import torch.nn as nn

class TemporalBuffer(nn.Module):
    def __init__(self, history_len: int):
        super().__init__()
        self.history_len = history_len
        self.buffer = None  # [B, H, D]

    def reset(self, dones=None):
        if self.buffer is None:
            return
        if dones is None:
            self.buffer = None
        else:
            self.buffer[dones] = 0.0

    def init_if_needed(self, batch_size, feat_dim, device, dtype):
        if self.buffer is None or self.buffer.shape[0] != batch_size or self.buffer.shape[2] != feat_dim:
            self.buffer = torch.zeros(
                batch_size, self.history_len, feat_dim,
                device=device, dtype=dtype
            )

    @torch.no_grad()
    def append(self, x: torch.Tensor):
        """
        x: [B, D]
        return: updated buffer [B, H, D]
        """
        B, D = x.shape
        self.init_if_needed(B, D, x.device, x.dtype)
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1, :] = x
        return self.buffer

    def get(self):
        return self.buffer