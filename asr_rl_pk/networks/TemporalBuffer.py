import torch
import torch.nn as nn

class TemporalBuffer(nn.Module):
    def __init__(self, history_len: int):
        super().__init__()
        self.history_len = history_len
        self.buffer = None          # [B, H, D]
        self.is_initialized = None  # [B] bool

    def reset(self, dones=None):
        if self.buffer is None:
            return
        if dones is None:
            self.buffer = None
            self.is_initialized = None
        else:
            self.is_initialized[dones] = False

    def init_if_needed(self, batch_size, feat_dim, device, dtype):
        need_reinit = (
            self.buffer is None
            or self.buffer.shape[0] != batch_size
            or self.buffer.shape[2] != feat_dim
        )
        if need_reinit:
            self.buffer = torch.zeros(
                batch_size, self.history_len, feat_dim,
                device=device, dtype=dtype
            )
            self.is_initialized = torch.zeros(
                batch_size, device=device, dtype=torch.bool
            )

    @torch.no_grad()
    def append(self, x: torch.Tensor):
        """
        x: [B, D]
        return: updated buffer [B, H, D]
        """
        B, D = x.shape
        self.init_if_needed(B, D, x.device, x.dtype)

        # 第一次使用（或 reset 後第一次）: 用當前 x 填滿整段 history
        new_ids = ~self.is_initialized
        if new_ids.any():
            self.buffer[new_ids] = x[new_ids].unsqueeze(1).repeat(1, self.history_len, 1)
            self.is_initialized[new_ids] = True

        # 已初始化者：正常左移再補最後一格
        old_ids = self.is_initialized & (~new_ids)
        if old_ids.any():
            self.buffer[old_ids] = torch.roll(self.buffer[old_ids], shifts=-1, dims=1)
            self.buffer[old_ids, -1, :] = x[old_ids]

        return self.buffer

    def get(self):
        return self.buffer