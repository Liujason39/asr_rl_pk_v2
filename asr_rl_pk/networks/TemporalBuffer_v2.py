import torch
import torch.nn as nn

class TemporalBuffer_v2(nn.Module):
    def __init__(self, history_len: int):
        super().__init__()
        self.history_len = history_len
        self.buffer = None
        self.is_initialized = None

    def reset(self, dones=None):
        if self.buffer is None:
            return
        if dones is None:
            self.buffer = None
            self.is_initialized = None
        else:
            self.is_initialized[dones] = False

    def init_if_needed(self, x):
        B = x.shape[0]
        feat_shape = x.shape[1:]

        need_reinit = (
            self.buffer is None
            or self.buffer.shape[0] != B
            or self.buffer.shape[2:] != feat_shape
        )

        if need_reinit:
            self.buffer = torch.zeros(
                B, self.history_len, *feat_shape,
                device=x.device,
                dtype=x.dtype,
            )
            self.is_initialized = torch.zeros(
                B, device=x.device, dtype=torch.bool
            )

    @torch.no_grad()
    def append(self, x: torch.Tensor):
        """
        x:
            [B, D]        -> output [B, H, D]
            [B, H2, D]    -> output [B, H, H2, D]
        """
        self.init_if_needed(x)

        new_ids = ~self.is_initialized
        if new_ids.any():
            self.buffer[new_ids] = x[new_ids].unsqueeze(1).repeat(
                1, self.history_len, *([1] * (x.dim() - 1))
            )
            self.is_initialized[new_ids] = True

        old_ids = self.is_initialized & (~new_ids)
        if old_ids.any():
            self.buffer[old_ids] = torch.roll(self.buffer[old_ids], shifts=-1, dims=1)
            self.buffer[old_ids, -1] = x[old_ids]

        return self.buffer