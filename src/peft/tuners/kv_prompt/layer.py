from __future__ import annotations
import torch
import torch.nn as nn

class KVPromptAdapter(nn.Module):
    """
    Per-layer KV prompt adapter.

    It owns ΔK and ΔV (optionally one or both), shaped (num_kv_heads, head_dim),
    and adds them to the LAST prompt position's key/value for that layer.

    We assume key_states, value_states have shape (B, H_kv, S, d).
    """

    def __init__(
        self, 
        num_kv_heads: int,
        head_dim: int,
        affect_keys: bool = True,
        affect_values: bool = True,
    ):
        super().__init__()

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.affect_keys = affect_keys
        self.affect_values = affect_values

        if affect_keys:
            self.delta_k = nn.Parameter(torch.zeros(num_kv_heads, head_dim))
        else:
            self.register_parameter("delta_k", None)

        if affect_values:
            self.delta_v = nn.Parameter(torch.zeros(num_kv_heads, head_dim))
        else:
            self.register_parameter("delta_v", None)

    def forward(
        self, 
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        prompt_length: int | None = None
    ):
        """
        key_states, value_states: (B, H_kv, S, d)
        prompt_length: length of the prompt (S_p). If None, use full S.
        """
        B, H, S, d = key_states.shape
        assert H == self.num_kv_heads and d == self.head_dim

        if prompt_length is None:
            idx = S - 1
        else:
            idx = prompt_length - 1

        if self.delta_k is not None:
            dk = self.delta_k[None, :, None, :]  # (1, H, 1, d)
            key_states[:, :, idx, :] = key_states[:, :, idx, :] + dk

        if self.delta_v is not None:
            dv = self.delta_v[None, :, None, :]
            value_states[:, :, idx, :] = value_states[:, :, idx, :] + dv

        return key_states, value_states
