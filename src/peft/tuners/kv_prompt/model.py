from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from .config import KVPromptConfig
from .layer import KVPromptAdapter
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
)


class KVPromptWrappedAttention(nn.Module, BaseTunerLayer):
    """
    Wrap a decoder self-attention module and inject ΔK/ΔV at the last prompt
    position on the *prompt pass* (when past_key_value is None).
    """

    adapter_layer_names = ("kv_prompt_adapters",)

    def __init__(self, base_attn: nn.Module, adapter_name: str, **adapter_kwargs):
        super().__init__()
        self.base_layer = base_attn
        self.kv_prompt_adapters = nn.ModuleDict({})
        self._active_adapter = adapter_name
        self._disable_adapters = False
        self.merged_adapters: list[str] = []

        self.update_layer(adapter_name, **adapter_kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def update_layer(
        self,
        adapter_name: str,
        num_kv_heads: int,
        head_dim: int,
        affect_keys: bool,
        affect_values: bool,
        inference_mode: bool = False,
        **kwargs,
    ):
        adapter = KVPromptAdapter(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            affect_keys=affect_keys,
            affect_values=affect_values,
        )
        self.kv_prompt_adapters[adapter_name] = adapter
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("KV prompt adapters cannot be merged into the base model.")

    def unmerge(self) -> None:
        raise NotImplementedError("KV prompt adapters do not support merging/unmerging.")

    def _apply_adapters(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        prompt_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for adapter_name in self.active_adapters:
            adapter = self.kv_prompt_adapters[adapter_name] if adapter_name in self.kv_prompt_adapters else None
            if adapter is None:
                raise ValueError(f"Adapter '{adapter_name}' not found in kv_prompt_adapters.")
            key_states, value_states = adapter(
                key_states,
                value_states,
                prompt_length=prompt_length,
            )
        return key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        if self.disable_adapters or not self.active_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        # ----------- ORIGINAL QWEN3 CODE -----------
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # project Q, K, V
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)

        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)

        value_states = (
            self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )

        # apply rotary
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # ----------- OUR INSERTION POINT -----------
        if past_key_values is None and self.active_adapters:
            # prompt_length = full sequence length
            prompt_length = key_states.shape[2]
            key_states, value_states = self._apply_adapters(key_states, value_states, prompt_length)

        # ----------- CACHING LOGIC -----------
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # ----------- ATTENTION (unchanged) -----------
        if self.config._attn_implementation != "eager":
            attn_fn = self.config._attn_implementation
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_fn]
        else:
            from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

            attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class KVPromptModel(BaseTuner):
    """
    BaseTuner implementation for KV prompt adapters.

    This:
      - freezes the base model,
      - attaches a KVPromptAdapter to selected decoder layers,
      - wraps each layer's self-attention to call the adapter.
    """

    prefix: str = "kv_prompt_"
    tuner_layer_cls = KVPromptWrappedAttention

    def _prepare_adapter_config(self, peft_config: KVPromptConfig, model_config: dict) -> KVPromptConfig:
        if peft_config.target_modules is None:
            peft_config.target_modules = ["self_attn"]
        return peft_config

    @staticmethod
    def _extract_layer_idx(module_key: str) -> Optional[int]:
        parts = module_key.split(".")
        for idx, name in enumerate(parts[:-1]):
            if name == "layers" and parts[idx + 1].isdigit():
                return int(parts[idx + 1])
        return None

    def _should_adapt_layer(self, peft_config: KVPromptConfig, module_key: str) -> bool:
        if peft_config.target_layers is None:
            return True
        layer_idx = self._extract_layer_idx(module_key)
        if layer_idx is None:
            return False
        return layer_idx in peft_config.target_layers

    @staticmethod
    def _infer_attention_metadata(target: nn.Module) -> tuple[Optional[int], Optional[int]]:
        num_kv_heads = getattr(target, "num_key_value_heads", None)
        head_dim = getattr(target, "head_dim", None)

        config = getattr(target, "config", None)
        num_heads = getattr(target, "num_heads", None) or getattr(config, "num_attention_heads", None)
        config_num_kv = getattr(config, "num_key_value_heads", None) if config is not None else None

        if num_kv_heads is None and config_num_kv is not None:
            num_kv_heads = config_num_kv
        if num_kv_heads is None:
            num_kv_heads = num_heads

        if head_dim is None and config is not None:
            head_dim = getattr(config, "head_dim", None)
        if head_dim is None and config is not None:
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is not None and num_heads:
                head_dim = hidden_size // num_heads

        q_proj = getattr(target, "q_proj", None)
        if head_dim is None and q_proj is not None:
            out_features = getattr(q_proj, "out_features", None)
            if out_features is not None and num_heads:
                head_dim = out_features // num_heads

        k_proj = getattr(target, "k_proj", None)
        if head_dim is None and k_proj is not None:
            out_features = getattr(k_proj, "out_features", None)
            candidate_heads = num_kv_heads or num_heads
            if out_features is not None and candidate_heads:
                head_dim = out_features // candidate_heads

        if num_kv_heads is None and head_dim is not None and k_proj is not None:
            out_features = getattr(k_proj, "out_features", None)
            if out_features is not None and head_dim != 0:
                num_kv_heads = out_features // head_dim

        return num_kv_heads, head_dim

    def _create_and_replace(
        self,
        peft_config: KVPromptConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **kwargs,
    ) -> None:
        if not self._should_adapt_layer(peft_config, current_key):
            if self.targeted_module_names:
                self.targeted_module_names.pop()
            return

        num_kv_heads, head_dim = self._infer_attention_metadata(target)

        if num_kv_heads is None or head_dim is None:
            raise ValueError(
                f"KVPromptModel: could not infer (num_kv_heads, head_dim) for module '{current_key}'."
            )

        adapter_kwargs = dict(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            affect_keys=peft_config.affect_keys,
            affect_values=peft_config.affect_values,
        )

        if isinstance(target, KVPromptWrappedAttention):
            target.update_layer(adapter_name, **adapter_kwargs, inference_mode=peft_config.inference_mode)
        else:
            new_module = KVPromptWrappedAttention(target, adapter_name, **adapter_kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
