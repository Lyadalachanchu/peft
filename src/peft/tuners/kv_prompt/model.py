from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from peft.peft_model import PeftModel
from peft.utils import PeftType
from .config import KVPromptConfig
from .layer import KVPromptAdapter
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


class KVPromptWrappedAttention(nn.Module):
    """
    Wrap a decoder self-attention module and inject ΔK/ΔV at the last prompt
    position on the *prompt pass* (when past_key_value is None).
    """
    def __init__(self, base_attn: nn.Module, kv_adapter: KVPromptAdapter):
        super().__init__()
        self.base_attn = base_attn
        self.kv_adapter = kv_adapter

        # copy all attributes that the rest of the model expects
        for name, value in base_attn.__dict__.items():
            if name.startswith("_") or name in ("base_attn", "kv_adapter"):
                continue
            setattr(self, name, value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
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
        if past_key_values is None:
            # prompt_length = full sequence length
            prompt_length = key_states.shape[2]
            key_states, value_states = self.kv_adapter(
                key_states,
                value_states,
                prompt_length=prompt_length,
            )

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



class KVPromptModel(PeftModel):
    """
    PEFT model for KV-prompt tuning.

    This:
      - freezes the base model,
      - attaches a KVPromptAdapter to selected decoder layers,
      - wraps each layer's self-attention to call the adapter.
    """

    def __init__(self, model: nn.Module, peft_config: KVPromptConfig, adapter_name: str = "default"):
        # Note to myself: PeftModel takes (model, peft_config, adapter_name)
        super().__init__(model, peft_config, adapter_name)
        self.peft_type = PeftType.KV_PROMPT
        self._prepare_kv_prompt(adapter_name)

    def _freeze_base_model(self):
        for p in self.get_base_model().parameters():
            p.requires_grad = False

    def _prepare_kv_prompt(self, adapter_name: str):
        peft_config: KVPromptConfig = self.peft_config[adapter_name]

        self._freeze_base_model()

        base = self.get_base_model()

        # For Qwen/LLaMA-style models: base.model.layers is a list of decoder blocks (from chat)
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            decoder_layers = base.model.layers
        elif hasattr(base, "layers"):
            decoder_layers = base.layers
        else:
            raise ValueError(
                "KVPromptModel: could not find decoder layers. "
                "Extend _prepare_kv_prompt for your model architecture."
            )

        num_layers = len(decoder_layers)

        if peft.config.target_layers is None:
            target_layers = list(range(num_layers))
        else:
            target_layers = peft_config.target_layers

        # create adapters and wrap attention in each target layer
        for layer_idx, layer in enumerate(decoder_layers):
            if layer_idx not in target_layers:
                continue

            if not hasattr(layer, "self_attn"):
                raise ValueError(
                    f"KVPromptModel: layer {layer_idx} has no `self_attn` attribute."
                )

            attn = layer.self_attn
            # for Qwen/LLaMA-style, attention module exposes num_key_value_heads / head_dim (chat)
            num_kv_heads = getattr(attn, "num_key_value_heads", getattr(attn, "num_heads", None))
            head_dim = getattr(attn, "head_dim", None)

            if num_kv_heads is None or head_dim is None:
                raise ValueError(
                    f"KVPromptModel: could not infer (num_kv_heads, head_dim) for layer {layer_idx}."
                )

            kv_adapter = KVPromptAdapter(
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    affect_keys=peft_config.affect_keys,
                    affect_values=peft_config.affect_values,
                )

            layer.kv_prompt_adapter = kv_adapter

            # wrap attention
            # TODO: implement KVPromptWrappedAttention
            wrapped_attn = KVPromptWrappedAttention(attn, kv_adapter)
            layer.self_attn = wrapped_attn

            #mark only KVPrompt params as trainable
            for name, param in self.named_parameters():
                if "kv_prompt_adapter" in name:
                    param.requires_grad = True
                else:
                    # Do not unfreeze any other PEFT parameters; base mode is frozen
                    param.requires_grad = getattr(param, "requires_grad", False)



