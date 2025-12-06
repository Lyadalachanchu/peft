from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType, TaskType

@dataclass
class KVPromptConfig(PeftConfig):
    """
    Configuration for KV-prompt adapters.

    This method injects learned ΔK/ΔV vectors into the last prompt position's
    key/value tensors at each transformer layer, without changing sequence length.

    Args:
        task_type:
            The downstream task type (CAUSAL_LM, SEQ_2_SEQ_LM, SEQ_CLS, ...).

        target_layers:
            Indices of decoder layers that should get KVPrompt adapters.
            If None, all layers are adapted.

        affect_keys:
            Whether to learn ΔK (in key space).

        affect_values:
            Whether to learn ΔV (in value space).

        init_std:
            Stddev for normal initialization of ΔK/ΔV when not zero-initialized.
    """

    target_layers: Optional[List[int]] = None
    affect_keys: bool = True
    affect_values: bool = True

    def __post_init__(self):
        if self.peft_type is None:
            self.peft_type = PeftType.KV_PROMPT

        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)

        # Sanity checks
        if not self.affect_keys and not self.affect_values:
            raise ValueError(
                "KVPromptConfig: at least one of `affect_keys` or `affect_values` must be True."
            )

        # Let parent perform its own checks (optional, depending on version)
        super().__post_init__() if hasattr(super(), "__post_init__") else None

