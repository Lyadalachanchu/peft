from peft.utils import register_peft_method

from .config import KVPromptConfig
from .model import KVPromptModel


__all__ = ["KVPromptConfig", "KVPromptModel"]

register_peft_method(
    name="kv_prompt",
    config_cls=KVPromptConfig,
    model_cls=KVPromptModel,
)
