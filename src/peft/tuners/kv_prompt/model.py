from peft.tuners.tuners_utils import BaseTuner


class KVPromptModel(BaseTuner):
    """
    Placeholder tuner for KV-prompt adapters.

    The actual injection logic still needs to be implemented.
    """

    prefix = "kv_prompt_"
    tuner_layer_cls = None
    target_module_mapping = {}

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("KVPromptModel is not implemented yet.")
