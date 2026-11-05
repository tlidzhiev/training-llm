from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel


def get_pretrained_model(
    name_or_path: str,
    dtype: str = 'float32',
    attn_implementation: str = 'sdpa',
    lora: dict[str, Any] | None = None,
) -> PreTrainedModel:
    """
    Load a pretrained causal LM; wrap it with LoRA adapters if `lora` is given.

    Parameters
    ----------
    name_or_path : str
        Hugging Face id or a local path.
    dtype : str
        Name of the torch dtype of the weights ('float32', 'bfloat16').
    attn_implementation : str
        Attention implementation ('sdpa', 'eager', 'flash_attention_2').
    lora : dict | None
        Arguments of peft.LoraConfig (r, lora_alpha, lora_dropout, target_modules, ...).
        None means full fine-tuning.
    """
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, dtype=getattr(torch, dtype), attn_implementation=attn_implementation
    )
    if lora is not None:
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, **lora))
    return model  # ty:ignore[invalid-return-type]
