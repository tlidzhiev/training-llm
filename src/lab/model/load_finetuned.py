import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

BASE_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'


def load_for_eval(
    model: str, device: str | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a model for evaluation: the base model, a full fine-tuned checkpoint or a LoRA adapter.

    Parameters
    ----------
    model : str
        'base' (Qwen2.5-1.5B-Instruct), a Hugging Face id, or a checkpoint folder saved by
        `lab.finetune` (a full model, or a LoRA adapter with adapter_config.json).
    device : str | None
        'cuda' or 'cpu'; None picks cuda when available.

    Returns
    -------
    tuple[PreTrainedModel, PreTrainedTokenizerBase]
        The model in eval mode (bf16 on GPU, fp32 on CPU) and its tokenizer.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    path = BASE_MODEL if model == 'base' else model

    adapter_config = Path(path) / 'adapter_config.json'
    if adapter_config.exists():
        # The adapter is kept unmerged, as during training and the test evaluation.
        base_name = json.loads(adapter_config.read_text())['base_model_name_or_path']
        base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
        net = PeftModel.from_pretrained(base, path)
    else:
        net = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(path)
    return net.to(device).eval(), tokenizer  # ty:ignore[invalid-argument-type, invalid-return-type]
