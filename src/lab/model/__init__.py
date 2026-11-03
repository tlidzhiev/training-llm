import math

from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

from .llama3 import Llama3Config, Llama3ForCausalLM

# name in the Hydra config -> (config class, model class)
MODELS: dict[str, tuple[type[PretrainedConfig], type[PreTrainedModel]]] = {
    'llama3': (Llama3Config, Llama3ForCausalLM),
}


def get_model(
    name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    pad_vocab_to_multiple_of: int = 64,
    **config_kwargs,
) -> PreTrainedModel:
    """
    Build a randomly initialized causal LM from the registry.

    The vocabulary size, the special tokens and the maximum length are taken from the
    tokenizer and the data, so they always match the training windows.

    Parameters
    ----------
    name : str
        Key of MODELS, e.g. 'llama3'.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer of the training data.
    max_seq_len : int
        Length of the training windows.
    pad_vocab_to_multiple_of : int
        Round the vocabulary size up to this multiple: the LM head matmul is faster on
        GPU when its size is a multiple of 64.
    **config_kwargs
        Other arguments of the model config (hidden_size, num_hidden_layers, ...).

    Returns
    -------
    PreTrainedModel
        The model.
    """
    if name not in MODELS:
        raise ValueError(f'Unknown model {name!r}. Supported models {list(MODELS)}')
    config_cls, model_cls = MODELS[name]
    config_kwargs.update(
        vocab_size=math.ceil(len(tokenizer) / pad_vocab_to_multiple_of) * pad_vocab_to_multiple_of,
        max_position_embeddings=max_seq_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return model_cls(config_cls(**config_kwargs))


__all__ = ['get_model']
