from typing import ClassVar

from transformers import PretrainedConfig


class Llama3Config(PretrainedConfig):
    model_type = 'lab_llama3'
    keys_to_ignore_at_inference: ClassVar[list[str]] = ['past_key_values']

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        max_position_embeddings: int = 1024,
        rope_theta: float = 500000.0,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        pad_token_id: int | None = None,
        **kwargs,
    ) -> None:
        if head_dim is None:
            if hidden_size % num_attention_heads != 0:
                raise ValueError(
                    f'hidden_size ({hidden_size}) must be divisible by '
                    f'num_attention_heads ({num_attention_heads}) when head_dim is not set'
                )
            head_dim = hidden_size // num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(f'head_dim must be even for rotary embeddings, got {head_dim}')
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f'num_attention_heads ({num_attention_heads}) must be divisible by '
                f'num_key_value_heads ({num_key_value_heads})'
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,  # ty:ignore[unknown-argument]
            eos_token_id=eos_token_id,  # ty:ignore[unknown-argument]
            pad_token_id=pad_token_id,  # ty:ignore[unknown-argument]
            tie_word_embeddings=tie_word_embeddings,  # ty:ignore[unknown-argument]
            **kwargs,
        )
