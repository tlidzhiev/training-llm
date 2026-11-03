import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers import __version__ as transformers_version
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_llama3 import Llama3Config

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:  # transformers < 4.52
    GradientCheckpointingLayer = nn.Module

TRANSFORMERS_V5 = int(transformers_version.split('.')[0]) >= 5


def rotary_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    exponents = torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float)
    inv_freq = 1.0 / (theta ** (exponents / head_dim))
    freqs = position_ids[..., None].float() * inv_freq
    angles = repeat(freqs, 'b s d -> b 1 s (two d)', two=2)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = rearrange(x, '... (two d) -> two ... d', two=2)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def build_attention_mask(
    attention_mask: torch.Tensor | None,
    q_len: int,
    past_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attention_mask is not None and attention_mask.dim() != 2:
        raise ValueError('attention_mask must have shape (batch, past + seq)')
    no_padding = attention_mask is None or bool(attention_mask.all())
    if no_padding and (past_len == 0 or q_len == 1):
        return None

    kv_len = past_len + q_len
    q_pos = rearrange(torch.arange(past_len, kv_len, device=device), 'q -> q 1')
    k_pos = rearrange(torch.arange(kv_len, device=device), 'k -> 1 k')
    mask = rearrange(k_pos <= q_pos, 'q k -> 1 1 q k')
    if attention_mask is not None:
        mask = mask & rearrange(attention_mask[:, :kv_len].bool(), 'b k -> b 1 1 k')
        # Padding queries would have no key to attend to and SDPA would return NaN,
        # which leaks into other positions through 0 * NaN; let them see themselves.
        mask = mask | rearrange(k_pos == q_pos, 'q k -> 1 1 q k')
    return mask


def make_cache(config: PretrainedConfig) -> DynamicCache:
    try:
        return DynamicCache(config=config)
    except TypeError:  # transformers < 4.56
        return DynamicCache()


class Llama3Attention(nn.Module):
    def __init__(self, config: Llama3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        qkv_heads = self.num_heads + 2 * self.num_kv_heads
        self.qkv_proj = nn.Linear(config.hidden_size, qkv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        qkv = rearrange(self.qkv_proj(hidden_states), 'b s (h d) -> b h s d', d=self.head_dim)
        query, key, value = qkv.split([self.num_heads, self.num_kv_heads, self.num_kv_heads], 1)

        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        groups = self.num_heads // self.num_kv_heads
        key = repeat(key, 'b h s d -> b (h g) s d', g=groups)
        value = repeat(value, 'b h s d -> b (h g) s d', g=groups)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=attention_mask is None and query.shape[2] > 1,
        )
        return self.o_proj(rearrange(output, 'b h s d -> b s (h d)'))


class Llama3MLP(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Llama3DecoderLayer(GradientCheckpointingLayer):  # ty:ignore[unsupported-base]
    def __init__(self, config: Llama3Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Llama3Attention(config, layer_idx)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Llama3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            attention_mask,
            past_key_values,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Llama3PreTrainedModel(PreTrainedModel):
    config_class = Llama3Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = GradientCheckpointingLayer is not nn.Module
    _no_split_modules = ['Llama3DecoderLayer']  # noqa: RUF012
    _skip_keys_device_placement = ['past_key_values']  # noqa: RUF012
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.RMSNorm) and module.weight is not None:
            nn.init.ones_(module.weight)


class Llama3Model(Llama3PreTrainedModel):
    def __init__(self, config: Llama3Config):
        super().__init__(config)
        # No padding_idx: tokenizers often use EOS as padding, and its embedding must train.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Llama3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('Specify exactly one of input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = self.config.use_cache and not self.training
        if use_cache and past_key_values is None:
            past_key_values = make_cache(self.config)

        q_len = inputs_embeds.shape[1]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            positions = torch.arange(past_len, past_len + q_len, device=inputs_embeds.device)
            position_ids = rearrange(positions, 's -> 1 s')

        mask = build_attention_mask(attention_mask, q_len, past_len, inputs_embeds.device)
        cos, sin = rotary_cos_sin(
            position_ids, self.config.head_dim, self.config.rope_theta, inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        all_hidden_states = []
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(
                hidden_states, cos, sin, attention_mask=mask, past_key_values=past_key_values
            )
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,  # ty:ignore[invalid-argument-type]
        )


class Llama3ForCausalLM(Llama3PreTrainedModel, GenerationMixin):
    # transformers 5 maps tied parameters to their source, older versions list them.
    _tied_weights_keys = (  # ty:ignore[invalid-assignment]
        {'lm_head.weight': 'model.embed_tokens.weight'} if TRANSFORMERS_V5 else ['lm_head.weight']
    )
    # Trainer passes num_items_in_batch, so the loss is normalized over the whole
    # accumulated batch.
    accepts_loss_kwargs = True

    def __init__(self, config: Llama3Config):
        super().__init__(config)
        self.model = Llama3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        num_items_in_batch: torch.Tensor | int | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast | tuple:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int):
            if logits_to_keep > 0:
                hidden_states = hidden_states[:, -logits_to_keep:]
        else:
            hidden_states = hidden_states[:, logits_to_keep]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                num_items_in_batch=num_items_in_batch,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        return output if return_dict else output.to_tuple()
