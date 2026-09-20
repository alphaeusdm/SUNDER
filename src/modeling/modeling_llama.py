from typing import Callable, Optional, Union

import torch
from torch import nn

torch.set_printoptions(threshold=torch.inf)  # no truncation


from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

# for dataclass
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore")


logger = logging.get_logger(__name__)

def get_selective_causal_mask(causal_mask: torch.Tensor, start_idx: int, end_idx: int):
    causal_mask_unmasked = causal_mask
    # for i in range(causal_mask.shape[0]):
    #     for start, end in zip(start_idx[i], end_idx[i]):
    #         causal_mask[i, :, start:end, :end] = 0
    for start, end in zip(start_idx, end_idx):
        causal_mask[:, :, start:end, :end] = 0
    return causal_mask_unmasked


# code copied from transformers.src.transformers.models.llama.modeling_llama.py
@auto_docstring
class SelectiveUnmaskingLlamaModel(LlamaPreTrainedModel):
    """
    Performs Selective unmasking for the base Llama Model
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_unmask: Optional[torch.Tensor] = None, # positions of inputs to be unmasked
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # attention_unmask = attention_unmask.squeeze(0)
        # diff = attention_unmask.diff(prepend=torch.tensor([0], device=attention_unmask.device))
        # start_idx = torch.where(diff==1)[0]
        # end_idx = torch.where(diff==-1)[0]

        # if attention_unmask[-1] == 1:
        #     end_idx = torch.cat([end_idx, torch.tensor([len(attention_unmask)], device=end_idx.device)])

        # start_idx = []
        # end_idx = []

        # for row in attention_unmask:
        #     diff = row.diff(prepend=torch.tensor([0], device=row.device))

        #     # Find start and end indices for this row
        #     row_start = torch.where(diff == 1)[0]
        #     row_end = torch.where(diff == -1)[0]
            
        #     # Handle case where last element is 1
        #     if row[-1] == 1:
        #         row_end = torch.cat([row_end, torch.tensor([len(row)], device=row_end.device)])
            
        #     start_idx.append(row_start)
        #     end_idx.append(row_end)

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # causal_mask_unmasked = None

        # if start_idx is not None:
        #     causal_mask_unmasked = get_selective_causal_mask(causal_mask, start_idx, end_idx)

        if attention_unmask.dim() == 1:
            attention_unmask = attention_unmask.unsqueeze(0)

        prepend = torch.zeros(*attention_unmask.shape[:-1], 1, dtype=attention_unmask.dtype, device=attention_unmask.device)
        diff = attention_unmask.diff(dim=-1, prepend=prepend)

        group_ids = (diff == 1).cumsum(-1) * attention_unmask  # (batch, seq_len)
        same_group = (group_ids[:, :, None] == group_ids[:, None, :]) & (group_ids[:, :, None] > 0)  # (batch, seq_len, seq_len)

        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_len > 0:
            q_len  = causal_mask.shape[2]
            kv_len = causal_mask.shape[3]

            same_group = same_group[:, -q_len:, :]

            if same_group.shape[-1] < kv_len:
                pad = torch.zeros(
                    same_group.shape[0], q_len, kv_len - same_group.shape[-1],
                    dtype=same_group.dtype, device=same_group.device
                )
                same_group = torch.cat([pad, same_group], dim=-1)

        same_group = same_group.unsqueeze(1)  # (batch, 1, seq_len, seq_len)

        causal_mask_unmasked = causal_mask.masked_fill(same_group, 0)
        
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_num, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_unmasked,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

@auto_docstring
class SelectiveUnmaskingLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = SelectiveUnmaskingLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_unmask: Optional[torch.Tensor] = None, # positions of inputs to be unmasked
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_unmask=attention_unmask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )