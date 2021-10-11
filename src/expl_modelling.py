# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.3
# Copyright 2020 https://github.com/allenai/visual-reasoning-rationalization
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is copied and adapted from Marasovic et al. 2020 (https://arxiv.org/abs/2010.07526)
https://github.com/allenai/visual-reasoning-rationalization
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel  # class to handle weights initialization
from transformers.modeling_bert import BertLayerNorm as LayerNorm
from transformers.modeling_gpt2 import Block


class GPT2VisionAttentiveTransformer(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2VisionAttentiveTransformer, self).__init__(config)
        # config contains exact model configurations
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        """
        Wrapper around PreTrainedModel._get_resized_embeddings

        Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.

        New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end.
        """
        self.wte = self._get_resized_embeddings(self.wte, new_num_tokens)
        return self.wte

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids,
        visual_representations=None,
        box_coordinates=None,
        orig_num_situ_or_object_boxes=None,
        position_ids=None,
        token_type_ids=None,
        past=None,
        head_mask=None,
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_ids.size(-1) + past_length,
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        token_type_embeds = 0
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)

        if visual_representations is None and box_coordinates is None:
            hidden_states = inputs_embeds + position_embeds + token_type_embeds

        if visual_representations is not None:
            inputs_embeds[
                :, 1 : visual_representations.shape[1] + 1, :
            ] = visual_representations
            hidden_states = inputs_embeds + position_embeds + token_type_embeds

        hidden_states = self.drop(hidden_states)  # dropout

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    hidden_states.view(*output_shape),
                )

            outputs = block(hidden_states, layer_past, head_mask[i])
            hidden_states, present = outputs[:2]
            presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = (
                input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            )
            all_attentions = tuple(
                t.view(*attention_output_shape) for t in all_attentions
            )
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, presents, (all hidden_states), (attentions)


class GPT2VisionAttentiveLMHead(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2VisionAttentiveLMHead, self).__init__(config)
        self.transformer = GPT2VisionAttentiveTransformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte)

    def forward(
        self,
        input_ids,
        labels=None,
        visual_representations=None,
        box_coordinates=None,
        orig_num_situ_or_object_boxes=None,
        position_ids=None,
        token_type_ids=None,
        past=None,
        head_mask=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            visual_representations=visual_representations,
            box_coordinates=box_coordinates,
            orig_num_situ_or_object_boxes=orig_num_situ_or_object_boxes,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past=past,
            head_mask=head_mask,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    def _resize_token_embeddings(self, new_num_tokens):
        self.transformer.resize_token_embeddings(new_num_tokens)
