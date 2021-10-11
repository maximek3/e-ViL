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
This script applies a pre-trained model to generat explanations on a selected set of samples. Adapted from Marasovic et al. 2020 https://github.com/allenai/visual-reasoning-rationalization.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def generate_text(
    decoder,
    tokenizer,
    input_ids,
    token_type_idss,
    visual_representations,
    max_rationale_length=30,
    num_samples=1,
    top_k=1,
    top_p=0,
):
    """
    Adopted from Marasovic et al. 2020 https://arxiv.org/abs/2010.07526
    https://github.com/allenai/visual-reasoning-rationalization
    """

    sample_texts = []
    for idx, input_ids in enumerate(input_ids):

        input_ids_copy = input_ids.unsqueeze(0)
        token_type_ids_copy = token_type_idss[idx].unsqueeze(0)
        visual_representations_copy = visual_representations[idx].unsqueeze(0)

        prompt_token_idx = tokenizer.convert_tokens_to_ids([tokenizer.begin_rationale])[
            0
        ]
        context = input_ids_copy
        idx_of_prompt_token = (context == prompt_token_idx).nonzero()[0][1].item()
        context = context[:, : idx_of_prompt_token + 1]
        context = context.repeat(num_samples, 1)
        generated = context

        token_type_ids = token_type_ids_copy[:, : idx_of_prompt_token + 1]
        token_type_ids = token_type_ids.repeat(num_samples, 1)
        rationale_type_idx = torch.max(token_type_ids, dim=1)[0].unsqueeze(0)

        for tok_idx in range(max_rationale_length):

            outputs = decoder(
                generated,
                token_type_ids=token_type_ids,
                visual_representations=visual_representations_copy,
            )

            next_token_logits = outputs[0][0, -1, :]
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=1, top_p=0)
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1), num_samples=num_samples
            )
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            token_type_ids = torch.cat((token_type_ids, rationale_type_idx), dim=1)

        idx_of_prompt_token = (
            (input_ids_copy == prompt_token_idx).nonzero()[0][1].item()
        )
        instance_generations_indices = generated[0, idx_of_prompt_token + 1 :].tolist()
        sample_text = tokenizer.decode(
            instance_generations_indices,
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )
        sample_texts.append(sample_text)

    return sample_texts
