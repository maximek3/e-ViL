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
from transformers import GPT2Tokenizer


class VCRGpt2Tokenizer(GPT2Tokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        begin_img="<|b_img|>",
        end_img="<|e_img|>",
        begin_question="<|b_qn|>",
        end_question="<|e_qn|>",
        begin_rationale="<|b_rtnl|>",
        end_rationale="<|e_rtnl|>",
        begin_answer="<|b_ans|>",
        end_answer="<|e_ans|>",
        begin_situation="<|b_situ|>",
        end_situation="<|e_situ|>",
        begin_verb="<|b_verb|>",
        end_verb="<|e_verb|>",
        begin_before="<|b_before|>",
        end_before="<|e_before|>",
        begin_after="<|b_after|>",
        end_after="<|e_after|>",
        begin_intent="<|b_intent|>",
        end_intent="<|e_intent|>",
        begin_viscomet_token="<|b_viscomet|>",
        end_viscomet_token="<|e_viscomet|>",
        **kwargs
    ):
        super(VCRGpt2Tokenizer, self).__init__(
            vocab_file,
            merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs
        )

        self.begin_img = begin_img
        self.end_img = end_img
        self.begin_question = begin_question
        self.end_question = end_question
        self.begin_rationale = begin_rationale
        self.end_rationale = end_rationale
        self.begin_answer = begin_answer
        self.end_answer = end_answer
        self.begin_situation = begin_situation
        self.end_situation = end_situation
        self.begin_verb = begin_verb
        self.end_verb = end_verb
        self.begin_before = begin_before
        self.end_before = end_before
        self.begin_after = begin_after
        self.end_after = end_after
        self.begin_intent = begin_intent
        self.end_intent = end_intent

        self.begin_viscomet_token = begin_viscomet_token
        self.end_viscomet_token = end_viscomet_token
        self.det_tokens = ["<|det%d|>" % i for i in range(45)]

        special_tokens = [
            self.begin_question,
            self.end_question,
            self.begin_answer,
            self.end_answer,
            self.begin_rationale,
            self.end_rationale,
            self.begin_img,
            self.end_img,
            self.begin_situation,
            self.end_situation,
            self.begin_verb,
            self.end_verb,
            self.begin_before,
            self.end_before,
            self.begin_after,
            self.end_after,
            self.begin_intent,
            self.end_intent,
        ] + self.det_tokens

        self.add_special_tokens({"additional_special_tokens": special_tokens})

        # self.max_len_single_sentence = self.max_len - len(special_tokens)

    def decode(
        self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
    ):
        text = super().decode(token_ids, False, clean_up_tokenization_spaces)
        tokens2remove = [
            self.begin_question,
            self.end_question,
            self.begin_answer,
            self.end_answer,
            self.begin_situation,
            self.end_situation,
            self.begin_rationale,
            self.unk_token,
        ]
        if skip_special_tokens:
            for t in tokens2remove:
                text = text.replace(t, " ")
            for i in range(45):
                text = text.replace("<|det" + str(i) + "|>", str(i))

        idx = text.find(self.end_rationale)
        if idx != -1:
            text = text[:idx]
        return text.strip()
