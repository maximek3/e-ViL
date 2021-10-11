# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 https://github.com/YIKUAN8/Transformers-VQA
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

import torch
import torch.nn as nn

from src.expl_tokenization import VCRGpt2Tokenizer
from src.expl_modelling import GPT2VisionAttentiveLMHead

from src.tokenization import BertTokenizer
from src.modeling import VISUAL_CONFIG
from src.modeling import UniterFeatureExtraction as UFE

from param import args

device = torch.device(args.device)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[: (max_seq_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

        if "[SEP]" in tokens_a:  # vcr training
            qa_cutoff = tokens.index("[SEP]") + 1
            segment_ids = [0] * qa_cutoff + [2] * (len(tokens) - qa_cutoff)
        else:
            segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
            )
        )
    return features


def _tokenize_append_truncate(tokenizer, text, max_length, end_token):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[: max_length - 1] + [end_token]
    elif len(tokens) < max_length:
        diff = max_length - len(tokens)
        tokens = tokens[: len(tokens) - 1]
        tokens.extend([tokenizer.unk_token] * diff)
        tokens.append(end_token)
    return tokens


def _tokenize_truncate(tokenizer, text, max_length, end_token):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[: max_length - 1] + [end_token]
    return tokens


def tokenize(tokenizer, text, max_length, end_token, do_padding):
    if do_padding:
        tokens = _tokenize_append_truncate(tokenizer, text, max_length, end_token)
    else:
        tokens = _tokenize_truncate(tokenizer, text, max_length, end_token)
    return tokens


def truncate(
    tokens,
    max_length,
    end_token,
    unk_token,
    token_type_ids=None,
    sequential_positions=None,
):
    if len(tokens) > max_length:
        tokens = tokens[: max_length - 1] + [end_token]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:max_length]
        if sequential_positions is not None:
            sequential_positions = sequential_positions[:max_length]
    else:
        diff = max_length - len(tokens)
        tokens.extend([unk_token] * diff)
        if token_type_ids is not None:
            final_segment_id = max(token_type_ids)
            token_type_ids.extend([final_segment_id] * diff)
        if sequential_positions is not None:
            sequential_positions.extend(list(range(max_length - diff, max_length)))
            assert len(tokens) == len(token_type_ids) == len(sequential_positions)
    return tokens, token_type_ids, sequential_positions


def get_labels(tokenizer: VCRGpt2Tokenizer, tokenized_text):
    try:
        rationale_start_token_idx = tokenizer.convert_tokens_to_ids(
            [tokenizer.begin_rationale]
        )[0]
        start_idx = tokenized_text.index(rationale_start_token_idx)
        rationale_end_token_idx = tokenizer.convert_tokens_to_ids(
            [tokenizer.end_rationale]
        )[0]
        end_idx = tokenized_text.index(rationale_end_token_idx)

        labels = [-1] * len(tokenized_text)
        labels[start_idx + 1 : end_idx + 1] = tokenized_text[
            start_idx + 1 : end_idx + 1
        ]

        assert len(tokenized_text) == len(labels)
        return labels
    except ValueError:
        # import pdb; pdb.set_trace()
        raise Exception("Failed to tokenize: {}".format(tokenized_text))


def preprocess_gpt2(
    tokenizer,
    questions,
    rationales,
    logit,
    uniter_dim,
    label_dict,
    do_padding=False,
    max_rationale_length=51,
    max_question_length=19,
):
    """
    Transforms text in required format for GPT-2.
    """

    answers = []
    score, label = logit.max(1)

    # get textual representation of answer
    if isinstance(label_dict[0], list):  # vcr
        for idx, l in enumerate(label.cpu().numpy()):
            answers.append(label_dict[idx][l])
        max_answer_length = 23
    else:  # esnlive and vqa_x
        for l in label.cpu().numpy():
            answers.append(label_dict[l])
        max_answer_length = 23

    block_size = (
        max_rationale_length + max_question_length + max_answer_length + uniter_dim + 1
    )

    inputs, token_type_ids_list, label_list = [], [], []

    for question, answer, rationale in zip(questions, answers, rationales):

        uniter_extended = " ".join(
            [tokenizer.begin_img, (uniter_dim + 1) * "u ", tokenizer.end_img]
        )  # dummy placeholders
        uniter_tokens = tokenize(
            tokenizer, uniter_extended, uniter_dim, tokenizer.end_img, do_padding
        )  # dummy placeholders
        question_extended = " ".join(
            [tokenizer.begin_question, question, tokenizer.end_question]
        )
        question_tokens = tokenize(
            tokenizer,
            question_extended,
            max_question_length,
            tokenizer.end_question,
            do_padding,
        )
        answer_extended = " ".join(
            [tokenizer.begin_answer, answer, tokenizer.end_answer]
        )
        answer_tokens = tokenize(
            tokenizer,
            answer_extended,
            max_answer_length,
            tokenizer.end_answer,
            do_padding,
        )
        rationale_extended = " ".join(
            [tokenizer.begin_rationale, rationale, tokenizer.end_rationale]
        )
        rationale_tokens = tokenize(
            tokenizer,
            rationale_extended,
            max_rationale_length,
            tokenizer.end_rationale,
            do_padding,
        )

        prompt = uniter_tokens + question_tokens + answer_tokens + rationale_tokens
        token_type_ids = (
            [0] * len(uniter_tokens)
            + [1] * len(question_tokens)
            + [2] * len(answer_tokens)
            + [3] * len(rationale_tokens)
        )
        prompt, token_type_ids, _ = truncate(
            prompt,
            block_size,
            tokenizer.end_rationale,
            tokenizer.unk_token,
            token_type_ids,
        )
        tokenized_text = tokenizer.convert_tokens_to_ids(prompt)
        labels = get_labels(tokenizer, tokenized_text)

        inputs.append(tokenized_text)
        token_type_ids_list.append(token_type_ids)
        label_list.append(labels)

    inputs = torch.tensor(inputs)
    labels = torch.tensor(label_list)
    token_type_ids = torch.tensor(token_type_ids_list)

    inputs = inputs.to(device)
    labels = labels.to(device)
    token_type_ids = token_type_ids.to(device)

    return inputs, token_type_ids, labels


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = 9
    VISUAL_CONFIG.x_layers = 5
    VISUAL_CONFIG.r_layers = 5


class UniterEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_seq_length = args.max_seq_length
        set_visual_config(args)
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased", do_lower_case=True
        )
        self.model = UFE.from_pretrained("bert-base-cased", args.type_vocab_size)

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, boxes, visual_attention_mask=None):
        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer
        )

        input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long
        )
        input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long
        )
        segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long
        )

        visual_segment_ids = torch.ones(
            input_ids.shape[0], feats.shape[1], dtype=torch.long
        )
        v_mask = torch.ones(input_mask.shape[0], feats.shape[1], dtype=torch.long)

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        visual_segment_ids = visual_segment_ids.to(device)
        v_mask = v_mask.to(device)

        output, lang_v_feats = self.model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            visual_feats=feats,
            visual_token_type_ids=visual_segment_ids,
            visual_attention_mask=v_mask,
            img_pos_feat=boxes,
        )
        return output, lang_v_feats

    def load(self, path):

        state_dict = torch.load(path)

        print("Load UNITER PreTrained Model from %s" % path)
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()
        if "uniter-base-vcr_2nd_stage.pt" in path:
            state_dict["uniter.embeddings.token_type_embeddings.weight"] = state_dict[
                "uniter.embeddings.token_type_embeddings.weight"
            ][:-1, :]
            state_dict["uniter.embeddings.word_embeddings.weight"] = state_dict[
                "uniter.embeddings.word_embeddings.weight"
            ][:28996, :]
        self.model.load_state_dict(state_dict, strict=False)


class gpt2(nn.Module):
    """
    GPT-2 adapted to take visual inputs.
    Inspired from Marasovic et al. 2020 https://arxiv.org/abs/2010.07526
    """

    def __init__(self, args):
        super().__init__()
        self.tokenizer = VCRGpt2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2VisionAttentiveLMHead.from_pretrained("gpt2")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.max_seq_length = args.max_seq_length

    def forward(self, inputs, token_type_ids, labels, uniter_feats):

        outputs = self.model(
            inputs,
            token_type_ids=token_type_ids,
            position_ids=None,
            labels=labels,
            visual_representations=uniter_feats,
            box_coordinates=None,
            orig_num_situ_or_object_boxes=None,
        )

        return outputs
