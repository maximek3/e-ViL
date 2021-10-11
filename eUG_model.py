# Copyright 2019 project LXRT.
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

from param import args
from src.entry import UniterEncoder, gpt2, preprocess_gpt2
from src.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = args.max_seq_length


def binary_to_mp(logit):
    """
    convert binary vcr logits to 4-way multiple choice classification logits
    """
    sm = nn.Softmax(dim=1)

    logit = sm(logit)
    logit = logit[:, 1]  # get the values for answer being true of all pairs
    logit = logit.split(4)  # group them into 4's
    logit = torch.stack(logit)  # make one tensor out of it

    return logit


class eUGModel(nn.Module):
    def __init__(self, train_type, num_answers, task, model="uniter"):

        super().__init__()
        self.model = model
        self.train_type = train_type
        self.task = task

        self.encoder = UniterEncoder(args)

        hid_dim = self.encoder.dim

        # VQA-X and e-SNLI-VE answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers),
        )
        self.logit_fc.apply(self.encoder.model.init_bert_weights)

        # VCR answer head
        self.vcr_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.ReLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 2),
        )
        self.vcr_fc.apply(self.encoder.model.init_bert_weights)

        # Explanation generator
        self.decoder = gpt2(args)

    def forward(self, feat, pos, sent, expls, answers, label_dict, gt_label=None):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """

        if self.task == "vcr":
            # create answer-question pairs
            model_input = [
                [s + " [SEP] " + answer for answer in answers[idx]]
                for idx, s in enumerate(sent)
            ]
            model_input = [
                item for sublist in model_input for item in sublist
            ]  # flatten
            x, _ = self.encoder(
                model_input,
                feat.repeat_interleave(repeats=4, dim=0),
                pos.repeat_interleave(repeats=4, dim=0),
            )
            logit = self.vcr_fc(x)

            # get 4-way logit for explanation generation
            logit_expl = binary_to_mp(logit)  # for explanation conditioning during eval

            # get feats without answer
            _, feats = self.encoder(sent, feat, pos)

        else:
            x, feats = self.encoder(sent, feat, pos)
            logit = self.logit_fc(x)
            logit_expl = logit  # for explanation conditioning during eval

        if self.train_type == "bb":  # backbone only
            return logit, None, None, None, None

        # pre-process for decoder
        uniter_dim = feats.shape[1]
        if type(gt_label) != type(None):
            # change to one-hot
            if self.task == "esnlive":
                gt_label = torch.nn.functional.one_hot(gt_label)
            inputs, token_type_ids, labels = preprocess_gpt2(
                self.decoder.tokenizer, sent, expls, gt_label, uniter_dim, label_dict
            )  # training
        else:
            inputs, token_type_ids, labels = preprocess_gpt2(
                self.decoder.tokenizer, sent, expls, logit_expl, uniter_dim, label_dict
            )  # val and test

        # sent through decoder
        expl_output = self.decoder(inputs, token_type_ids, labels, feats)

        return logit, expl_output, inputs, token_type_ids, feats
