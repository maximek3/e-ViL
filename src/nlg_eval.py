# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.3
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

from cococaption.pycocoevalcap.bleu.bleu import Bleu
from cococaption.pycocoevalcap.cider.cider import Cider
from cococaption.pycocoevalcap.meteor.meteor import Meteor
from cococaption.pycocoevalcap.rouge.rouge import Rouge
from cococaption.pycocoevalcap.spice.spice import Spice

from datasets import load_metric
import torch
import statistics

import subprocess


def eval_nlp_scores(pred, gt, verbose=False):
    """
    evaluates the nlp scores bleu1-bleu4, meteor, rouge-l, cider, spice

    Args:
        pred (List): List of predictions
        gt (List): List of ground truths
    """
    if len(pred) == len(gt) == 0:
        return {}

    gts = {}
    res = {}

    for imgId in range(len(pred)):
        gts[imgId] = gt[imgId]
        res[imgId] = pred[imgId]

    # Set up scorers
    if verbose:
        print("Setting up scorers...")
    results = {}
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), ["METEOR"]),
        (Rouge(), ["ROUGE_L"]),
        (Cider(), ["CIDEr"]),
        (Spice(), ["SPICE"]),  # NOTE: SPICE is VERY slow
    ]
    # Compute scores
    for scorer, method in scorers:
        if verbose:
            print("Computing %s score..." % (scorer.method()))
        try:
            # NOTE: may crash when run with very little training
            corpus_score, sentence_scores = scorer.compute_score(gts, res)

            # iterate (for bleu)
            for ind in range(len(method)):
                cs, ss = corpus_score, sentence_scores
                if isinstance(corpus_score, list):
                    cs, ss = corpus_score[ind], sentence_scores[ind]

                results[method[ind]] = cs, ss

                if verbose:
                    print("%s: %0.3f" % (method[ind], cs))

        except subprocess.CalledProcessError:
            if verbose:
                print(
                    f"Error during calling of {method} in local rank {self.local_rank}."
                )

    return results


def input_subset(
    correct_idx,
    input_ids,
    token_type_ids,
    visual_representations,
    expl,
    triple_expl,
    device,
):
    """
    Get subset of examples where label was predicted correctly. We only measure NLG metrics for those.
    """

    input_ids = input_ids.index_select(0, torch.tensor(correct_idx).to(device))
    token_type_ids = token_type_ids.index_select(
        0, torch.tensor(correct_idx).to(device)
    )
    visual_representations = visual_representations.index_select(
        0, torch.tensor(correct_idx).to(device)
    )

    expl = [expl[i] for i in correct_idx]
    if triple_expl:
        triple_expl = [triple_expl[i] for i in correct_idx]

    return input_ids, token_type_ids, visual_representations, expl, triple_expl


def get_average_nlg(nlg_global_scores):
    """
    Combines the different NLG metrics into two scores.
    """
    chosen_metrics = ["METEOR", "ROUGE_L", "CIDEr", "SPICE"]
    chosen_overlap_scores = dict((k, nlg_global_scores[k]) for k in chosen_metrics)
    nlg_global_scores["avg_overlap"] = statistics.harmonic_mean(
        list(chosen_overlap_scores.values())
    )
    nlg_global_scores["avg_all"] = statistics.harmonic_mean(
        [nlg_global_scores["avg_overlap"], nlg_global_scores["BERTScore"]]
    )

    return nlg_global_scores


def get_nlg_scores(dtype, gen_expl, gt_expl, bert_metric, device):

    # getting NLG metrics
    if dtype == "vqa_x":
        nlg_scores = eval_nlp_scores([[x] for x in gen_expl], gt_expl)
    else:
        nlg_scores = eval_nlp_scores([[x] for x in gen_expl], [[x] for x in gt_expl])

    nlg_global_scores = {k: v[0] for k, v in nlg_scores.items()}

    try:
        bert_scores = bert_metric.compute(
            model_type="distilbert-base-uncased", device=device
        )
        nlg_global_scores["BERTScore"] = bert_scores["f1"].mean().item()
    except:
        print("BERTScore failed, so set to 0.8")
        nlg_global_scores["BERTScore"] = 0.8

    nlg_global_scores = get_average_nlg(nlg_global_scores)

    return nlg_global_scores
