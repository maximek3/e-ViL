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

import os
import numpy as np
import collections
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from time import gmtime, strftime
from param import args
import random
import json

print(f"process ID: {os.getpid()}")

from transformers import get_linear_schedule_with_warmup

from datasets import load_metric

from tensorboardX import SummaryWriter

from src.nlg_eval import eval_nlp_scores, input_subset, get_nlg_scores

from eUG_model import eUGModel, binary_to_mp
from eViL_data import eViLDataset, eViLTorchDataset, VQAXEvaluator, bbox_collate
from eUG_generation import generate_text

from src.expl_tokenization import VCRGpt2Tokenizer

DataTuple = collections.namedtuple("DataTuple", "dataset loader evaluator")


def ctime():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())


def print_log(args, log_str):
    with open(os.path.join(args.output, "log.log"), "a") as f:
        f.write(log_str)
        f.flush()


def print_dict(dicto):
    out_str = ""
    for k, v in dicto.items():
        out_str += f"{k}: {v:.3f} | "
    return out_str


def map_vcr_tag_to_num(expl):
    dets = ["<|det" + str(i) + "|>" for i in range(10)]
    for idx, det in enumerate(dets):
        expl = expl.replace(det, str(idx))
    return expl


def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:

    dset = eViLDataset(args, splits)
    tset = eViLTorchDataset(args, dset, args.model)
    evaluator = VQAXEvaluator(dset)

    if args.task == "vqa_x":
        collate_fn = None
    else:
        collate_fn = bbox_collate

    data_loader = DataLoader(
        tset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


def dwa(prev_losses, temp=2):
    """
    Implements Dynamic Weight Average (DWA). https://arxiv.org/abs/1803.10704

        inputs:
            task_loss: classification loss on the VL task
            expl_loss: language generation loss of the explanation generation
            prev_losses: list of previous losses
            temp: hyperparameter for the loss average
        output:
            dictionary with weights of classification and explanation loss
    """

    k = 2  # number of tasks

    # fix weights for first step
    if len(prev_losses[0]) <= 3:
        return {"task": 1, "expl": 1}

    task_loss = prev_losses[0][-1]
    expl_loss = prev_losses[1][-1]

    task_prev = prev_losses[0][-2]
    expl_prev = prev_losses[1][-2]

    w1 = task_loss / task_prev
    w2 = expl_loss / expl_prev

    denom = np.exp(w1 / temp) + np.exp(w2 / temp)
    lambda1 = k * np.exp(w1 / temp) / denom
    lambda2 = k * np.exp(w2 / temp) / denom

    return {"task": lambda1, "expl": lambda2}


def weighted_loss(task_loss, expl_loss, loss_weights, cweight):

    # get loss after dwa weighting
    l_c = loss_weights["task"] * task_loss
    l_e = loss_weights["expl"] * expl_loss

    # makes sure sum of losses remains the same and ratio changes cweight-fold
    w_e = (float(l_c) + float(l_e)) / (cweight * float(l_c) + float(l_e))
    w_c = cweight * w_e

    return w_c * l_c + w_e * l_e


def random_print_samples(sent, label, generated_explanations, label2ans):
    """
    Prints a random subset of generated explanations.
    """
    if np.random.choice(np.arange(0, 2), p=[1 - len(sent) / 100, len(sent) / 100]):
        idx = random.randrange(len(sent))
        question_ex = sent[idx]
        label_ex = label[idx]
        if isinstance(label2ans[0], list):
            answer_ex = label2ans[idx][label_ex]
        else:
            answer_ex = label2ans[label_ex]
        explanation_ex = generated_explanations[idx]

        print(
            f"\n********** EVAL EXAMPLE ********** || Question: {question_ex} | Answer: {answer_ex} | Explanation: {explanation_ex}"
        )


def write_items(items, output_file):
    with open(output_file, "w") as f:
        for item in items:
            f.write(str(item) + "\n")
    f.close()


class VQA:
    def __init__(self):

        self.train_type = args.train_type
        self.device = torch.device(args.device)

        # Dataloaders for train and val set
        if not args.test:
            self.valid_tuple = get_data_tuple(
                args.valid, bs=args.batch_size, shuffle=False, drop_last=False
            )
            self.train_tuple = get_data_tuple(
                args.train, bs=args.batch_size, shuffle=True, drop_last=True
            )
            num_answers = self.train_tuple.dataset.num_answers
            file_name = args.train
            log_str = f"\n{ctime()} || Loaded train set of size {len(self.train_tuple[0])} and val set of size {len(self.valid_tuple[0])}."
        else:
            self.test_tuple = get_data_tuple(
                args.test, bs=args.batch_size, shuffle=False, drop_last=False
            )
            num_answers = self.test_tuple.dataset.num_answers
            file_name = args.test
            log_str = (
                f"\n{ctime()} || Loaded test set of size {len(self.test_tuple[0])}."
            )

        # get dataset name
        self.dtype = args.task

        # Model
        self.model = eUGModel(self.train_type, num_answers, self.dtype, args.model)

        # Load pre-trained weights
        if self.train_type == "expl" and args.bb_path is not None:
            self.model.load_state_dict(torch.load(args.bb_path))
            # freeze backbone
            for p, n in self.model.named_parameters():
                if "decoder.model.transformer" not in p:
                    n.requires_grad = False
        elif args.load_pretrained is not None:
            self.model.encoder.load(args.load_pretrained)

        self.model = self.model.to(self.device)

        # Loss and Optimizer
        if not args.test:
            if self.dtype == "vqa_x":
                self.loss_func = nn.BCEWithLogitsLoss()
            else:
                self.loss_func = nn.CrossEntropyLoss()

            batch_per_epoch = len(self.train_tuple.loader) / args.grad_accum
            t_total = int(batch_per_epoch * args.epochs)

            if "bert" in args.optim:
                print("BertAdam Total Iters: %d" % t_total)
                from src.optimization import BertAdam

                self.optim = BertAdam(
                    list(self.model.parameters()),
                    lr=args.lr,
                    warmup=0.1,
                    t_total=t_total,
                )
            else:
                self.optim = args.optimizer(self.model.parameters(), args.lr)
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optim,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=t_total,
                )
        self.grad_accum = args.grad_accum

        # Output Directory
        self.output = args.output
        self.save_steps = args.save_steps
        os.makedirs(self.output, exist_ok=True)

        # print logs
        log_str += f"\n{ctime()} || Model loaded. Batch size {args.batch_size*args.grad_accum} | lr {args.lr} | task: {self.dtype} | type: {self.train_type}."
        print_log(args, log_str)

    def train(self, train_tuple, eval_tuple):

        tb_writer = SummaryWriter(self.output)

        dset, loader, evaluator = train_tuple
        iter_wrapper = (
            (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        )

        # logger initialisations
        best_task = 0.0  # this refers to the model with the best S_T score
        best_expl = 0.0  # this refers to the model with the best S_E score
        best_global = 0.0  # this refers to the model with the best S_O score
        prev_losses = [[1], [1]]
        prev_task, prev_expl = 0, 0
        global_step = 0
        t_loss, tt_loss, te_loss = 0, 0, 0
        step_per_eval = 0

        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (
                ques_id,
                feats,
                boxes,
                sent,
                target,
                expl,
                answer_choices,
            ) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                expl_gt = target

                if self.dtype == "vcr":
                    model_dict = answer_choices
                    target = target.flatten()
                else:
                    model_dict = dset.label2ans

                logit, output, _, _, _ = self.model(
                    feats.to(self.device),
                    boxes.to(self.device),
                    sent,
                    expl,
                    answer_choices,
                    model_dict,
                    expl_gt,
                )

                if self.dtype == "vqa_x":
                    loss_multiplier = logit.size(1)
                elif self.dtype == "vcr":
                    loss_multiplier = 4
                else:
                    loss_multiplier = 1

                if self.train_type == "all":
                    task_loss = (
                        self.loss_func(logit, target.to(self.device)) * loss_multiplier
                    )
                    expl_loss = output[0]
                    # loss_weights = dwa(prev_losses, temp=args.temperature)
                    loss_weights = {"task": 1, "expl": 1}
                    # loss = loss_weights['task']*task_loss + loss_weights['expl']*expl_loss
                    loss = weighted_loss(
                        task_loss, expl_loss, loss_weights, args.classifier_weight
                    )
                    loss /= self.grad_accum

                    prev_task += float(task_loss)
                    prev_expl += float(expl_loss)

                    # record loss for every 1024 datapoints
                    if (i + 1) % int((1024 / args.batch_size)) == 0:
                        prev_losses[0].append(prev_task / (1024 / args.batch_size))
                        prev_losses[1].append(prev_expl / (1024 / args.batch_size))
                        prev_task, prev_expl = 0, 0

                elif self.train_type == "bb":
                    loss = (
                        self.loss_func(logit, target.to(self.device)) * loss_multiplier
                    )
                    loss /= self.grad_accum
                    task_loss = float(loss)
                    expl_loss = 0

                elif self.train_type == "expl":
                    loss = output[0]
                    loss /= self.grad_accum
                    task_loss = 0
                    expl_loss = float(loss)

                loss.backward()

                if self.dtype == "vcr":
                    logit = binary_to_mp(logit)

                score, label = logit.max(1)
                if not isinstance(ques_id, list):
                    ques_id = ques_id.cpu().numpy()

                if self.dtype == "vcr":  # vcr
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        ans = dset.label2ans[qid][l]
                        quesid2ans[qid] = ans
                else:
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        ans = dset.label2ans[l]
                        quesid2ans[qid] = ans

                t_loss += float(loss) * self.grad_accum
                tt_loss += float(task_loss)
                te_loss += float(expl_loss)
                step_per_eval += 1

                # global step
                # grad accum snippet: https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3
                if (i + 1) % self.grad_accum == 0:

                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optim.step()
                    if args.optim != "bert":
                        self.scheduler.step()  # Update learning rate schedule

                    # logging
                    tb_writer.add_scalar("task loss", task_loss, global_step)
                    tb_writer.add_scalar("explanation loss", expl_loss, global_step)
                    tb_writer.add_scalar(
                        "total loss", float(loss) * self.grad_accum, global_step
                    )
                    if self.train_type == "all":
                        tb_writer.add_scalar(
                            "task weight", loss_weights["task"], global_step
                        )
                        tb_writer.add_scalar(
                            "explanation weight", loss_weights["expl"], global_step
                        )

                    global_step += 1

                    # do eval
                    if self.save_steps > 0 and global_step % self.save_steps == 0:
                        log_str = f"\n\n{ctime()} || EVALUATION TIME"
                        log_str += f"\nEpoch-step {epoch}-{global_step}: Loss {t_loss/step_per_eval:.2f} | Task loss {tt_loss/step_per_eval:.2f} | Expl loss {te_loss/step_per_eval:.2f} | Train acc {evaluator.evaluate(quesid2ans)[0]:.2f}"
                        print_log(args, log_str)
                        t_loss, tt_loss, te_loss = 0, 0, 0
                        step_per_eval = 0

                        if self.valid_tuple is not None:  # Do Validation
                            valid_score, valid_perplexity, nlg_scores = self.evaluate(
                                eval_tuple
                            )

                            # no explanations generated
                            if not nlg_scores:

                                if valid_score > best_task:
                                    best_task = valid_score
                                    self.save("best_task")

                                log_str = f"\nEpoch-step {epoch}-{global_step}: Valid Score: {valid_score:.3f} | Best Valid Score: {best_task:.3f}"
                                tb_writer.add_scalar(
                                    "valid_task_score", valid_score * 100.0, global_step
                                )
                                tb_writer.add_scalar(
                                    "valid_expl_perplexity",
                                    valid_perplexity * 100.0,
                                    global_step,
                                )
                                print_log(args, log_str)
                                continue

                            if valid_score > best_task:
                                best_task = valid_score
                                self.save("best_task")

                            if self.train_type == "bb":
                                nlg_avg = 0
                                global_score = 0
                                valid_perplexity = 0
                            else:
                                global_score = nlg_scores["global_score"]
                                if global_score > best_global:
                                    best_global = global_score
                                    self.save("best_global")

                                nlg_avg = nlg_scores["avg_all"]
                                if nlg_avg > best_expl:
                                    best_expl = nlg_avg
                                    self.save("best_expl")

                            log_str = f"\nEpoch-step {epoch}-{global_step}: Valid Score: {valid_score:.3f} | NLG average: {nlg_avg:.3f} | Global score: {global_score:.3f}"
                            log_str += f"\nEpoch-step {epoch}-{global_step}: Best Valid Score: {best_task:.3f} | Best NLG: {best_expl:.3f} | Best overall: {best_global:.3f}"

                            tb_writer.add_scalar(
                                "valid_task_score", valid_score * 100.0, global_step
                            )
                            tb_writer.add_scalar(
                                "valid_expl_perplexity",
                                valid_perplexity * 100.0,
                                global_step,
                            )

                            if nlg_scores:
                                log_str += f"\nEpoch-step {epoch}-{global_step}: {print_dict(nlg_scores)}"
                                for k, v in nlg_scores.items():
                                    tb_writer.add_scalar(k, v, global_step)

                        print(log_str, end="")

                        print_log(args, log_str)

                        tb_writer.flush()

        self.save("LAST")
        tb_writer.close()

    def predict(self, train_type, eval_tuple: DataTuple, dump=None, gen_dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """

        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        expl_loss = 0.0
        nb_eval_steps = 0
        generated_explanations = None
        test_output = []

        if "bb" not in train_type:
            # initialisations for NL evaluation
            try:
                bert_metric = load_metric(
                    "bertscore",
                    experiment_id=str(random.randrange(999999)),
                    device=self.device,
                )
            except:
                bert_metric = None
            all_generated_explanations = []
            all_gt_expls = []
            tokenizer = VCRGpt2Tokenizer.from_pretrained("gpt2")
            gen_model = self.model.decoder.model.to(self.device)

        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent, label, expl, answers = datum_tuple

            if args.gt_cond:
                gt = label
            else:
                gt = None

            if self.dtype == "vcr":  # different label dict
                model_dict = answers
            else:
                model_dict = dset.label2ans

            if self.dtype == "vqa_x":  # multiple explanations
                triple_expl = [[x[y] for x in expl] for y in range(len(expl[0]))]
                expl = expl[0]
            else:
                triple_expl = None

            with torch.no_grad():
                feats, boxes = feats.to(self.device), boxes.to(self.device)
                (
                    logit,
                    expl_output,
                    input_ids,
                    token_type_ids,
                    visual_representations,
                ) = self.model(feats, boxes, sent, expl, answers, model_dict, gt)

                # get indices for when to generate explanations
                if self.dtype == "vqa_x":
                    if args.gt_cond:
                        logit = label
                    correct_indices = []
                    for idx, prediction in enumerate(
                        list(torch.argmax(logit, 1).detach().cpu().numpy())
                    ):
                        if float(label[idx][prediction]) != 0:
                            correct_indices.append(idx)
                    correct_indices = torch.tensor(correct_indices)
                elif self.dtype == "vcr":
                    logit = binary_to_mp(logit)  # transform binary labels into 4-way
                    correct_indices = (
                        torch.where(label.argmax(1) == logit.cpu().argmax(1))[0]
                        .detach()
                        .cpu()
                    )
                else:
                    correct_indices = (
                        torch.where(label.to(self.device) == torch.argmax(logit, 1))[0]
                        .detach()
                        .cpu()
                    )
                    if args.gt_cond:
                        correct_indices = torch.range(0, label.size(0) - 1, dtype=int)

                # populate quesid2ans (where ans is predicted ans)
                if not isinstance(ques_id, list):
                    ques_id = ques_id.cpu().numpy()
                score, label = logit.max(1)
                if self.dtype == "vcr":
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        ans = dset.label2ans[qid][l]
                        quesid2ans[qid] = ans
                else:
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        ans = dset.label2ans[l]
                        quesid2ans[qid] = ans

                # generate and evaluate explanations
                get_gen_expl = 0
                if "bb" not in train_type:
                    expl_loss += expl_output[0].mean().item()

                    # only evaluate random subset during validation to save time
                    if args.test:
                        get_gen_expl = 1
                    else:
                        get_gen_expl = np.random.choice(
                            np.arange(0, 2), p=[1 - args.prob_eval, args.prob_eval]
                        )

                    # get subset where label was predicted correctly
                    (
                        input_ids,
                        token_type_ids,
                        visual_representations,
                        expl,
                        triple_expl,
                    ) = input_subset(
                        correct_indices,
                        input_ids,
                        token_type_ids,
                        visual_representations,
                        expl,
                        triple_expl,
                        self.device,
                    )
                    generated_explanations = None

                    if input_ids.shape[0] != 0:  # if not all predictions were wrong
                        if get_gen_expl:
                            generated_explanations = generate_text(
                                gen_model,
                                tokenizer,
                                input_ids,
                                token_type_ids,
                                visual_representations,
                                max_rationale_length=51,
                            )

                            if self.dtype == "vcr":
                                expl = [
                                    map_vcr_tag_to_num(x) for x in expl
                                ]  # to make sure same kind of explanations are compared

                            # free memory
                            input_ids, token_type_ids, visual_representations = (
                                None,
                                None,
                                None,
                            )

                            if self.dtype == "vqa_x":
                                try:
                                    bert_metric.add_batch(
                                        predictions=generated_explanations,
                                        references=triple_expl,
                                    )
                                except:
                                    print("BertScore failed")
                                all_gt_expls.extend(triple_expl)
                            else:
                                try:
                                    bert_metric.add_batch(
                                        predictions=generated_explanations,
                                        references=expl,
                                    )
                                except:
                                    print("BertScore failed")
                                all_gt_expls.extend(expl)

                            all_generated_explanations.extend(generated_explanations)

                            # printing examples during eval
                            if not args.test:
                                if self.dtype == "vcr":
                                    labels = [
                                        label[i].max(0)[1].item()
                                        for i in correct_indices
                                    ]
                                    model_dict = [answers[i] for i in correct_indices]
                                else:
                                    labels = [label[i].item() for i in correct_indices]
                                random_print_samples(
                                    [sent[i] for i in correct_indices],
                                    labels,
                                    generated_explanations,
                                    model_dict,
                                )

                gen_expl_all = len(ques_id) * ["None"]
                if generated_explanations:
                    for ci, gen_expl in zip(correct_indices, generated_explanations):
                        gen_expl_all[ci] = gen_expl

                # write explanations to file
                if gen_dump:
                    for idx, (qid, gen_expl) in enumerate(
                        zip(list(ques_id), gen_expl_all)
                    ):
                        input_record = {}

                        input_record["question_id"] = str(qid)
                        input_record["question"] = dset.id2datum[qid]["sent"]
                        input_record["generated_explanation"] = gen_expl
                        if self.dtype == "vcr":
                            input_record["correct_explanations"] = (
                                dset.id2datum[qid]["explanation"]
                                .replace("<|det", "")
                                .replace("|>", "")
                            )
                        else:
                            input_record["correct_explanations"] = dset.id2datum[qid][
                                "explanation"
                            ]
                        input_record["prediction"] = quesid2ans[qid]
                        input_record["gt"] = dset.id2datum[qid]["label"]
                        if self.dtype == "vcr":
                            input_record["img_id"] = dset.id2datum[qid]["raw_img_id"]
                            input_record["movie"] = dset.id2datum[qid]["movie"]
                            input_record["answer_choices"] = [
                                x.replace("<|det", "").replace("|>", "")
                                for x in dset.id2datum[qid]["answer_choices"]
                            ]
                        elif self.dtype == "vqax":
                            input_record["img_id"] = dset.id2datum[qid]["img_id"]
                        else:
                            input_record["img_id"] = str(qid)[:-5]
                        if idx in list(correct_indices.numpy()):
                            input_record["correct"] = 1
                        else:
                            input_record["correct"] = 0

                        test_output.append(input_record)

            nb_eval_steps += 1

        valid_score, correct_idx = eval_tuple.evaluator.evaluate(quesid2ans)
        nlg_weight = correct_idx.count(1) / len(
            correct_idx
        )  # because for vqa-x we also take half-correct answers

        # getting perplexity
        expl_loss = expl_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(expl_loss)).item()

        if "bb" not in train_type and len(all_generated_explanations) != 0:

            # getting NLG metrics
            nlg_global_scores = get_nlg_scores(
                self.dtype,
                all_generated_explanations,
                all_gt_expls,
                bert_metric,
                self.device,
            )
            nlg_global_scores["global_score"] = (
                nlg_global_scores["avg_all"] * nlg_weight
            )
            if not nlg_global_scores["global_score"]:
                nlg_global_scores["global_score"] = 0

            if gen_dump is not None:
                scores_to_print = nlg_global_scores
                scores_to_print["task_score"] = valid_score
                write_items(
                    [json.dumps(r) for r in ["scores", scores_to_print]],
                    os.path.join(args.output, "scores.json"),
                )
                write_items(
                    [json.dumps(r) for r in test_output],
                    os.path.join(args.output, "gen_test.json"),
                )

            return valid_score, perplexity, nlg_global_scores
        else:
            scores_to_print = {"task_score": valid_score}
            print("Task Score: ", valid_score)
            write_items(
                [json.dumps(r) for r in ["scores", scores_to_print]],
                os.path.join(args.output, "scores.json"),
            )
            return valid_score, perplexity, None

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        valid_score, expl_perplexity, nlg_global_scores = self.predict(
            self.train_type, eval_tuple, dump
        )
        return valid_score, expl_perplexity, nlg_global_scores

    @staticmethod
    def oracle_score(data_tuple):
        """
        Purpose:
        """
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)


if __name__ == "__main__":

    # logging
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print_log(args, "\n" + str(args) + "\n")
    tb_path = os.path.join(os.getcwd(), args.output)
    log_str = f"\ntensorboard dev upload --logdir {tb_path} --name ug-tt_{args.train_type}-bs{args.batch_size*args.grad_accum}-lr{args.lr}-t{args.temperature}"
    log_str += f"\n Device: {torch.cuda.current_device()}"
    log_str += f"\n Process ID: {os.getpid()}"
    print_log(args, log_str)

    # Build Class
    vqa = VQA()

    # Load VQA model weights
    if args.load_trained is not None:
        vqa.load(args.load_trained)

    # Test or Train
    if args.test:
        valid_score, perplexity, nlg_global_scores = vqa.predict(
            args.train_type,
            vqa.test_tuple,
            dump=os.path.join(args.output, "test_predict.json"),
            gen_dump=os.path.join(args.output, "gen_output.json"),
        )
    else:
        print("Splits in Train data:", vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print("Splits in Valid data:", vqa.valid_tuple.dataset.splits)
            # print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)
