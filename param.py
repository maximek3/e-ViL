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

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == "rms":
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == "adam":
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == "adamax":
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == "sgd":
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif "bert" in optim:
        optimizer = "bert"  # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="uniter")
    parser.add_argument(
        "--task", type=str, help="name of dataset: [vqa_x, vcr, esnlive]"
    )

    # Data Splits
    parser.add_argument("--train", default="train_x", help="path to train set")
    parser.add_argument("--valid", default="val_x", help="path to dev set")
    parser.add_argument("--test", default=None, help="path to test set")

    # Training Hyper-parameters
    parser.add_argument("--batchSize", dest="batch_size", type=int, default=32)
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--train_type",
        type=str,
        default="all",
        help="[all: train-both, bb: backbone-only, expl: expl-only]",
    )
    parser.add_argument(
        "--BBPath",
        dest="bb_path",
        type=str,
        default=None,
        help="If pretrained path is fed, task model will remain frozen.",
    )
    parser.add_argument("--optim", default="bert")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=2,
        help="Tempereature parameter for DWA loss.",
    )
    parser.add_argument(
        "--warmup_steps", default=50, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=9595, help="random seed")
    parser.add_argument(
        "--max_seq_length", type=int, default=20, help="max sequence_length"
    )
    parser.add_argument(
        "--classifier_weight",
        default=1,
        type=float,
        help="The weight of the classification loss in the total loss.",
    )

    # Debugging
    parser.add_argument(
        "--output",
        type=str,
        default="models/trained/",
        help="location to store logs and weights",
    )
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument(
        "--prob_eval",
        default=1,
        type=float,
        help="Probability of generating and evaluating explanation for an example.",
    )
    parser.add_argument("--tqdm", action="store_const", default=True, const=True)

    # Model Loading
    parser.add_argument(
        "--load_trained",
        type=str,
        default=None,
        help="Load the model (usually the fine-tuned model).",
    )
    parser.add_argument(
        "--load_pretrained",
        dest="load_pretrained",
        type=str,
        default=None,
        help="Load the pre-trained LXMERT/VisualBERT/UNITER model.",
    )
    parser.add_argument(
        "--fromScratch",
        dest="from_scratch",
        action="store_const",
        default=False,
        const=True,
        help="If none of the --load_trained, --load_pretrained, is set, "
        "the model would be trained from scratch. If --fromScratch is"
        " not specified, the model would load BERT-pre-trained weights by"
        " default. ",
    )

    # Optimization
    parser.add_argument(
        "--mceLoss", dest="mce_loss", action="store_const", default=False, const=True
    )

    # Testing option
    parser.add_argument(
        "--gt_cond",
        dest="gt_cond",
        type=int,
        default=None,
        help="Will condition all the explanations on the ground-truth "
        "answer instead of the predicted one during evaluation",
    )

    # Training configuration
    parser.add_argument("--multiGPU", action="store_const", default=False, const=True)
    parser.add_argument("--numWorkers", dest="num_workers", default=0)
    parser.add_argument("--device", dest="device", default="cpu")

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # type_vocab_size
    if args.task == "vcr":
        args.type_vocab_size = 3
    else:
        args.type_vocab_size = 2

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
