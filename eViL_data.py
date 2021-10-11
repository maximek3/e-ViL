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

import json
import os
import base64

import numpy as np
import torch
from torch.utils.data import Dataset

import lmdb
import msgpack
import msgpack_numpy

msgpack_numpy.patch()


# The path to data and image features.
VQA_DATA_ROOT = "data/"
MSCOCO_IMGFEAT_ROOT = "data/fasterRCNN_features/"
SPLIT2NAME = {
    "train": "train2014",
    "valid": "val2014",
    "minival": "val2014",
    "nominival": "val2014",
    "test": "test2015",
    "train_x": "train2014",
    "val_x": "val2014",
    "test_x": "val2014",
}

FLICKR30KDB = "data/esnlive/img_db/flickr30k/feat_th0.2_max100_min10"
FLICKR30KDB_NBB = "data/esnlive/img_db/flickr30k/nbb_th0.2_max100_min10.json"


def vcr_path(split):
    db = f"data/vcr/img_db/vcr_{split}/feat_th0.2_max100_min10"
    nbb = f"data/vcr/img_db/vcr_{split}/nbb_th0.2_max100_min10.json"
    return db, nbb


def pad_array(vec, pad_size):
    return np.pad(vec, [(0, pad_size - vec.shape[0]), (0, 0)], mode="constant")


def bbox_collate(batch):

    ques_id, feats, boxes, sent, target, expl, answers = [], [], [], [], [], [], []

    for example in batch:
        ques_id.append(example[0])
        feats.append(example[1])
        boxes.append(example[2])
        sent.append(example[3])
        if isinstance(example[4], int):
            target.append(example[4])
        else:
            target.append(example[4].tolist())
        expl.append(example[5])
        answers.append(example[6])

    max_len = max(map(lambda x: x.shape[0], feats))
    padded_feats = [pad_array(x, max_len) for x in feats]
    padded_boxes = [pad_array(x, max_len) for x in boxes]

    if not answers[0]:
        answers = None

    return (
        ques_id,
        torch.tensor(padded_feats).float(),
        torch.tensor(padded_boxes).float(),
        tuple(sent),
        torch.tensor(target),
        tuple(expl),
        answers,
    )


class eViLDataset:
    """
    Initialises id2datum (dict where keys are id's of each datapoint)
    Initialises an2label and label2ans, which are required throughout the code.
    """

    def __init__(self, args, splits: str):

        self.task = args.task
        self.name = splits
        self.splits = splits.split(",")

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {datum["question_id"]: datum for datum in self.data}

        # Answers
        if self.task == "vqa_x":
            self.ans2label = json.load(open("data/trainval_ans2label.json"))
            self.label2ans = json.load(open("data/trainval_label2ans.json"))
            assert len(self.ans2label) == len(self.label2ans)

        elif self.task == "esnlive":
            self.ans2label = {"contradiction": 0, "neutral": 1, "entailment": 2}
            self.label2ans = {0: "contradiction", 1: "neutral", 2: "entailment"}

        elif self.task == "vcr":
            self.label2ans = {x["question_id"]: x["answer_choices"] for x in self.data}

        else:
            print("missing or invalid task")

    @property
    def num_answers(self):
        if self.task == "vcr":
            return 4
        else:
            return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""


class eViLTorchDataset(Dataset):
    def __init__(self, args, dataset: eViLDataset, model="lxmert", max_length=50):

        super().__init__()
        self.raw_dataset = dataset
        self.model = model
        self.task = args.task

        if self.task == "vqa_x":
            self.offset = {}
            for split in self.raw_dataset.splits:
                f = open(
                    os.path.join(
                        MSCOCO_IMGFEAT_ROOT,
                        "%s_offset.txt"
                        % (
                            SPLIT2NAME[
                                split.replace("_toy", "")
                                .replace(".json", "")
                                .split(
                                    "/",
                                )[-1]
                            ]
                        ),
                    )
                )
                offset = f.readlines()
                for l in offset:
                    self.offset[l.split("\t")[0]] = int(l.split("\t")[1].strip())

            # remove all images for which no features are available
            self.data = []
            for datum in self.raw_dataset.data:
                if datum["img_id"] in self.offset.keys():
                    self.data.append(datum)
            print(
                f"Used {len(self.data)} of {len(self.raw_dataset)} available datapoints from the {self.task} dataset."
            )
            print()
        else:  # initialize mdb stuff
            if self.task == "esnlive":
                img_path = FLICKR30KDB
                nbb_path = FLICKR30KDB_NBB
            elif self.task == "vcr":
                if "_train_" in self.raw_dataset.splits[0]:
                    img_path, nbb_path = vcr_path("train")
                elif "_dev_" in self.raw_dataset.splits[0]:
                    img_path, nbb_path = vcr_path("train")
                else:
                    img_path, nbb_path = vcr_path("val")
            self.env = lmdb.open(
                img_path, readonly=True, create=False, readahead=not False
            )
            self.txn = self.env.begin(buffers=True)
            self.name2nbb = json.load(open(nbb_path))

    def __len__(self):
        if self.task == "vqa_x":
            return len(self.data)
        else:
            return len(self.raw_dataset.data)

    def __getitem__(self, item: int):

        datum = self.raw_dataset.data[item]
        img_id = datum["img_id"]
        ques_id = datum["question_id"]
        ques = datum["sent"]

        # getting image features
        if self.task == "vqa_x":

            img_offset = self.offset[img_id]
            img_split = img_id[5:7]
            if img_split == "tr":
                f = open(
                    os.path.join(
                        MSCOCO_IMGFEAT_ROOT, "%s_obj36.tsv" % (SPLIT2NAME["train"])
                    )
                )
            elif img_split == "va":
                f = open(
                    os.path.join(
                        MSCOCO_IMGFEAT_ROOT, "%s_obj36.tsv" % (SPLIT2NAME["valid"])
                    )
                )
            else:
                f = open(
                    os.path.join(
                        MSCOCO_IMGFEAT_ROOT, "%s_obj36.tsv" % (SPLIT2NAME["test"])
                    )
                )
            f.seek(img_offset)
            img_info = f.readline()
            f.close()

            assert img_info.startswith("COCO") and img_info.endswith(
                "\n"
            ), "Offset is inappropriate"
            img_info = img_info.split("\t")

            decode_img = self._decodeIMG(img_info)
            img_h = decode_img[0]
            img_w = decode_img[1]
            feats = decode_img[-1].copy()
            boxes = decode_img[-2].copy()
            del decode_img

            # Normalize the boxes (to 0 ~ 1)
            if self.model == "uniter":
                boxes = self._uniterBoxes(boxes)
            else:
                boxes[:, (0, 2)] /= img_w
                boxes[:, (1, 3)] /= img_h

                np.testing.assert_array_less(boxes, 1 + 1e-5)
                np.testing.assert_array_less(-boxes, 0 + 1e-5)

        else:

            dump = self.txn.get(img_id.encode("utf-8"))
            nbb = self.name2nbb[img_id]
            img_dump = msgpack.loads(dump, raw=False)
            feats = img_dump["features"][:nbb, :]
            img_bb = img_dump["norm_bb"][:nbb, :]

            # get box to same format than used by code's authors
            boxes = np.zeros((img_bb.shape[0], 7), dtype="float32")
            boxes[:, :-1] = img_bb[:, :]
            boxes[:, 4] = img_bb[:, 5]
            boxes[:, 5] = img_bb[:, 4]
            boxes[:, 4] = img_bb[:, 5]
            boxes[:, 6] = boxes[:, 4] * boxes[:, 5]

        if "label" in datum:
            label = datum["label"]
            if self.task == "vqa_x":
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    target[self.raw_dataset.ans2label[ans]] = score
            elif self.task == "esnlive":
                target = self.raw_dataset.ans2label[label]
            elif self.task == "vcr":
                target = torch.zeros(self.raw_dataset.num_answers).int()
                target[label] = 1
            else:
                print("Task not defined")

            if "explanation" in datum:
                # get multiple expl for validatin of vqa-x, else just one
                if self.task == "vcr":
                    expl = datum["explanation"]
                elif self.task == "esnlive":
                    expl = datum["explanation"][0]
                else:
                    if "train" in self.raw_dataset.splits[0]:
                        expl = datum["explanation"][
                            0
                        ]  # we only consider the first explanation for training
                    else:
                        expl = datum["explanation"]
                if "answer_choices" in datum:  # required for conditioning explanations
                    answers = datum["answer_choices"]
                else:
                    answers = 1

                return ques_id, feats, boxes, ques, target, expl, answers
            else:
                return ques_id, feats, boxes, ques, target

        else:
            return ques_id, feats, boxes, ques

    def _decodeIMG(self, img_info):
        img_h = int(img_info[1])
        img_w = int(img_info[2])
        boxes = img_info[-2]
        boxes = np.frombuffer(base64.b64decode(boxes), dtype=np.float32)
        boxes = boxes.reshape(36, 4)
        boxes.setflags(write=False)
        feats = img_info[-1]
        feats = np.frombuffer(base64.b64decode(feats), dtype=np.float32)
        feats = feats.reshape(36, -1)
        feats.setflags(write=False)
        return [img_h, img_w, boxes, feats]

    def _uniterBoxes(self, boxes):
        new_boxes = np.zeros((boxes.shape[0], 7), dtype="float32")
        new_boxes = np.zeros((boxes.shape[0], 7), dtype="float32")
        new_boxes[:, 1] = boxes[:, 0]
        new_boxes[:, 0] = boxes[:, 1]
        new_boxes[:, 3] = boxes[:, 2]
        new_boxes[:, 2] = boxes[:, 3]
        new_boxes[:, 4] = new_boxes[:, 3] - new_boxes[:, 1]
        new_boxes[:, 5] = new_boxes[:, 2] - new_boxes[:, 0]
        new_boxes[:, 6] = new_boxes[:, 4] * new_boxes[:, 5]
        return new_boxes


class VQAXEvaluator:
    def __init__(self, dataset: eViLDataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.0
        correct_idx = []
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum["label"]
            correct = 0
            if isinstance(label, dict):  # vqa-x
                if ans in label:
                    score += label[ans]
                    correct = 1
            elif "vcr" in self.dataset.name:  # vcr
                if ans == datum["answer_choices"][label]:
                    score += 1
                    correct = 1
            else:  # esnlive
                if ans == label:
                    score += 1
                    correct = 1
            correct_idx.append(correct)

        return score / len(quesid2ans), correct_idx

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, "w") as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({"question_id": ques_id, "answer": ans})
            json.dump(result, f, indent=4, sort_keys=True)
