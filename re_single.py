# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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



""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function
import argparse
import glob
import operator
import logging
import os
import random
from collections import defaultdict
import re
import shutil
import numpy
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import time
from ner_single import predict_ner
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertTokenizer,
                          RobertaConfig,
                          RobertaTokenizer,
                          get_linear_schedule_with_warmup,
                          AdamW,
                          BertForACEBothOneDropoutSub,
                          AlbertForACEBothSub,
                          AlbertConfig,
                          AlbertTokenizer,
                          AlbertForACEBothOneDropoutSub,
                          BertForACEBothOneDropoutSubNoNer,
                          )

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset
import json
import pickle
import numpy as np
import unicodedata
import itertools
import timeit

from tqdm import tqdm
from nltk.tree import Tree
from nltk.chunk import conlltags2tree



logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, AlbertConfig)), ())

MODEL_CLASSES = {
    'bertsub': (BertConfig, BertForACEBothOneDropoutSub, BertTokenizer),
    'bertnonersub': (BertConfig, BertForACEBothOneDropoutSubNoNer, BertTokenizer),
    'albertsub': (AlbertConfig, AlbertForACEBothOneDropoutSub, AlbertTokenizer),
}

task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['PER-SOC', 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
}


class ACEDataset_re_single(Dataset):
    def __init__(self, sentence,  tokenizer, args=None, evaluate=False, do_test=False, max_pair_length=None):
        self.sentence = sentence
        # if not evaluate:
        #     file_path = os.path.join(args.data_dir, args.train_file)
        # else:
        #     if do_test:
        #         if args.test_file.find('models') == -1:
        #             file_path = os.path.join(args.data_dir, args.test_file)
        #         else:
        #             file_path = args.test_file
        #     else:
        #         if args.dev_file.find('models') == -1:
        #             file_path = os.path.join(args.data_dir, args.dev_file)
        #         else:
        #             file_path = args.dev_file
        #
        # assert os.path.isfile(file_path)


        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_pair_length = max_pair_length
        self.max_entity_length = self.max_pair_length * 2

        self.evaluate = evaluate
        self.use_typemarker = args.use_typemarker
        self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type
        self.no_sym = args.no_sym

        if args.data_dir.find('ace05') != -1:
            self.ner_label_list = ['NIL', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']

            if args.no_sym:
                label_list = ['PER-SOC', 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE']
                self.sym_labels = ['NIL', 'PER-SOC']
                self.label_list = self.sym_labels + label_list

        elif args.data_dir.find('ace04') != -1:
            self.ner_label_list = ['NIL', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']

            if args.no_sym:
                label_list = ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS']
                self.sym_labels = ['NIL', 'PER-SOC']
                self.label_list = self.sym_labels + label_list

        elif args.data_dir.find('scierc') != -1:
            self.ner_label_list = ['NIL', 'Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric']

            if args.no_sym:
                label_list = ['CONJUNCTION', 'COMPARE', 'PART-OF', 'USED-FOR', 'FEATURE-OF', 'EVALUATE-FOR',
                              'HYPONYM-OF']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'EVALUATE-FOR', 'HYPONYM-OF']
                self.sym_labels = ['NIL', 'CONJUNCTION', 'COMPARE']
                self.label_list = self.sym_labels + label_list

        else:
            self.ner_label_list = ['NIL', 'entity']

            if args.no_sym:
                label_list = ['股票代码', '合作客户', '研究领域', '细分', '下游企业', '应用领域', '主营', '细分产品', '别称', '上游企业', '国别', 'domain_relation', '商品名']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
                # self.label_list = ['主营','产品','细分主营','研究领域','应用领域','合作客户','上游企业','下游企业']
            else:
                label_list = ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'EVALUATE-FOR', 'HYPONYM-OF']
                self.sym_labels = ['NIL', 'CONJUNCTION', 'COMPARE']
                self.label_list = self.sym_labels + label_list
                self.label_list = ['主营', '产品', '细分主营', '研究领域', '应用领域', '合作客户', '上游企业', '下游企业']

        self.global_predicted_ners = {}
        self.initialize()

    def initialize(self):
        tokenizer = self.tokenizer
        vocab_size = tokenizer.vocab_size
        max_num_subwords = self.max_seq_length - 4  # for two marker
        label_map = {label: i for i, label in enumerate(self.label_list)}
        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}

        def tokenize_word(text):
            if (
                    isinstance(tokenizer, RobertaTokenizer)
                    and (text[0] != "'")
                    and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        #f = open(self.file_path, "r", encoding='utf-8')
        self.ner_tot_recall = 0
        self.tot_recall = 0
        self.data = []
        self.ner_golden_labels = set([])
        self.golden_labels = set([])
        self.golden_labels_withner = set([])
        maxR = 0
        maxL = 0
        l_idx = 0

        data = json.loads(self.sentence)


        #
        sentences = data['sentences']
        # if 'predicted_ner' in data:  # e2e predict
        #     ners = data['predicted_ner']
        # else:
        ners = data['predicted_ner']

        std_ners = data['predicted_ner']


        #relations = data['relations']

        # for sentence_relation in relations:
        #     for x in sentence_relation:
        #         if x[4] in self.sym_labels[1:]:
        #             self.tot_recall += 2
        #         else:
        #             self.tot_recall += 1

        sentence_boundaries = [0]
        words = []
        L = 0
        for i in range(len(sentences)):
            L += len(sentences[i])
            sentence_boundaries.append(L)
            words += sentences[i]

        tokens = [tokenize_word(w) for w in words]
        subwords = [w for li in tokens for w in li]
        maxL = max(maxL, len(subwords))
        subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
        token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
        subword_start_positions = frozenset(token2subword)
        subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]



        for n in range(len(subword_sentence_boundaries) - 1):

            sentence_ners = ners[n]
            #sentence_relations = relations[n]
            std_ner = std_ners[n]



            std_entity_labels = {}
            self.ner_tot_recall += len(std_ner)

            for start, end, label in std_ner:
                std_entity_labels[(start, end)] = label
                self.ner_golden_labels.add(((l_idx, n), (start, end), label))

            self.global_predicted_ners[(l_idx, n)] = list(sentence_ners)

            doc_sent_start, doc_sent_end = subword_sentence_boundaries[n: n + 2]

            left_length = doc_sent_start
            right_length = len(subwords) - doc_sent_end
            sentence_length = doc_sent_end - doc_sent_start
            half_context_length = int((max_num_subwords - sentence_length) / 2)

            if sentence_length < max_num_subwords:

                if left_length < right_length:
                    left_context_length = min(left_length, half_context_length)
                    right_context_length = min(right_length,
                                               max_num_subwords - left_context_length - sentence_length)
                else:
                    right_context_length = min(right_length, half_context_length)
                    left_context_length = min(left_length,
                                              max_num_subwords - right_context_length - sentence_length)

            doc_offset = doc_sent_start - left_context_length
            target_tokens = subwords[doc_offset: doc_sent_end + right_context_length]
            target_tokens = [tokenizer.cls_token] + target_tokens[: self.max_seq_length - 4] + [tokenizer.sep_token]
            assert (len(target_tokens) <= self.max_seq_length - 2)

            pos2label = {}
            # for x in sentence_relations:
            #     pos2label[(x[0], x[1], x[2], x[3])] = label_map[x[4]]
            #     self.golden_labels.add(((l_idx, n), (x[0], x[1]), (x[2], x[3]), x[4]))
            #     self.golden_labels_withner.add(((l_idx, n), (x[0], x[1], std_entity_labels[(x[0], x[1])]),
            #                                     (x[2], x[3], std_entity_labels[(x[2], x[3])]), x[4]))
            #     if x[4] in self.sym_labels[1:]:
            #         self.golden_labels.add(((l_idx, n), (x[2], x[3]), (x[0], x[1]), x[4]))
            #         self.golden_labels_withner.add(((l_idx, n), (x[2], x[3], std_entity_labels[(x[2], x[3])]),
            #                                         (x[0], x[1], std_entity_labels[(x[0], x[1])]), x[4]))

            entities = list(sentence_ners)

            # for x in sentence_relations:
            #     w = (x[2], x[3], x[0], x[1])
            #     if w not in pos2label:
            #         if x[4] in self.sym_labels[1:]:
            #             pos2label[w] = label_map[x[4]]  # bug
            #         else:
            #             pos2label[w] = label_map[x[4]] + len(label_map) - len(self.sym_labels)

            if not self.evaluate:
                entities.append((10000, 10000, 'NIL'))  # only for NER

            for sub in entities:
                cur_ins = []

                if sub[0] < 10000:
                    sub_s = token2subword[sub[0]] - doc_offset + 1
                    sub_e = token2subword[sub[1] +1] - doc_offset
                    sub_label = ner_label_map[sub[2]]

                    if self.use_typemarker:
                        l_m = '[unused%d]' % (2 + sub_label)
                        r_m = '[unused%d]' % (2 + sub_label + len(self.ner_label_list))
                    else:
                        l_m = '[unused0]'
                        r_m = '[unused1]'

                    sub_tokens = target_tokens[:sub_s] + [l_m] + target_tokens[sub_s:sub_e + 1] + [
                        r_m] + target_tokens[sub_e + 1:]
                    sub_e += 2
                else:
                    sub_s = len(target_tokens)
                    sub_e = len(target_tokens) + 1
                    sub_tokens = target_tokens + ['[unused0]', '[unused1]']
                    sub_label = -1

                if sub_e >= self.max_seq_length - 1:
                    continue
                # assert(sub_e < self.max_seq_length)
                for start, end, obj_label in sentence_ners:
                    if self.model_type.endswith('nersub'):
                        if start == sub[0] and end == sub[1]:
                            continue

                    doc_entity_start = token2subword[start]
                    doc_entity_end = token2subword[end +1]
                    left = doc_entity_start - doc_offset + 1
                    right = doc_entity_end - doc_offset

                    obj = (start, end)
                    if obj[0] >= sub[0]:
                        left += 1
                        if obj[0] > sub[1]:
                            left += 1

                    if obj[1] >= sub[0]:
                        right += 1
                        if obj[1] > sub[1]:
                            right += 1

                    label = pos2label.get((sub[0], sub[1], obj[0], obj[1]), 0)

                    if right >= self.max_seq_length - 1:
                        continue

                    cur_ins.append(((left, right, ner_label_map[obj_label]), label, obj))

                maxR = max(maxR, len(cur_ins))
                dL = self.max_pair_length
                if self.args.shuffle:
                    np.random.shuffle(cur_ins)

                for i in range(0, len(cur_ins), dL):
                    examples = cur_ins[i: i + dL]
                    item = {
                        'index': (l_idx, n),
                        'sentence': sub_tokens,
                        'examples': examples,
                        'sub': (sub, (sub_s, sub_e), sub_label),  # (sub[0], sub[1], sub_label),
                    }

                    self.data.append(item)
        logger.info('maxR: %s', maxR)
        logger.info('maxL: %s', maxL)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        sub, sub_position, sub_label = entry['sub']
        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])

        L = len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))

        attention_mask = torch.zeros(
            (self.max_entity_length + self.max_seq_length, self.max_entity_length + self.max_seq_length),
            dtype=torch.int64)
        attention_mask[:L, :L] = 1

        if self.model_type.startswith('albert'):
            input_ids = input_ids + [30002] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (
                        self.max_pair_length - len(entry['examples']))
            input_ids = input_ids + [30003] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (
                        self.max_pair_length - len(entry['examples']))  # for debug
        else:
            input_ids = input_ids + [3] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (
                        self.max_pair_length - len(entry['examples']))
            input_ids = input_ids + [4] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (
                        self.max_pair_length - len(entry['examples']))  # for debug

        labels = []
        ner_labels = []
        mention_pos = []
        mention_2 = []
        position_ids = list(range(self.max_seq_length)) + [0] * self.max_entity_length
        num_pair = self.max_pair_length

        for x_idx, obj in enumerate(entry['examples']):
            m2 = obj[0]
            label = obj[1]

            mention_pos.append((m2[0], m2[1]))
            mention_2.append(obj[2])

            w1 = x_idx
            w2 = w1 + num_pair

            w1 += self.max_seq_length
            w2 += self.max_seq_length

            position_ids[w1] = m2[0]
            position_ids[w2] = m2[1]

            for xx in [w1, w2]:
                for yy in [w1, w2]:
                    attention_mask[xx, yy] = 1
                attention_mask[xx, :L] = 1

            labels.append(label)
            ner_labels.append(m2[2])

            if self.use_typemarker:
                l_m = '[unused%d]' % (2 + m2[2] + len(self.ner_label_list) * 2)
                r_m = '[unused%d]' % (2 + m2[2] + len(self.ner_label_list) * 3)
                l_m = self.tokenizer._convert_token_to_id(l_m)
                r_m = self.tokenizer._convert_token_to_id(r_m)
                input_ids[w1] = l_m
                input_ids[w2] = r_m

        pair_L = len(entry['examples'])
        if self.args.att_left:
            attention_mask[self.max_seq_length: self.max_seq_length + pair_L,
            self.max_seq_length: self.max_seq_length + pair_L] = 1
        if self.args.att_right:
            attention_mask[self.max_seq_length + num_pair: self.max_seq_length + num_pair + pair_L,
            self.max_seq_length + num_pair: self.max_seq_length + num_pair + pair_L] = 1

        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))
        labels += [-1] * (num_pair - len(labels))
        ner_labels += [-1] * (num_pair - len(ner_labels))

        item = [torch.tensor(input_ids),
                attention_mask,
                torch.tensor(position_ids),
                torch.tensor(sub_position),
                torch.tensor(mention_pos),
                torch.tensor(labels, dtype=torch.int64),
                torch.tensor(ner_labels, dtype=torch.int64),
                torch.tensor(sub_label, dtype=torch.int64)
                ]

        if self.evaluate:
            item.append(entry['index'])
            item.append(sub)
            item.append(mention_2)

        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]

        num_metadata_fields = 3
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        return stacked_fields


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)





def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(sentence, args, model, tokenizer, prefix="", do_test=False):
    eval_output_dir = args.output_dir


    eval_dataset = ACEDataset_re_single(sentence= sentence, tokenizer=tokenizer, args=args, evaluate=True, do_test=do_test,
                              max_pair_length=args.max_pair_length)

    #golden_labels = set(eval_dataset.golden_labels)
    #golden_labels_withner = set(eval_dataset.golden_labels_withner)
    label_list = list(eval_dataset.label_list)
    sym_labels = list(eval_dataset.sym_labels)
    #tot_recall = eval_dataset.tot_recall

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = 1

    scores = defaultdict(dict)
    # ner_pred = not args.model_type.endswith('noner')
    example_subs = set([])
    num_label = len(label_list)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=ACEDataset_re_single.collate_fn,
                                 num_workers=4 * int(args.output_dir.find('test') == -1))

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))

    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        indexs = batch[-3]
        subs = batch[-2]
        batch_m2s = batch[-1]
        ner_labels = batch[6]

        batch = tuple(t.to(args.device) for t in batch[:-3])


        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'position_ids': batch[2],
                      #   'labels':         batch[4],
                      #   'ner_labels':     batch[5],
                      }

            inputs['sub_positions'] = batch[3]
            if args.model_type.find('span') != -1:
                inputs['mention_pos'] = batch[4]

            outputs = model(**inputs)

            logits = outputs[0]


            if args.eval_logsoftmax:  # perform a bit better
                logits = torch.nn.functional.log_softmax(logits, dim=-1)
                #ner_values, ner_preds = torch.max(logits, dim=-1)

            elif args.eval_softmax:
                logits = torch.nn.functional.softmax(logits, dim=-1)
                #ner_values, ner_preds = torch.max(logits, dim=-1)
            logits = torch.nn.functional.softmax(logits, dim=-1)
            if args.use_ner_results or args.model_type.endswith('nonersub'):
                ner_preds = ner_labels
            else:
                ner_preds = torch.argmax(outputs[1], dim=-1)

            logits = logits.cpu().numpy()
            ner_preds = ner_preds.cpu().numpy()
            for i in range(len(indexs)):
                index = indexs[i]
                sub = subs[i]
                m2s = batch_m2s[i]
                example_subs.add(((index[0], index[1]), (sub[0], sub[1])))
                for j in range(len(m2s)):
                    obj = m2s[j]
                    ner_label = eval_dataset.ner_label_list[ner_preds[i, j]]
                    scores[(index[0], index[1])][((sub[0], sub[1]), (obj[0], obj[1]))] = (
                    logits[i, j].tolist(), ner_label)
                    #print(float(ner_values[i, j]))



    cor = 0
    tot_pred = 0
    cor_with_ner = 0
    global_predicted_ners = eval_dataset.global_predicted_ners
    ner_golden_labels = eval_dataset.ner_golden_labels

    ner_cor = 0
    ner_tot_pred = 0
    ner_ori_cor = 0
    tot_output_results = defaultdict(list)
    confidence_scores = []

    if not args.eval_unidirect:  # eval_unidrect is for ablation study
        # print (len(scores))
        for example_index, pair_dict in sorted(scores.items(), key=lambda x: x[0]):
            visited = set([])
            sentence_results = []
            for k1, (v1, v2_ner_label) in pair_dict.items():

                if k1 in visited:
                    continue
                visited.add(k1)

                if v2_ner_label == 'NIL':
                    continue
                v1 = list(v1)
                m1 = k1[0]
                m2 = k1[1]
                if m1 == m2:
                    continue
                k2 = (m2, m1)
                v2s = pair_dict.get(k2, None)
                if v2s is not None:
                    visited.add(k2)
                    v2, v1_ner_label = v2s
                    v2 = v2[: len(sym_labels)] + v2[num_label:] + v2[len(sym_labels): num_label]

                    for j in range(len(v2)):
                        v1[j] += v2[j]
                else:
                    assert (False)

                if v1_ner_label == 'NIL':
                    continue

                pred_label = np.argmax(v1)
                x= v1 / np.sum(v1)

                if pred_label > 0:
                    if pred_label >= num_label:
                        pred_label = pred_label - num_label + len(sym_labels)
                        m1, m2 = m2, m1
                        v1_ner_label, v2_ner_label = v2_ner_label, v1_ner_label

                    pred_score = x[pred_label]




                    sentence_results.append((pred_score, m1, m2, pred_label, v1_ner_label, v2_ner_label))
                    confidence_scores.append((pred_score, m1, m2, label_list[pred_label], v1_ner_label, v2_ner_label))
                    #confidence_scores.append(pred_score)


            sentence_results.sort(key=lambda x: -x[0])
            confidence_scores.sort(key=lambda x: -x[0])

            no_overlap = []


            def is_overlap(m1, m2):
                if m2[0] <= m1[0] and m1[0] <= m2[1]:
                    return True
                if m1[0] <= m2[0] and m2[0] <= m1[1]:
                    return True
                return False

            output_preds = []


            for item in sentence_results:
                m1 = item[1]
                m2 = item[2]
                overlap = False
                for x in no_overlap:
                    _m1 = x[1]
                    _m2 = x[2]
                    # same relation type & overlap subject & overlap object --> delete
                    if item[3] == x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)):
                        overlap = True
                        break

                pred_label = label_list[item[3]]

                if not overlap:
                    no_overlap.append(item)

            pos2ner = {}

            for item in no_overlap:
                m1 = item[1]
                m2 = item[2]
                pred_label = label_list[item[3]]
                tot_pred += 1
                # if pred_label in sym_labels:
                #     tot_pred += 1  # duplicate
                #     if (example_index, m1, m2, pred_label) in golden_labels or (
                #     example_index, m2, m1, pred_label) in golden_labels:
                #         cor += 2
                # else:
                #     if (example_index, m1, m2, pred_label) in golden_labels:
                #         cor += 1

                if m1 not in pos2ner:
                    pos2ner[m1] = item[4]
                if m2 not in pos2ner:
                    pos2ner[m2] = item[5]

                output_preds.append((m1, m2, pred_label))
                # if pred_label in sym_labels:
                #     if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]),
                #         pred_label) in golden_labels_withner \
                #             or (example_index, (m2[0], m2[1], pos2ner[m2]), (m1[0], m1[1], pos2ner[m1]),
                #                 pred_label) in golden_labels_withner:
                #         cor_with_ner += 2
                # else:
                #     if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]),
                #         pred_label) in golden_labels_withner:
                #         cor_with_ner += 1
            do_test=True
            if do_test:
                # output_w.write(json.dumps(output_preds) + '\n')
                tot_output_results[example_index[0]].append((example_index[1], output_preds))



            # refine NER results
            ner_results = list(global_predicted_ners[example_index])
            for i in range(len(ner_results)):
                start, end, label = ner_results[i]
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_ori_cor += 1
                if (start, end) in pos2ner:
                    label = pos2ner[(start, end)]
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_cor += 1
                ner_tot_pred += 1



    return tot_output_results, confidence_scores

def predict(sentence):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="data/json/", type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default="bertsub", type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="bert_model/", type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default="output_results/re-bert-42", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', default=False,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=True,
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',default=True,
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=5,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=2500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',default=True,
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--save_total_limit', type=int, default=1,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')

    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument('--max_pair_length', type=int, default=40, help="")
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--eval_logsoftmax', action='store_true',default=True)
    parser.add_argument('--eval_softmax', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--lminit', action='store_true')
    parser.add_argument('--no_sym', action='store_true', default=True)
    parser.add_argument('--att_left', action='store_true')
    parser.add_argument('--att_right', action='store_true')
    parser.add_argument('--use_ner_results', action='store_true', default=True)
    parser.add_argument('--use_typemarker', action='store_true')
    parser.add_argument('--eval_unidirect', action='store_true')

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.endswith("test"):
            return
        if not os.path.exists(path):
            os.mkdir(path)

        print('Experiment dir : {}'.format(path))
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    if args.do_train and args.local_rank in [-1, 0] and args.output_dir.find('test') == -1:
        create_exp_dir(args.output_dir, scripts_to_save=['run_re.py', 'transformers/src/transformers/modeling_bert.py',
                                                         'transformers/src/transformers/modeling_albert.py'])

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    if args.data_dir.find('ace') != -1:
        num_ner_labels = 8

        if args.no_sym:
            num_labels = 7 + 7 - 1
        else:
            num_labels = 7 + 7 - 2
    elif args.data_dir.find('scierc') != -1:
        num_ner_labels = 7

        if args.no_sym:
            num_labels = 8 + 8 - 1
        else:
            num_labels = 8 + 8 - 3
    else:
        num_ner_labels = 2

        if args.no_sym:
            num_labels = 14 + 14 - 1
        else:
            num_labels = 14 + 14 - 3

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.num_ner_labels = num_ner_labels

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.model_type.startswith('albert'):
        if args.use_typemarker:
            special_tokens_dict = {
                'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(num_ner_labels * 4 + 2)]}
        else:
            special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # print ('add tokens:', tokenizer.additional_special_tokens)
        # print ('add ids:', tokenizer.additional_special_tokens_ids)
        model.albert.resize_token_embeddings(len(tokenizer))

    if args.do_train:
        subject_id = tokenizer.encode('主', add_special_tokens=False)
        assert (len(subject_id) == 1)
        subject_id = subject_id[0]
        object_id = tokenizer.encode('宾', add_special_tokens=False)
        assert (len(object_id) == 1)
        object_id = object_id[0]

        mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)
        assert (len(mask_id) == 1)
        mask_id = mask_id[0]

        logger.info(" subject_id = %s, object_id = %s, mask_id = %s", subject_id, object_id, mask_id)

        if args.lminit:
            if args.model_type.startswith('albert'):
                word_embeddings = model.albert.embeddings.word_embeddings.weight.data
                subs = 30000
                sube = 30001
                objs = 30002
                obje = 30003
            else:
                word_embeddings = model.bert.embeddings.word_embeddings.weight.data
                subs = 1
                sube = 2
                objs = 3
                obje = 4

            word_embeddings[subs].copy_(word_embeddings[mask_id])
            word_embeddings[sube].copy_(word_embeddings[subject_id])

            word_embeddings[objs].copy_(word_embeddings[mask_id])
            word_embeddings[obje].copy_(word_embeddings[object_id])

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0


    # Evaluation
    results = {'dev_best_f1': best_f1}

    if args.do_eval and args.local_rank in [-1, 0]:


        checkpoints = [args.output_dir]

        WEIGHTS_NAME = 'pytorch_model.bin'

        if args.eval_all_checkpoints:

            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:

            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

            model = model_class.from_pretrained(checkpoint, config=config)

            model.to(args.device)
            result, scores = evaluate(sentence, args, model, tokenizer, prefix=global_step, do_test=not args.no_test)

    return result, scores

def spliteKeyWord(str):
    regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*|[^\w\s]"
    matches = re.findall(regex, str, re.UNICODE)
    return matches

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def pattern_index_broadcasting(all_data, search_data):
    n = len(search_data)
    all_data = np.asarray(all_data)
    all_data_2D = strided_app(np.asarray(all_data), n, S=1)
    return np.flatnonzero((all_data_2D == search_data).all(1))


def get_start_end(s,b):
    l = spliteKeyWord(s)

    results = []

    m = b
    out = list(pattern_index_broadcasting(l, m)[:,None] + np.arange(len(m)))

    if out != []:
        result = out[0]
        if len(result)==1:
            return [result[0],result[0],'entity'], l
        if len(result)>1:
            return [result[0], result[-1], 'entity'], l
    else:
        return [], l

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    # input = {"clusters": [[[17, 20], [23, 23]]], "sentences": [
    #     ["English", "is", "shown", "to", "be", "trans-context-free", "on", "the", "basis", "of", "coordinations", "of",
    #      "the", "respectively", "type", "that", "involve", "strictly", "syntactic", "cross-serial", "agreement", ".",
    #      "The", "agreement", "in", "question", "involves", "number", "in", "nouns", "and", "reflexive", "pronouns",
    #      "and", "is", "syntactic", "rather", "than", "semantic", "in", "nature", "because", "grammatical", "number",
    #      "in", "English", ",", "like", "grammatical", "gender", "in", "languages", "such", "as", "French", ",", "is",
    #      "partly", "arbitrary", ".", "The", "formal", "proof", ",", "which", "makes", "crucial", "use", "of", "the",
    #      "Interchange", "Lemma", "of", "Ogden", "et", "al.", ",", "is", "so", "constructed", "as", "to", "be", "valid",
    #      "even", "if", "English", "is", "presumed", "to", "contain", "grammatical", "sentences", "in", "which",
    #      "respectively", "operates", "across", "a", "pair", "of", "coordinate", "phrases", "one", "of", "whose",
    #      "members", "has", "fewer", "conjuncts", "than", "the", "other", ";", "it", "thus", "goes", "through",
    #      "whatever", "the", "facts", "may", "be", "regarding", "constructions", "with", "unequal", "numbers", "of",
    #      "conjuncts", "in", "the", "scope", "of", "respectively", ",", "whereas", "other", "arguments", "have",
    #      "foundered", "on", "this", "problem", "."]], "predicted_ner": [
    #     [[0, 0, "Material"], [10, 10, "OtherScientificTerm"], [17, 20, "OtherScientificTerm"], [23, 23, "Generic"],
    #      [29, 29, "OtherScientificTerm"], [31, 32, "OtherScientificTerm"], [42, 43, "OtherScientificTerm"],
    #      [45, 45, "Material"], [48, 49, "OtherScientificTerm"], [51, 51, "Material"], [54, 54, "Material"],
    #      [70, 71, "Method"], [86, 86, "Material"]]], "relations": [[], []], "doc_key": "J87-1003"}
    # input = {"sentences": [["招", "商", "轮", "船", ":", "本", "公", "司", "主", "营", "业", "务", "为", "远", "洋", "油", "轮", "及", "散", "货", "船", "运", "输", "，", "现", "有", "油", "轮", "1", "4", "艘", "，", "合", "计", "载", "重", "吨", "2", "5", "6", "万", "吨", "；", "散", "货", "船", "1", "4", "艘", "，", "合", "计", "载", "重", "吨", "7", "0", "万", "吨", "，", "分", "别", "由", "本", "公", "司", "全", "资", "拥", "有", "的", "两", "个", "专", "业", "管", "理", "公", "司", "海", "宏", "公", "司", "及", "香", "港", "明", "华", "进", "行", "日", "常", "经", "营", "管", "理", "。"]],
    #          "predicted_ner": [[[0, 3, "entity"], [13, 16, "entity"], [18, 22, "entity"]]], "relations": [[], []]}
    # input_json = json.dumps(input)
    # result_span, scores = predict(input_json)
    # print(result_span)
    # exit()

    ner_dict = defaultdict(list)


    sentence ="中国石油天然气股份:petrochina,中国石油股份"



    text = sentence
    sentence = spliteKeyWord(sentence)

    sentence_copy = deepcopy(sentence)

    print(sentence)
    input = {"clusters": [[], []], "sentences": [sentence
       ], "ner": [[], []], "relations": [[], []], "doc_key": "J87-1003"}

    input_json = json.dumps(input)
    pred_ner_span, span_ner_scores = predict_ner(input_json)



    pred_ner_span = [list(ele) for ele in pred_ner_span[0]]


    input['predicted_ner'] = [pred_ner_span]
    input_json = json.dumps(input)

    result_span, span_re_scores = predict(input_json)

    span_results = []

    for a, (b,c),(d,e),f,g,h in span_re_scores:
        span_results.append([''.join(sentence_copy[b:c + 1]), f,''.join(sentence_copy[d:e + 1]), a])

    # print(span_ner_scores)
    for a, (b,c), d in span_ner_scores:
        ner_dict[''.join(sentence_copy[b:c + 1])].append(a)
    # print(ner_dict)






    from bert import Ner
    model = Ner('/tmp/kashif/pytorch-bert-ner/output/')

    sentence = ' '.join(sentence)

    output, tokens, tags = model.predict(sentence)

    pos_tags = ['IN'] * len(tokens)
    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)
    original_text = []
    for subtree in ne_tree:
        # skipping 'O' tags
        if type(subtree) == Tree:
            original_label = subtree.label()
            # original_string = "".join([token for token, pos in subtree.leaves()])
            original_string = [token for token, pos in subtree.leaves()]
            original_text.append((original_string, original_label))
    pp = []
    for (a, b) in original_text:
        pp.append([a, b])
    pred_ner_BIO = []
    ner_bio_scores = []
    for x in pp:
        entity, _ = get_start_end(text, x[0])
        if entity != []:
            total_confidence = 0
            for index in range(entity[0], entity[1] + 1):
                confidence = output[index][tokens[index]]['confidence']
                total_confidence += confidence
            total_confidence = total_confidence / (entity[1] - entity[0] + 1)
            pred_ner_BIO.append([entity[0], entity[1], x[1]])
            ner_bio_scores.append([entity[0], entity[1], x[1], total_confidence])
        else:
            c=1


    input['predicted_ner'] = [pred_ner_BIO]
    input_json = json.dumps(input, cls=NpEncoder)

    result_BIO, BIO_re_scores = predict(input_json)

    BIO_results = []

    for a, (b, c), (d, e), f, g, h in BIO_re_scores:
        span_results.append([''.join(sentence_copy[b:c + 1]), f, ''.join(sentence_copy[d:e + 1]), a])


    for term in ner_bio_scores:
        ner_dict[''.join(sentence_copy[term[0]:term[1]+1])].append(term[3])
    # print(ner_bio_scores)
    # print(ner_dict)
    ner_dict_final = {}
    for key, value in ner_dict.items():
        ner_dict_final[key] = sum(value)/len(value)
    # print(ner_dict_final)
    #
    # exit()



    final_results = []
    for x in (span_results + BIO_results):
        if x not in final_results:
            final_results.append(x)
    print(final_results)
    print(ner_dict_final)

    print('SPAN')
    print(span_ner_scores)
    print(span_re_scores)
    print('BIO')
    print(ner_bio_scores)
    print(BIO_re_scores)






    # ordered = sorted(result_BIO.items(), key=operator.itemgetter(1))
    # relations_BIO = []
    # for (a,b) in ordered:
    #     x = b[0][1]
    #     for (s1,e1),(s2,e2),r in x:
    #         relations_BIO.append([''.join(sentence_copy[s1:e1+1]),r,''.join(sentence_copy[s2:e2+1])])
    #
    # for i in range(len(relations_BIO)):
    #     relations_BIO[i].append(BIO_re_scores[i])
    #
    #
    #
    #
    #
    #
    # ordered = sorted(result_span.items(), key=operator.itemgetter(1))
    # relations_span = []
    # for (a, b) in ordered:
    #     x = b[0][1]
    #     for (s1, e1), (s2, e2), r in x:
    #         relations_span.append([''.join(sentence_copy[s1:e1 + 1]), r,''.join(sentence_copy[s2:e2 + 1])])
    # for i in range(len(relations_span)):
    #     relations_span[i].append(span_re_scores[i])



    # print('Span SOP results:')
    #
    # print(pred_ner_span)
    # print(span_ner_scores)
    # print(relations_span)
    # print(span_re_scores)
    #
    # print('*'*20)
    #
    #
    # final_results = []
    # for x in (relations_span+relations_BIO):
    #     if x not in final_results:
    #         final_results.append(x)
    # print('BIO SOP results')
    # print(pred_ner_BIO)
    # print(ner_bio_scores)
    # print(relations_BIO)
    # print(BIO_re_scores)
    # print('*' * 20)
    #
    # print(final_results)





