# -*- coding: utf-8 -*-

import random
from collections import namedtuple
from typing import Dict

import torch

from beaver.data.field import Field

Batch = namedtuple("Batch", ['src', 'tgt', 'batch_size'])
BatchVec = namedtuple("Batch", ['src', 'tgt', 'batch_size', 'en_vec'])
Example = namedtuple("Example", ['src', 'tgt'])
ExampleVec = namedtuple("ExampleVec", ['src', 'tgt', 'en_vec'])


class TranslationDataset(object):

    def __init__(self,
                 src_path: str,
                 tgt_path: str,
                 batch_size: int,
                 device: torch.device,
                 train: bool,
                 fields: Dict[str, Field],
                 en_vecs=None):

        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.fields = fields
        self.sort_key = lambda ex: (len(ex.src), len(ex.tgt))
        self.en_vecs = True if en_vecs is not None else False
        examples = []
    
        if en_vecs is None:
          for src_line, tgt_line in zip(read_file(src_path), read_file(tgt_path)):
              examples.append(Example(src_line, tgt_line))
        else:
          for src_line, tgt_line, en_vec in zip(read_file(src_path), read_file(tgt_path), en_vecs):
              examples.append(ExampleVec(src_line, tgt_line, en_vec))
          print(len(examples))
          print(len(en_vecs))
        examples, self.seed = self.sort(examples)

        self.num_examples = len(examples)
        self.batches = list(batch(examples, self.batch_size))

    def __iter__(self):
        while True:
            if self.train:
                random.shuffle(self.batches)
            for minibatch in self.batches:
                src = self.fields["src"].process([x.src for x in minibatch], self.device)
                tgt = self.fields["tgt"].process([x.tgt for x in minibatch], self.device)
                if self.en_vecs:
                  en_vec = torch.Tensor([x.en_vec for x in minibatch]).to(self.device)
                  yield BatchVec(src=src, tgt=tgt, batch_size=len(minibatch), en_vec=en_vec)
                else:
                  yield Batch(src=src, tgt=tgt, batch_size=len(minibatch))
            if not self.train:
                break

    def sort(self, examples):
        seed = sorted(range(len(examples)), key=lambda idx: self.sort_key(examples[idx]))
        return sorted(examples, key=self.sort_key), seed


def read_file(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()


def batch(data, batch_size):
    minibatch, cur_len = [], 0
    for ex in data:
        minibatch.append(ex)
        cur_len = max(cur_len, len(ex.src), len(ex.tgt))
        if cur_len * len(minibatch) > batch_size:
            yield minibatch[:-1]
            minibatch, cur_len = [ex], max(len(ex.src), len(ex.tgt))
    if minibatch:
        yield minibatch

