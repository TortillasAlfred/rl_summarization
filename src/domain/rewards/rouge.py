# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROUGE Metric Implementation
This is a very slightly version of:
https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
---
ROUGe metric implementation.
This is a modified and slightly extended verison of
https://github.com/miso-belica/sumy/blob/dev/sumy/evaluation/rouge.py.
This is a modified version of 
https://github.com/pltrdy/rouge/blob/master/rouge/rouge_score.py.
"""
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
import itertools

from copy import deepcopy
import torch

from rouge import Rouge

Rouge.DEFAULT_STATS = ['f']


def preprocess_texts(texts, pad_idx, itos):
    texts = [t.tolist() if isinstance(t, torch.Tensor) else t for t in texts]
    texts = [[[w for w in s if w is not pad_idx] for s in text]
             for text in texts]
    texts = [[s for s in text if len(s) > 0] for text in texts]

    texts = [[[itos[w] for w in s] for s in t] for t in texts]
    texts = [' . '.join([' '.join(s) for s in t]) for t in texts]

    return texts


class RougeReward:
    def __init__(self, itos, pad_idx, avg=True):
        self.itos = itos
        self.pad_idx = pad_idx
        self.avg = avg
        self.scorer = Rouge()

    def __call__(self, hyps, refs, pad_idx, max_n=2):
        scores = []

        hyps = preprocess_texts(hyps, pad_idx, itos=self.itos)
        refs = preprocess_texts(refs, pad_idx, itos=self.itos)

        scores = self.scorer.get_scores(hyps, refs, avg=self.avg)
        scores = [[f for metric in s.values() for f in metric.values()]
                  for s in scores]

        scores = torch.tensor(scores)

        if self.avg:
            return scores.mean(dim=0)
        else:
            return scores

    @staticmethod
    def from_config(dataset, config):
        return RougeReward(dataset.itos, dataset.pad_idx, config['rouge_avg'])


if __name__ == '__main__':
    from src.domain.dataset import CnnDailyMailDataset
    import logging

    logging.info('Begin')

    dataset = CnnDailyMailDataset('./data/cnn_dailymail',
                                  'glove.6B.300d',
                                  dev=True,
                                  sets=['test'])

    test_loader = dataset.get_loaders(batch_size=5)['test']
    scorer = RougeReward()

    for batch in test_loader:
        contents, refs = batch
        l3 = contents[:, :3]

        n = len(batch)
        scores_ids = [(i, i + n) for i in range(n)]
        seqs = l3.tolist()
        seqs.extend(refs.tolist())

        logging.info(
            scorer(seqs, scores_ids=scores_ids, pad_idx=dataset.pad_idx))
