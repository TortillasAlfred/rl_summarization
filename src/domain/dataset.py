from src.domain.utils import datetime_tqdm

import tarfile
import io
import os
import logging
import json

from collections import OrderedDict
from torchtext.data import Dataset, Example, RawField, Field, NestedField, BucketIterator


class SummarizationDataset(Dataset):
    def __init__(self,
                 subsets,
                 fields,
                 vectors,
                 vectors_cache,
                 filter_pred=None):
        self.subsets = OrderedDict([(key, Dataset(subset, fields, filter_pred))
                                    for key, subset in subsets])
        self._build_vocabs(vectors, vectors_cache)

    def _build_vocabs(self):
        raise NotImplementedError()

    def get_loaders(self,
                    batch_sizes=None,
                    batch_size=1,
                    shuffle=True,
                    sort=False,
                    device=None):
        loaders = BucketIterator.splits(tuple(self.subsets.values()),
                                        batch_sizes=batch_sizes,
                                        batch_size=batch_size,
                                        device=device,
                                        sort=sort,
                                        shuffle=shuffle)

        return dict(zip(self.subsets.keys(), loaders))


def not_empty_example(example):
    return not (len(example.content) == 0 or len(example.abstract) == 0)


class CnnDailyMailDataset(SummarizationDataset):
    def __init__(self,
                 path,
                 vectors,
                 *,
                 sets=['train', 'val', 'test'],
                 dev=False,
                 vectors_cache='./data/embeddings',
                 max_tokens_per_sent=80,
                 max_sents_per_article=50,
                 filter_pred=not_empty_example):
        self.path = path
        self.max_tokens_per_sent = max_tokens_per_sent
        self.max_sents_per_article = max_sents_per_article
        self._build_reading_fields()
        subsets = self._load_all(sets, dev)
        super(CnnDailyMailDataset,
              self).__init__(subsets, self.fields, vectors, vectors_cache,
                             filter_pred)

    def _build_reading_fields(self):
        self.raw_content = RawField()
        self.raw_abstract = RawField(is_target=True)
        self.content = NestedField(Field())
        self.abstract = NestedField(Field())
        self.abstract.is_target = True

        self.fields = {
            'article': [('raw_content', self.raw_content),
                        ('content', self.content)],
            'abstract': [('raw_abstract', self.raw_abstract),
                         ('abstract', self.abstract)]
        }

    def _load_all(self, sets, dev):
        if 'train' in sets and sets.index('train') != 0:
            raise ValueError(
                'If loading the training dataset, it must be first in the sets list.'
            )

        loaded_sets = []

        for split in sets:
            loaded_sets.append((split, self._load_split(split, dev)))

        self.fields = [
            field for field_list in self.fields.values()
            for field in field_list
        ]

        return loaded_sets

    def _load_split(self, split, dev):
        finished_path = os.path.join(self.path, 'finished_files')
        reading_path = os.path.join(finished_path, split)

        if not os.path.isdir(reading_path):
            with tarfile.open(reading_path + '.tar') as tar:
                logging.info(
                    f'Split {split} is not yet extracted to {reading_path}. Doing it now.'
                )
                tar.extractall(finished_path)

        all_articles = []
        all_files = os.listdir(reading_path)

        if dev:
            all_files = all_files[:100]

        for fname in datetime_tqdm(all_files,
                                   desc=f'Reading {split} files...'):
            with open(os.path.join(reading_path, fname), 'r') as data:
                all_articles.append(Example.fromJSON(data.read(), self.fields))

        return all_articles

    def _build_vocabs(self, vectors, vectors_cache):
        logging.info('Building vocab from the whole dataset.')

        self.content.build_vocab([
            field for subset in self.subsets.values()
            for field in [subset.content, subset.abstract]
        ],
                                 vectors=vectors,
                                 vectors_cache=vectors_cache)

        self.abstract.vocab = self.content.vocab
        self.abstract.nesting_field.vocab = self.content.vocab
        self.pad_idx = self.content.vocab.stoi['<pad>']
        self.itos = self.content.vocab.itos
        self.stoi = self.content.vocab.stoi



if __name__ == '__main__':
    from src.domain.utils import configure_logging
    configure_logging()

    logging.info('Begin')

    dataset = CnnDailyMailDataset('./data/cnn_dailymail',
                                  'glove.6B.300d',
                                  dev=True)

    train_loader = dataset.get_loaders(device='cuda:0')['train']

    for batch in train_loader:
        print('hihi')
