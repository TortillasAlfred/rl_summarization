from src.domain.utils import datetime_tqdm

import tarfile
import io
import os
import logging
import json
import torch
from collections import defaultdict

from joblib import Parallel, delayed
from collections import OrderedDict
from torchtext.data import (
    Dataset,
    Example,
    RawField,
    Field,
    NestedField,
)


class SummarizationDataset(Dataset):
    def __init__(self, subsets, fields, vectors, vectors_cache, filter_pred=None):
        self.subsets = {
            key: Dataset(subset, fields, filter_pred) for key, subset in subsets
        }
        self._build_vocabs(vectors, vectors_cache)

    def _build_vocabs(self):
        raise NotImplementedError()

    def get_splits(self):
        return {name: TextDataset(dataset) for name, dataset in self.subsets.items()}


def not_empty_example(example):
    return len(example.content) > 3 and len(example.abstract) > 0


class CnnDailyMailDataset(SummarizationDataset):
    def __init__(
        self,
        path,
        vectors,
        *,
        sets=["train", "val", "test"],
        begin_idx=None,
        end_idx=None,
        dev=False,
        vectors_cache="./data/embeddings",
        filter_pred=not_empty_example,
    ):
        self.path = path
        self._build_reading_fields()
        subsets, self.fpaths = self._load_all(sets, dev, begin_idx, end_idx)
        super(CnnDailyMailDataset, self).__init__(
            subsets, self.fields, vectors, vectors_cache, filter_pred
        )

    def _build_reading_fields(self):
        self.raw_content = RawField()
        self.id = RawField()
        self.raw_abstract = RawField(is_target=True)
        self.content = NestedField(Field(fix_length=80), fix_length=50)
        self.abstract = NestedField(Field())
        self.abstract.is_target = True

        self.fields = {
            "article": [("raw_content", self.raw_content), ("content", self.content)],
            "abstract": [
                ("raw_abstract", self.raw_abstract),
                ("abstract", self.abstract),
            ],
            "id": [("id", self.id)],
        }

    def _load_all(self, sets, dev, begin_idx, end_idx):
        if "train" in sets and sets.index("train") != 0:
            raise ValueError(
                "If loading the training dataset, it must be first in the sets list."
            )

        loaded_sets = []
        paths = {}

        for split in sets:
            articles, fpaths = self._load_split(split, dev, begin_idx, end_idx)
            loaded_sets.append((split, articles))
            paths[split] = fpaths

        self.fields = [
            field for field_list in self.fields.values() for field in field_list
        ]

        return loaded_sets, paths

    def _load_split(self, split, dev, begin_idx, end_idx):
        finished_path = os.path.join(self.path, "finished_files")
        reading_path = os.path.join(finished_path, split)

        if not os.path.isdir(reading_path):
            with tarfile.open(reading_path + ".tar") as tar:
                logging.info(
                    f"Split {split} is not yet extracted to {reading_path}. Doing it now."
                )
                tar.extractall(finished_path)

        all_articles = []
        all_paths = []
        all_files = os.listdir(reading_path)

        if dev:
            all_files = all_files[:25000]
        elif begin_idx is not None and end_idx is not None:
            all_files = all_files[begin_idx:end_idx]

        exs_and_paths = Parallel(n_jobs=-1)(
            delayed(load_fname)(fname, reading_path, self.fields)
            for fname in datetime_tqdm(all_files, desc=f"Reading {split} files...")
        )

        all_articles, all_paths = zip(*exs_and_paths)

        return list(all_articles), list(all_paths)

    def _build_vocabs(self, vectors, vectors_cache):
        logging.info("Building vocab from the whole dataset.")

        self.content.build_vocab(
            [
                field
                for subset in self.subsets.values()
                for field in [subset.content, subset.abstract]
            ],
            vectors=vectors,
            vectors_cache=vectors_cache,
        )

        self.abstract.vocab = self.content.vocab
        self.abstract.nesting_field.vocab = self.content.vocab
        self.pad_idx = self.content.vocab.stoi["<pad>"]
        self.embedding_dim = self.content.vocab.vectors.shape[1]
        self.itos = self.content.vocab.itos
        self.stoi = self.content.vocab.stoi
        self.vocab = self.content.vocab

    @staticmethod
    def from_config(config):
        return CnnDailyMailDataset(
            config.data_path,
            config.embeddings,
            sets=config.sets.split("-"),
            dev=config.dev,
            vectors_cache=config.embeddings_location,
        )


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.examples = dataset.examples
        self.fields = dataset.fields

    def subset(self, n):
        self.examples = self.examples[:n]

    def __getitem__(self, i):
        return self.__process_example(self.examples[i])

    def __process_example(self, x):
        return {name: f.preprocess(getattr(x, name)) for name, f in self.fields.items()}

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2 ** 32

    def __iter__(self):
        for x in self.examples:
            yield self.__process_example(x)


def load_fname(fname, reading_path, fields):
    fpath = os.path.join(reading_path, fname)
    with open(fpath, "r") as data:
        ex = Example.fromJSON(data.read(), fields)
    return (ex, fpath)


if __name__ == "__main__":
    from src.domain.utils import configure_logging

    configure_logging()

    logging.info("Begin")

    dataset = CnnDailyMailDataset(
        "./data/cnn_dailymail", "glove.6B.100d", dev=True, sets=["train"]
    )

    train_set = dataset.get_splits()["train"]

    train_loader = BucketIterator(train_set, train=True, batch_size=10, device="cpu")

    for batch in train_loader:
        print("hihi")
