from src.domain.rewards.rouge_python import RougePythonReward
from src.domain.dataset_bert import MIN_NUM_SEN_PER_DOCUMENT

import os
import tarfile
import logging
import numpy as np

from collections import OrderedDict, Counter
from scipy.sparse import dok_matrix
from sklearn.decomposition import TruncatedSVD


class BertTextDataCollator:
    def __init__(self, fields, reward_builder, subset, n_grams_loader=None):
        self.fields = fields
        self.reward_builder = reward_builder
        self.subset = subset
        self.n_grams_loader = n_grams_loader

    def __call__(self, data):
        data_filtered = [
            x for x in data if x["content"] is not None and sum(x["content"]["mark_clss"]) >= MIN_NUM_SEN_PER_DOCUMENT
        ]

        batch = {}
        for name, f in self.fields:
            data_for_current_field = []
            for d in data_filtered:
                data_for_current_field.append(d[name])
            batch[name] = f.process(data_for_current_field)

        if self.reward_builder:
            batch["scorers"] = get_reward_scorers(self.reward_builder, batch["id"], self.subset)

        if self.n_grams_loader:
            batch["ngrams_dense"] = [self.n_grams_loader(id, self.subset) for id in batch["id"]]

        return list(batch.values())


class TextDataCollator:
    def __init__(self, fields, reward_builder, subset, n_grams_loader=None):
        self.fields = fields
        self.reward_builder = reward_builder
        self.subset = subset
        self.n_grams_loader = n_grams_loader

    def __call__(self, data):
        batch = {name: f.process([d[name] for d in data]) for name, f in self.fields}

        if self.reward_builder:
            batch["scorers"] = get_reward_scorers(self.reward_builder, batch["id"], self.subset)

        if self.n_grams_loader:
            batch["ngrams_dense"] = [self.n_grams_loader(id, self.subset) for id in batch["id"]]

        return list(batch.values())


def get_reward_scorers(reward_builder, ids, subset):
    if subset in ["train"]:
        return [reward_builder.init_scorer(id, subset) for id in ids]
    elif subset in ["val", "test"]:
        return RougePythonReward()
    else:
        raise ValueError(f'Bad subset : {subset}. Should be one of ["train", "val", "test].')


class NGRAMSLoader:
    def __init__(self, base_path):
        self.base_path = os.path.join(base_path, "pca")

        if not os.path.isdir(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)
            if self.base_path[-1] == "/":
                reading_path = self.base_path[:-1] + ".tar"
            else:
                reading_path = self.base_path + ".tar"
            with tarfile.open(reading_path) as tar:
                logging.info(f"PCA vectors not yet extracted to {self.base_path} folder. Doing it now.")
                tar.extractall(base_path)

    def __call__(self, id, subset):
        return np.load(os.path.join(self.base_path, subset, f"{id}.npy"))


class NGRAMSSaver:
    def __init__(self, base_path, subset, pad_idx=1):
        self.base_path = base_path
        self.subset = subset
        self.pad_idx = pad_idx

    def __call__(self, doc_contents, id):
        if not os.path.isfile(os.path.join(self.base_path, self.subset, f"{id}.npy")):
            n_grams_dense = get_ngrams_dense(doc_contents, self.pad_idx)
            np.save(os.path.join(self.base_path, self.subset, f"{id}.npy"), n_grams_dense)


def get_ngrams_dense(doc_contents, pad_idx, n=2):
    sents = doc_contents
    n_grams_per_sent = [list(get_ngrams([w for w in sent if w != pad_idx], n=n)) for sent in sents]

    n_grams_per_sent = [l for l in n_grams_per_sent if len(l) > 0]
    n_grams_dict = OrderedDict()
    all_ngrams = set([ngram for sent in n_grams_per_sent for ngram in sent])
    for n_gram in all_ngrams:
        n_grams_dict[n_gram] = len(n_grams_dict)
    sent_n_grams = [Counter([n_grams_dict[ngram] for ngram in sent]) for sent in n_grams_per_sent]
    ngrams_sparse = dok_matrix((len(n_grams_per_sent), len(all_ngrams)), dtype=np.float64)
    for sent_i, n_grams in enumerate(sent_n_grams):
        count_sent = sum(n_grams.values())
        for n_gram_i, count_i in n_grams.items():
            ngrams_sparse[(sent_i, n_gram_i)] = count_i / count_sent

    return reduce_dim(ngrams_sparse)


def get_ngrams(token_list, n=2):
    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])

    for x in token_list:
        yield (x,)
    for n in range(2, n + 1):
        for x in _get_ngrams(n):
            yield x


def reduce_dim(ngrams, pca_dim=50, n_iter=25):
    if pca_dim < ngrams.shape[1]:
        ngrams = TruncatedSVD(n_components=pca_dim, n_iter=n_iter).fit_transform(ngrams)
    else:
        ngrams = ngrams.todense()
    ngrams /= np.linalg.norm(ngrams, axis=-1, keepdims=True)
    ngrams = ngrams.astype(np.float32)

    return ngrams
