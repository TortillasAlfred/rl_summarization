from collections import OrderedDict, Counter
from scipy.sparse import dok_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np


class TextDataCollator:
    def __init__(self, fields, reward_builder, subset):
        self.fields = fields
        self.reward_builder = reward_builder
        self.subset = subset

    def __call__(self, data):
        batch = {name: f.process([d[name] for d in data]) for name, f in self.fields}

        batch["scorers"] = get_reward_scorers(
            self.reward_builder, batch["id"], self.subset
        )

        return batch


def get_reward_scorers(reward_builder, ids, subset):
    if subset in ["train", "val", "test"]:
        return [reward_builder.init_scorer(id, subset) for id in ids]
    # elif subset in ["val", "test"]:
    #     return [RougePythonReward() for _ in range(batch_size)]
    else:
        raise ValueError(
            f'Bad subset : {subset}. Should be one of ["train", "val", "test].'
        )


def get_ngrams_dense(
    doc_contents,
    raw_doc_contents,
    pad_idx,
    use_torchtext=True,
    normalize=True,
    n=2,
    add_bias=False,
    pca_dim=50,
    unif_norm=True,
):
    if use_torchtext:
        sents = doc_contents
        n_grams_per_sent = [
            list(get_ngrams([w for w in sent if w != pad_idx], n=n)) for sent in sents
        ]
    else:
        sents = [_.split() for _ in raw_doc_contents]
        n_grams_per_sent = [list(get_ngrams([w for w in sent], n=n)) for sent in sents]

    n_grams_per_sent = [l for l in n_grams_per_sent if len(l) > 0]
    n_grams_dict = OrderedDict()
    all_ngrams = set([ngram for sent in n_grams_per_sent for ngram in sent])
    for n_gram in all_ngrams:
        n_grams_dict[n_gram] = len(n_grams_dict)
    sent_n_grams = [
        Counter([n_grams_dict[ngram] for ngram in sent]) for sent in n_grams_per_sent
    ]
    if add_bias:
        ngrams_sparse = dok_matrix(
            (len(n_grams_per_sent), len(all_ngrams) + 1), dtype=np.float32
        )
        for sent_i, n_grams in enumerate(sent_n_grams):
            count_sent = sum(n_grams.values())
            for n_gram_i, count_i in n_grams.items():
                if normalize:
                    ngrams_sparse[(sent_i, n_gram_i)] = count_i / count_sent
                else:
                    ngrams_sparse[(sent_i, n_gram_i)] = count_i
                ngrams_sparse[(sent_i, len(all_ngrams))] = 1
    else:
        ngrams_sparse = dok_matrix(
            (len(n_grams_per_sent), len(all_ngrams)), dtype=np.float32
        )
        for sent_i, n_grams in enumerate(sent_n_grams):
            count_sent = sum(n_grams.values())
            for n_gram_i, count_i in n_grams.items():
                if normalize:
                    ngrams_sparse[(sent_i, n_gram_i)] = count_i / count_sent
                else:
                    ngrams_sparse[(sent_i, n_gram_i)] = count_i

    return reduce_dim(ngrams_sparse, pca_dim, unif_norm)


def get_ngrams(token_list, n=2):
    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])

    for x in token_list:
        yield (x,)
    for n in range(2, n + 1):
        for x in _get_ngrams(n):
            yield x


def reduce_dim(ngrams, pca_dim=50, unif_norm=True):
    ngrams = TruncatedSVD(n_components=pca_dim).fit_transform(ngrams)

    if unif_norm:
        ngrams /= np.linalg.norm(ngrams, axis=-1, keepdims=True).max()
    else:
        ngrams /= np.linalg.norm(ngrams, axis=-1, keepdims=True)

    return ngrams
