import os
import numpy as np
import pickle
import logging

from collections import defaultdict


class PretrainedEmbeddings:
    def __init__(self, input_path):
        save_path = self._get_save_path(input_path)

        if not os.path.exists(save_path):
            self.load_from_input(input_path, save_path)
        else:
            self.load_from_saved(save_path)

        assert len(self.words_mapping) == len(self.word_vectors)
        assert len(self.idx_mapping) == len(self.word_vectors)

    def _get_save_path(self, input_path):
        output_dir = os.path.splitext(input_path)[0]
        output_dir = os.path.join(output_dir, "saved")

        return output_dir

    def load_from_input(self, input_path, save_path):
        logging.info(
            f"No presaved custom embeddings found. Reading input from {input_path}"
        )
        self.words_mapping = {}
        self.idx_mapping = {}

        self.word_vectors = np.loadtxt(
            strip_first_col(input_path), dtype=float, comments=None
        )
        self.emb_dim = self.word_vectors.shape[1]

        words = np.genfromtxt(input_path, dtype=str, usecols=0, comments=None)

        for idx, word in enumerate(words):
            self._add_token_idx_pair(word, idx)

        self._add_preprocessing_tokens()

        logging.info("Reading and preparation done.")
        self._save(save_path)

    def _save(self, save_path):
        logging.info(f"Saving custom embeddings to {save_path} folder.")
        os.makedirs(save_path)

        word_vectors_path = os.path.join(save_path, "word_vectors.npy")
        np.save(word_vectors_path, self.word_vectors)

        words_mapping_path = os.path.join(save_path, "words_mapping.pck")
        with open(words_mapping_path, "wb") as handle:
            pickle.dump(self.words_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

        idx_mapping_path = os.path.join(save_path, "idx_mapping.pck")
        with open(idx_mapping_path, "wb") as handle:
            pickle.dump(self.idx_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_saved(self, save_path):
        logging.info(f"Loading custom embeddings from folder {save_path}.")
        word_vectors_path = os.path.join(save_path, "word_vectors.npy")
        self.word_vectors = np.load(word_vectors_path)
        self.emb_dim = self.word_vectors.shape[1]

        words_mapping_path = os.path.join(save_path, "words_mapping.pck")
        with open(words_mapping_path, "rb") as handle:
            self.words_mapping = pickle.load(handle)

        idx_mapping_path = os.path.join(save_path, "idx_mapping.pck")
        with open(idx_mapping_path, "rb") as handle:
            self.idx_mapping = pickle.load(handle)

        self._init_preprocessing_attributes()

    def _add_token_idx_pair(self, token, idx):
        self.words_mapping[token] = idx
        self.idx_mapping[idx] = token

    def _add_preprocessing_tokens(self):
        preprocessing_tokens = ["<BOS>", "<EOS>", "<UNK>", "<PAD>"]

        init_vector_values = np.random.normal(
            size=(len(preprocessing_tokens) - 1, self.emb_dim), scale=0.1
        )
        init_vector_values = np.vstack(
            (init_vector_values, np.zeros((1, self.emb_dim)))
        )

        for token, word_vector in zip(preprocessing_tokens, init_vector_values):
            self._add_token_idx_pair(token, len(self.word_vectors))
            self.word_vectors = np.vstack((self.word_vectors, word_vector))

        self._init_preprocessing_attributes()

    def _init_preprocessing_attributes(self):
        self.bos = self.words_mapping["<BOS>"]
        self.eos = self.words_mapping["<EOS>"]
        self.unk = self.words_mapping["<UNK>"]
        self.pad = self.words_mapping["<PAD>"]

    def fit_to_vocab(self, vocab, return_unk_words=False):
        logging.info("Fitting embeddings to vocabulary...")
        vocab_uniques = set(vocab)
        embeddings_uniques = set(self.words_mapping.keys())

        retained_words = vocab_uniques & embeddings_uniques

        self._filter_embeddings(retained_words)

        if return_unk_words:
            return vocab_uniques - retained_words

    def _filter_embeddings(self, embeddings_to_keep):
        retained_indices = [self.words_mapping[word] for word in embeddings_to_keep]

        self.word_vectors = self.word_vectors[retained_indices]

        self.words_mapping = {}
        self.idx_mapping = {}

        for idx, word in enumerate(embeddings_to_keep):
            self._add_token_idx_pair(idx, word)

        self._add_preprocessing_tokens()

    def find(self, item):
        if isinstance(item, str):
            return self.words_mapping.get(item, self.unk)
        else:
            return self.idx_mapping[item]

    def __len__(self):
        return len(self.word_vectors)


def strip_first_col(fname, delimiter=None):
    with open(fname, "r") as fin:
        for line in fin:
            try:
                yield line.split(delimiter, 1)[1]
            except IndexError:
                continue


if __name__ == "__main__":
    emb = PretrainedEmbeddings("./data/embeddings/glove/glove.6B.50d.txt")

    import pickle

    with open("./data/cnn_dailymail/analysis/vocab_cnt.pkl", "rb") as f:
        vocab = pickle.load(f)

    emb.fit_to_vocab(vocab.keys())
