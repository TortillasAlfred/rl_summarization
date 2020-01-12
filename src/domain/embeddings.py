import os
import numpy as np
import pickle

from collections import defaultdict


class PretrainedEmbeddings:
    def __init__(self, input_path):
        save_path = self._get_save_path(input_path)

        if not os.path.exists(save_path):
            self.load_from_input(input_path, save_path)
        else:
            self.load_from_saved(save_path)

    def _get_save_path(self, input_path):
        output_dir = os.path.splitext(input_path)[0]
        output_dir = os.path.join(output_dir, 'saved')

        return output_dir

    def load_from_input(self, input_path, save_path):
        self.mapping = {}

        embeddings = np.loadtxt(input_path, dtype='str', comments=None)

        for idx, word in enumerate(embeddings[:, 0]):
            self._add_token_idx_pair(word, idx)

        self.word_vectors = embeddings[:, 1:].astype('float')
        self.emb_dim = self.word_vectors.shape[1]

        self._add_preprocessing_tokens()

        self._save(save_path)

    def _save(self, save_path):
        os.makedirs(save_path)

        word_vectors_path = os.path.join(save_path, 'word_vectors.npy')
        np.save(word_vectors_path, self.word_vectors)

        mapping_path = os.path.join(save_path, 'mapping.pck')
        with open(mapping_path, 'wb') as handle:
            pickle.dump(self.mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_saved(self, save_path):
        word_vectors_path = os.path.join(save_path, 'word_vectors.npy')
        self.word_vectors = np.load(word_vectors_path)
        self.emb_dim = self.word_vectors.shape[1]

        mapping_path = os.path.join(save_path, 'mapping.pck')
        with open(mapping_path, 'rb') as handle:
            self.mapping = pickle.load(handle)

        self._init_preprocessing_attributes()

    def _add_token_idx_pair(self, token, idx):
        self.mapping[token] = idx
        self.mapping[idx] = token

    def _add_preprocessing_tokens(self):
        preprocessing_tokens = ['<BOS>', '<EOS>', '<UNK>', '<PAD>']

        init_vector_values = np.random.normal(
            size=(len(preprocessing_tokens) - 1, self.emb_dim), scale=0.1)
        init_vector_values = np.vstack(
            (init_vector_values, np.zeros((1, self.emb_dim))))

        for token, word_vector in zip(preprocessing_tokens, init_vector_values):
            self._add_token_idx_pair(token, len(self.word_vectors))
            self.word_vectors = np.vstack((self.word_vectors, word_vector))

        self._init_preprocessing_attributes()

    def _init_preprocessing_attributes(self):
        self.bos = self.mapping['<BOS>']
        self.eos = self.mapping['<EOS>']
        self.unk = self.mapping['<UNK>']
        self.pad = self.mapping['<PAD>']

    def find(self, item):
        if isinstance(item, str):
            return self.mapping.get(item, self.unk)
        else:
            return self.mapping[item]


if __name__ == '__main__':
    emb = PretrainedEmbeddings('./data/embeddings/glove/glove.6B.50d.txt')
