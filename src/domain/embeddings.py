import os
import numpy as np

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from collections import defaultdict


class PretrainedEmbeddings:
    def __init__(self, save_path, emb_dim, input_path=None):
        self.emb_dim = emb_dim

        if not os.path.exists(save_path):
            self.load_from_input(input_path, save_path)
        else:
            self.load_from_saved(save_path)

        self._add_preprocessing_tokens()

    def load_from_input(self, input_path, save_path):
        self.model = KeyedVectors.load_word2vec_format(input_path)

        self.model.save(save_path)

    def load_from_saved(self, save_path):
        self.model = KeyedVectors.load(save_path)

    def _add_preprocessing_tokens(self):
        preprocessing_tokens = ['<BOS>', '<EOS>', '<UNK>', '<PAD>']

        init_vector_values = np.random.normal(
            size=(len(preprocessing_tokens), self.emb_dim))

        self.model.add(preprocessing_tokens, init_vector_values)

        self.preprocessing_idx = {}
        for token in preprocessing_tokens:
            self.preprocessing_idx[token] = self.model.vocab[token].index

    def word2index(self):
        word2index = defaultdict(lambda: self.preprocessing_idx['<UNK>'])
        word2index.update({token: token_index for token_index,
                           token in enumerate(self.model.index2word)})

        return word2index


class GloveEmbeddings(PretrainedEmbeddings):
    def load_from_input(self, input_path, save_path):
        glove_input_path = input_path + '.glove'
        glove2word2vec(input_path, glove_input_path)
        super().load_from_input(glove_input_path, save_path)
