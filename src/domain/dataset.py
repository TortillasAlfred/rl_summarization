from src.domain.utils import datetime_tqdm

import os
import tarfile
import json
import logging
import pickle

from os.path import join
from torch.utils.data import Dataset
from collections import Counter


DEFAULT_MAX_SENT_LENGTH = 80
DEFAULT_MAX_SENT_NUMBER = 50


class SummarizationDataset(Dataset):
    def __init__(self, split, path, data_fetcher):
        super(SummarizationDataset, self).__init__()
        self.texts_fetcher = data_fetcher
        self.path = path
        self.split = split
        self.texts = self.texts_fetcher(self._files_path(), self.split)
        self._load_vocab()

    def preprocess(self, embeddings):
        embeddings.fit_to_vocab(self.vocab)
        self.texts = list(map(lambda text: text.preprocess(embeddings), datetime_tqdm(
            self.texts, desc='Preprocessing dataset texts...')))

    def _load_vocab(self):
        if not os.path.isfile(self._vocab_path()):
            logging.info(f'No saved vocabulary found in {self._vocab_path()}.')
            self.compute_save_vocab(self._vocab_path())

        with open(self._vocab_path(), 'rb') as f:
            self.vocab = pickle.load(f)

    def _files_path(self):
        return join(self.path, 'finished_files')

    def _vocab_path(self):
        return join(self._files_path(), 'vocab_cnt.pkl')

    def compute_save_vocab(self, save_path):
        if self.split is not 'train':
            raise ValueError(
                f'Vocabulary can only be computed for train split, not {self.split} !')

        logging.info('Building vocabulary...')

        vocab = Counter()
        for text in datetime_tqdm(self.texts, desc='Reading dataset texts...'):
            text_vocab = []
            for sentences in [text.abstract, text.content]:
                for sent in sentences:
                    text_vocab.extend(sent.split())

            vocab.update(text_vocab)

        logging.info(f'Vocabulary done. Saving to {save_path}')
        with open(save_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

        return vocab

    def __getitem__(self, index):
        return self.texts[index]

    def __len__(self):
        return len(self.texts)


class CnnDmDataset(SummarizationDataset):
    DEFAULT_CNN_DM_PATH = './data/cnn_dailymail/'

    def __init__(self, split, path=DEFAULT_CNN_DM_PATH):
        super(CnnDmDataset, self).__init__(split, path, cnn_dm_fetcher)


# Fetchers
def cnn_dm_fetcher(path, split):
    reading_path = join(path, split)

    if not os.path.isdir(reading_path):
        with tarfile.open(reading_path + '.tar') as tar:
            logging.info(
                f'Split {split} is not yet extracted to {reading_path}. Doing it now.')
            tar.extractall(path)

    all_articles = []

    for fname in datetime_tqdm(os.listdir(reading_path), desc='Reading dataset files...'):
        with open(join(reading_path, fname), 'r') as f:
            article_data = json.loads(f.read())
            article = CnnDmArticle(article_data)

            if article.is_valid:
                all_articles.append(article)

    return all_articles


class Text:
    def __init__(self, content, abstract, id):
        self.content = [sent.split() for sent in content]
        self.abstract = [sent.split() for sent in abstract]
        self.id = id
        self.is_valid = len(self.content) > 0 and len(self.abstract) > 0

    def preprocess(self, embeddings, apply_padding=True, max_sent_length=DEFAULT_MAX_SENT_LENGTH, max_sent_number=DEFAULT_MAX_SENT_NUMBER):
        if apply_padding:
            content_to_parse = self._apply_padding(
                self.content, max_sent_length, max_sent_number)
            abstract_to_parse = self._apply_padding(
                self.abstract, max_sent_length, max_sent_number, enclose=False)
        else:
            content_to_parse = self.content
            abstract_to_parse = self.abstract

        def convert_sent(sent):
            return list(map(lambda word: embeddings.find(word), sent))

        self.idx_content = list(map(convert_sent, content_to_parse))
        self.idx_abstract = list(map(convert_sent, abstract_to_parse))

        return self

    def _apply_padding(self, sents, max_sent_length, max_sent_number, enclose=True):
        sents = sents[:max_sent_number]

        if enclose:
            sents = [['<BOS>'] + s + ['<EOS>'] for s in sents]

        tokens_per_sent = min(max([len(s)
                                   for s in sents]), max_sent_length)
        sents = [s[:tokens_per_sent] for s in sents]
        return [s + (tokens_per_sent - len(s)) * ['<PAD>'] for s in sents]


class CnnDmArticle(Text):
    def __init__(self, json_data):
        super(CnnDmArticle, self).__init__(json_data['article'],
                                           json_data['abstract'],
                                           json_data['id'])


def build_dev_dataset(out_file='./data/cnn_dailymail/finished_files/dev.tar'):
    import tarfile
    import io

    cnn_dm_dataset = CnnDmDataset('val')

    with tarfile.open(out_file, 'w') as writer:
        for idx in range(100):
            article = cnn_dm_dataset[idx]
            js_example = {}
            js_example['id'] = article.id
            js_example['article'] = article.content
            js_example['abstract'] = article.abstract
            js_serialized = json.dumps(js_example, indent=4).encode()
            save_file = io.BytesIO(js_serialized)
            tar_info = tarfile.TarInfo('{}/{}.json'.format(
                os.path.basename(out_file).replace('.tar', ''), idx))
            tar_info.size = len(js_serialized)
            writer.addfile(tar_info, save_file)


def build_max_dataset(out_file='./data/cnn_dailymail/finished_files/max.tar'):
    import tarfile
    import io

    with tarfile.open(out_file, 'w') as writer:
        for idx in range(100):
            js_example = {}
            js_example['id'] = idx
            js_example['article'] = [['<PAD>'] *
                                     DEFAULT_MAX_SENT_LENGTH] * DEFAULT_MAX_SENT_NUMBER
            js_example['abstract'] = [['<PAD>'] * DEFAULT_MAX_SENT_LENGTH] * 3
            js_serialized = json.dumps(js_example, indent=4).encode()
            save_file = io.BytesIO(js_serialized)
            tar_info = tarfile.TarInfo('{}/{}.json'.format(
                os.path.basename(out_file).replace('.tar', ''), idx))
            tar_info.size = len(js_serialized)
            writer.addfile(tar_info, save_file)


if __name__ == '__main__':
    # build_dev_dataset()
    # build_max_dataset()

    from src.domain.utils import configure_logging
    from src.domain.embeddings import PretrainedEmbeddings

    configure_logging()

    emb = PretrainedEmbeddings('./data/embeddings/glove/glove.6B.50d.txt')

    cnn_dm_dataset = CnnDmDataset('dev')
    cnn_dm_dataset.preprocess(emb)
