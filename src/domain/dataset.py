import os
import tarfile
import json

from os.path import join
from torch.utils.data import Dataset


class SummarizationDataset(Dataset):
    def __init__(self, split, path, data_fetcher, word2idx):
        self.texts_fetcher = data_fetcher
        self.path = path
        self.split = split
        self.word2idx = word2idx

        self._build()

    def _build(self):
        self.texts = self.texts_fetcher(self.path, self.split)

        self._preprocess()

    def _preprocess(self):
        # map(lambda text: text.preprocess(self.word2idx), self.texts)
        for text in self.texts:
            text.preprocess(self.word2idx)

    def __getitem__(self, index):
        return self.texts[index]

    def __len__(self):
        return len(self.texts)


class CnnDmDataset(SummarizationDataset):
    DEFAULT_CNN_DM_PATH = './data/finished_files/'

    def __init__(self, split, word2idx, path=DEFAULT_CNN_DM_PATH):
        super(CnnDmDataset, self).__init__(
            split, path, cnn_dm_fetcher, word2idx)


# Fetchers
def cnn_dm_fetcher(path, split):
    reading_path = join(path, split) + '.tar'

    all_articles = []

    with tarfile.open(reading_path) as all_samples_file:
        for sample_path in all_samples_file:
            with all_samples_file.extractfile(sample_path) as sample_file:
                article_data = json.load(sample_file)

                all_articles.append((CnnDmArticle(article_data)))

    return all_articles


class Text:
    def __init__(self, content, abstract, id):
        self.content = content
        self.abstract = abstract
        self.id = id

    def preprocess(self, word2idx, apply_padding=True, max_sent_length=80, max_sent_number=50):
        if apply_padding:
            content_to_parse = self._apply_padding(
                self.content, max_sent_length, max_sent_number)
            abstract_to_parse = self._apply_padding(
                self.abstract, max_sent_length, max_sent_number, enclose=False)
        else:
            content_to_parse = self.content
            abstract_to_parse = self.abstract

        def convert_sent(sent):
            return list(map(lambda word: word2idx[word], sent.split()))

        self.idx_content = list(map(convert_sent, content_to_parse))
        self.idx_abstract = list(map(convert_sent, abstract_to_parse))

    def _apply_padding(self, sents, max_sent_length, max_sent_number, enclose=True):
        sents = sents[:max_sent_number]

        if enclose:
            sents = ['<BOS> ' + s + ' <EOS>' for s in sents]

        sents = [s.split() for s in sents]
        tokens_per_sent = min(max([len(s)
                                   for s in sents]), max_sent_length)
        sents = [s[:tokens_per_sent] for s in sents]
        sents = [s + (tokens_per_sent - len(s)) * ['<PAD>'] for s in sents]
        return [' '.join(s) for s in sents]


class CnnDmArticle(Text):
    def __init__(self, json_data):
        super().__init__(json_data['article'],
                         json_data['abstract'],
                         json_data['id'])


if __name__ == '__main__':
    from embeddings import GloveEmbeddings

    emb = GloveEmbeddings('./data/embeddings/glove/glove.6B.50d.bin', 50,
                          './data/embeddings/glove/glove.6B.50d.txt')

    cnn_dm_dataset = CnnDmDataset('test', emb.word2index())
