import os
import tarfile
import json

from os.path import join
from torch.utils.data import DataLoader


class SummarizationDataset:
    def __init__(self, split, path, data_fetcher, word2idx):
        self.data_fetcher = data_fetcher
        self.path = path
        self.split = split
        self.word2idx = word2idx

        self._build()

    def _build(self):
        self.data = self.data_fetcher(self.path, self.split)

        self._convert_to_idx()

    def _convert_to_idx(self):
        def convert_sent(sent):
            return list(map(lambda word: self.word2idx[word], sent.split()))

        def convert_enclose_sent(sent):
            return convert_sent('<BOS> ' + sent + ' <EOS>')

        def convert_article(article):
            return list(map(convert_enclose_sent, article[0])), list(map(convert_sent, article[1]))

        self.data = list(map(convert_article, self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


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
                article = json.load(sample_file)

                all_articles.append((article['article'], article['abstract']))

    return all_articles


if __name__ == '__main__':
    from embeddings import GloveEmbeddings

    emb = GloveEmbeddings('./data/embeddings/glove/glove.6B.50d.bin', 50,
                          './data/embeddings/glove/glove.6B.50d.txt')

    cnn_dm_dataset = CnnDmDataset('test', emb.word2index())
