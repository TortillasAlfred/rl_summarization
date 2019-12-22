import os
import tarfile
import json

from os.path import join
from torch.utils.data import DataLoader


class SummarizationDataset:
    def __init__(self, split, path, data_fetcher):
        self.data_fetcher = data_fetcher
        self.path = path
        self.split = split

        self._build()

    def _build(self):
        self.data = self.data_fetcher(self.path, self.split)

        self._pad()

        # TODO: Add preprocessing to get from lists of words to integer tensors from embedding layer

    def _pad(self):
        self.data = list(map(lambda article: (list(map(
            lambda sent: '<BOS> ' + sent + ' <EOS>', article[0])), article[1]), self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CnnDmDataset(SummarizationDataset):
    DEFAULT_CNN_DM_PATH = './data/finished_files/'

    def __init__(self, split, path=DEFAULT_CNN_DM_PATH):
        super(CnnDmDataset, self).__init__(split, path, cnn_dm_fetcher)


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
    cnn_dm_dataset = CnnDmDataset('test')
