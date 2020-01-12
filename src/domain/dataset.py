import os
import tarfile
import json

from os.path import join
from torch.utils.data import Dataset


DEFAULT_MAX_SENT_LENGTH = 80
DEFAULT_MAX_SENT_NUMBER = 50


class SummarizationDataset(Dataset):
    def __init__(self, split, path, data_fetcher):
        super(SummarizationDataset, self).__init__()
        self.texts_fetcher = data_fetcher
        self.path = path
        self.split = split
        self.texts = self.texts_fetcher(self.path, self.split)

    def preprocess(self, word2idx):
        self.texts = list(
            map(lambda text: text.preprocess(word2idx), self.texts))

    def __getitem__(self, index):
        return self.texts[index]

    def __len__(self):
        return len(self.texts)


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
                article_data = json.load(sample_file)

                all_articles.append((CnnDmArticle(article_data)))

    return all_articles


class Text:
    def __init__(self, content, abstract, id):
        self.content = [sent.split() for sent in content]
        self.abstract = [sent.split() for sent in abstract]
        self.id = id

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


def build_dev_dataset(out_file='./data/finished_files/dev.tar'):
    import tarfile
    import io

    cnn_dm_dataset = CnnDmDataset('val')

    with tarfile.open(out_file, 'w') as writer:
        for idx in range(100):
            article = cnn_dm_dataset[idx]
            js_example = {}
            js_example['id'] = article.id
            js_example['article'] = [' '.join(sent)
                                     for sent in article.content]
            js_example['abstract'] = [
                ' '.join(sent) for sent in article.abstract]
            js_serialized = json.dumps(js_example, indent=4).encode()
            save_file = io.BytesIO(js_serialized)
            tar_info = tarfile.TarInfo('{}/{}.json'.format(
                os.path.basename(out_file).replace('.tar', ''), idx))
            tar_info.size = len(js_serialized)
            writer.addfile(tar_info, save_file)


def build_max_dataset(out_file='./data/finished_files/max.tar'):
    import tarfile
    import io

    with tarfile.open(out_file, 'w') as writer:
        for idx in range(100):
            js_example = {}
            js_example['id'] = idx
            js_example['article'] = [
                ' '.join(['<PAD>'] * DEFAULT_MAX_SENT_LENGTH)] * DEFAULT_MAX_SENT_NUMBER
            js_example['abstract'] = [
                ' '.join(['<PAD>'] * DEFAULT_MAX_SENT_LENGTH)] * 3
            js_serialized = json.dumps(js_example, indent=4).encode()
            save_file = io.BytesIO(js_serialized)
            tar_info = tarfile.TarInfo('{}/{}.json'.format(
                os.path.basename(out_file).replace('.tar', ''), idx))
            tar_info.size = len(js_serialized)
            writer.addfile(tar_info, save_file)


if __name__ == '__main__':
    # build_dev_dataset()
    # build_max_dataset()

    from embeddings import PretrainedEmbeddings

    emb = PretrainedEmbeddings('./data/embeddings/glove/glove.6B.50d.txt')

    cnn_dm_dataset = CnnDmDataset('dev')
    cnn_dm_dataset.preprocess(emb)
