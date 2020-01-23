from src.domain.utils import configure_logging, datetime_tqdm

import tarfile
import io
import os
import logging
import json

from torchtext.data import Dataset, Example, Field, RawField, NestedField, BucketIterator

configure_logging()


class SummarizationDataset(Dataset):
    def __init__(self,
                 train,
                 val,
                 test,
                 fields,
                 vectors,
                 vectors_cache,
                 filter_pred=None):
        self.train = Dataset(train, fields, filter_pred)
        self.val = Dataset(val, fields, filter_pred)
        self.test = Dataset(test, fields, filter_pred)
        self._build_vocabs(vectors, vectors_cache)

    def _build_vocabs(self):
        raise NotImplementedError()

    def get_loaders(self, batch_sizes=None, batch_size=1, device=None):
        return BucketIterator.splits((self.train, self.val, self.test),
                                     batch_sizes=batch_sizes,
                                     batch_size=batch_size,
                                     device=device)


def not_empty_example(example):
    return not (len(example.content) == 0 or len(example.abstract) == 0)


class CnnDailyMailDataset(SummarizationDataset):
    def __init__(self,
                 path,
                 vectors,
                 *,
                 dev=False,
                 vectors_cache='./data/embeddings',
                 max_tokens_per_sent=80,
                 max_sents_per_article=50,
                 filter_pred=not_empty_example):
        self.path = path
        self._build_fields()
        self.max_tokens_per_sent = max_tokens_per_sent
        self.max_sents_per_article = max_sents_per_article
        train, val, test = self._load_all(dev)
        super(CnnDailyMailDataset,
              self).__init__(train, val, test, self.fields.values(), vectors,
                             vectors_cache, filter_pred)

    def _build_fields(self):
        self.content = NestedField(Field())
        self.abstract = NestedField(Field())
        self.abstract.is_target = True
        self.fields = {
            'article': ('content', self.content),
            'abstract': ('abstract', self.abstract)
        }

    def _load_all(self, dev):
        val_set = self._load_split('val')
        test_set = self._load_split('test')

        if dev:
            dev_set = self._check_dev_set(test_set)
            return dev_set, val_set, test_set
        else:
            train_set = self._load_split('train')
            return train_set, val_set, test_set

    def _check_dev_set(self, test_set):
        reading_path = os.path.join(self.path, 'finished_files', 'dev')

        if not os.path.isdir(reading_path):
            logging.info(
                'No dev set built yet. Building it from the test set now.')

            out_file = reading_path + '.tar'

            with tarfile.open(out_file, 'w') as writer:
                for idx in range(100):
                    article = test_set[idx]
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

        return self._load_split('dev')

    def _load_split(self, split):
        finished_path = os.path.join(self.path, 'finished_files')
        reading_path = os.path.join(finished_path, split)

        if not os.path.isdir(reading_path):
            with tarfile.open(reading_path + '.tar') as tar:
                logging.info(
                    f'Split {split} is not yet extracted to {reading_path}. Doing it now.'
                )
                tar.extractall(finished_path)

        all_articles = []

        for fname in datetime_tqdm(os.listdir(reading_path),
                                   desc=f'Reading {split} files...'):
            with open(os.path.join(reading_path, fname), 'r') as data:
                all_articles.append(Example.fromJSON(data.read(), self.fields))

        return all_articles

    def _build_vocabs(self, vectors, vectors_cache):
        logging.info('Building vocab from the whole dataset.')

        self.content.build_vocab(
            (self.train.content, self.train.abstract, self.val.content,
             self.val.abstract, self.test.content, self.test.abstract),
            vectors=vectors,
            vectors_cache=vectors_cache)

        self.abstract.vocab = self.content.vocab
        self.abstract.nesting_field.vocab = self.content.vocab


if __name__ == '__main__':
    logging.info('Begin')

    dataset = CnnDailyMailDataset('./data/cnn_dailymail',
                                  'glove.6B.300d',
                                  dev=True)

    train_loader, val_loader, test_loader = dataset.get_loaders()

    for batch in train_loader:
        print('hihi')
