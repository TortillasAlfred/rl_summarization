from utils import configure_logging, datetime_tqdm
from embeddings import PretrainedEmbeddings

import pickle
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter


class Analyzer:

    def analyze_dataset_embeddings_pair(self, dataset, embeddings_path):
        logging.info('Beginning analysis')
        self.save_path = os.path.join(dataset.path, 'analysis')
        os.makedirs(self.save_path, exist_ok=True)

        texts_analysis = list(map(self.analyze_text, datetime_tqdm(
            dataset, desc='Analyzing dataset texts...')))

        self.report_path = os.path.join(self.save_path, 'report.txt')

        if os.path.isfile(self.report_path):
            logging.info(
                f'A preexisting report was found in {self.report_path}. Overriding it.')
            os.remove(self.report_path)

        self.overview_section(texts_analysis)
        self.article_stats_section(texts_analysis)
        self.vocab_stats(texts_analysis, embeddings_path)
        logging.info(
            f'Analysis done. A detailed report can be found in {self.report_path}')

    def _n_most_frequent(self, occurences_dict, n):
        text = f'\n\nHere are the {n} most frequent ones:\n\n'

        most_frequent = occurences_dict.most_common(n)
        for idx, (word, occ) in enumerate(most_frequent):
            text += f'{idx + 1} - {word} : {occ} occurences.\n'

        return text

    def overview_section(self, texts_analysis):
        logging.info('Begin Overview Section')
        section = '***Dataset overview***\n\n'

        total = Counter(
            [word for t in texts_analysis for word in t['all_tokens']])
        n_total = sum(total.values())
        n_unique = len(total)
        section += f"The dataset is made of {n_total} tokens, of which are {n_unique} unique."
        section += self._n_most_frequent(total, 25)

        with open(self.report_path, 'a') as f:
            f.write(section)

    def _per_article_stat(self, stat_array, stat_name):
        text = f'\n\nFor the {stat_name}, the statistics are :\n\n'

        stat_array = np.asarray(stat_array)
        text += f'Mean : {stat_array.mean():.2f}\n'
        text += f'Std : {stat_array.std():.2f}\n'
        text += f'Min : {stat_array.min()}\n'
        text += f'Max : {stat_array.max()}\n'

        hist_name = stat_name.replace(' ', '_') + '.png'
        hist_path = os.path.join(self.save_path, hist_name)
        _ = plt.hist(stat_array, bins='auto')
        plt.title(stat_name)
        plt.savefig(hist_path)
        plt.close()

        text += f'An histogram of the statistics has been saved to {hist_path}\n'

        return text

    def article_stats_section(self, texts_analysis):
        logging.info('Begin Article Stats Section')
        section = '\n\n***Statistics per article***'

        section += self._per_article_stat([t['n_tokens_content']
                                           for t in texts_analysis], 'number of tokens per content')
        section += self._per_article_stat([t['n_sents_content']
                                           for t in texts_analysis], 'number of sentences per content')
        section += self._per_article_stat([sent_len for t in texts_analysis for sent_len in t['n_tokens_sent_content']],
                                          'number of tokens per sentence in content')
        section += self._per_article_stat([t['n_tokens_abstract']
                                           for t in texts_analysis], 'number of tokens per abstract')
        section += self._per_article_stat([t['n_sents_abstract']
                                           for t in texts_analysis], 'number of sentences per abstract')
        section += self._per_article_stat([sent_len for t in texts_analysis for sent_len in t['n_tokens_sent_abstract']],
                                          'number of tokens per sentence in abstract')

        with open(self.report_path, 'a') as f:
            f.write(section)

    def vocab_stats(self, texts_analysis, embeddings_path):
        logging.info('Begin Vocabulary Stats Section')

        vocab, vocab_path = self._build_save_vocab(texts_analysis)
        embeddings = PretrainedEmbeddings(embeddings_path)
        original_size = len(embeddings)
        unk_words = embeddings.fit_to_vocab(vocab, return_unk_words=True)
        post_size = len(embeddings)

        section = '\n\n***Vocabulary Statistics***\n\n'

        section += f'A vocabulary counter dictionary for the dataset has been saved to {vocab_path}\n\n'
        section += f'The embeddings used for the following analysis were loaded from {embeddings_path}\n\n'
        section += f'The embeddings countain a total of {original_size} individual tokens. Out of them, a subset of {post_size} are present in the dataset and are to be retained.'
        section += f'A total of {len(unk_words)} unique tokens were present in the dataset but not in the embeddings words.'
        unk_words = Counter({word: vocab[word] for word in iter(unk_words)})
        section += self._n_most_frequent(unk_words, 25)

        with open(self.report_path, 'a') as f:
            f.write(section)

    def _build_save_vocab(self, texts_analysis):
        logging.info('Building vocabulary...')

        vocab = Counter(
            [word for analysis in texts_analysis for word in analysis['all_tokens']])

        vocab_path = os.path.join(self.save_path, 'vocab_cnt.pkl')
        logging.info(f'Vocabulary done. Saving to {vocab_path}')
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

        return vocab, vocab_path

    def analyze_text(self, text):
        analysis_report = {}

        split_content = [sent.split() for sent in text.content]
        split_abstract = [sent.split() for sent in text.abstract]

        # Number of total tokens
        analysis_report['n_tokens_sent_content'] = [
            len(sent) for sent in split_content]
        analysis_report['n_tokens_content'] = sum(
            analysis_report['n_tokens_sent_content'])
        analysis_report['n_tokens_sent_abstract'] = [
            len(sent) for sent in split_abstract]
        analysis_report['n_tokens_abstract'] = sum(
            analysis_report['n_tokens_sent_abstract'])
        analysis_report['n_tokens_total'] = analysis_report['n_tokens_content'] + \
            analysis_report['n_tokens_abstract']
        analysis_report['all_tokens'] = [word for text in [
            split_content, split_abstract] for sent in text for word in sent]

        # Unique tokens
        content_unique_tokens = set(
            [word for sent in split_content for word in sent])
        abstract_unique_tokens = set(
            [word for sent in split_abstract for word in sent])
        total_unique_tokens = content_unique_tokens | abstract_unique_tokens
        analysis_report['n_unique_content'] = len(content_unique_tokens)
        analysis_report['n_unique_abstract'] = len(abstract_unique_tokens)
        analysis_report['n_unique_total'] = len(total_unique_tokens)
        analysis_report['all_unique_tokens'] = total_unique_tokens

        # Number of sentences
        analysis_report['n_sents_content'] = len(text.content)
        analysis_report['n_sents_abstract'] = len(text.abstract)

        return analysis_report


if __name__ == '__main__':
    configure_logging()

    from dataset import CnnDmDataset

    cnn_dm_dataset = CnnDmDataset('dev')
    analyzer = Analyzer()
    analyzer.analyze_dataset_embeddings_pair(
        cnn_dm_dataset, './data/embeddings/glove/glove.6B.50d.txt')
