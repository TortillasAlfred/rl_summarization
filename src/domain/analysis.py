from utils import configure_logging

import pickle
import os
import logging

from utils import datetime_tqdm
from collections import Counter


class Analyzer:

    def analyze_dataset_embeddings_pair(self, dataset, embeddings):
        logging.info('Beginning analysis')
        self.save_path = os.path.join(dataset.path, 'analysis')
        os.makedirs(self.save_path, exist_ok=True)

        texts_analysis = list(map(self.analyze_text, datetime_tqdm(
            dataset, desc='Analyzing dataset texts...')))

        vocab, vocab_path = self._build_save_vocab(texts_analysis)
        unk_words = embeddings.fit_to_vocab(vocab, return_unk_words=True)

        self.report_path = os.path.join(self.save_path, 'report.txt')
        self.overview_section(texts_analysis)
        self.article_stats_section(texts_analysis)
        self.vocab_stats(texts_analysis, vocab, vocab_path, unk_words)
        logging.info(
            f'Analysis done. A detailed report can be found in {self.report_path}')

    def overview_section(self, texts_analysis):
        pass

    def article_stats_section(self, texts_analysis):
        pass

    def vocab_stats(self, texts_analysis, vocab, vocab_path, unk_words):
        pass

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
        analysis_report['n_tokens_content'] = sum(
            [len(sent) for sent in split_content])
        analysis_report['n_tokens_abstract'] = sum(
            [len(sent) for sent in split_abstract])
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

    from embeddings import PretrainedEmbeddings
    from dataset import CnnDmDataset

    emb = PretrainedEmbeddings('./data/embeddings/glove/glove.6B.50d.txt')

    cnn_dm_dataset = CnnDmDataset('val')
    analyzer = Analyzer()
    analyzer.analyze_dataset_embeddings_pair(cnn_dm_dataset, emb)
