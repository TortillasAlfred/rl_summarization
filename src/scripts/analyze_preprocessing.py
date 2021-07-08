from src.domain.dataset_bert import CnnDailyMailDatasetBert, MIN_NUM_SEN_PER_DOCUMENT, PAD
from src.domain.rewards.rouge import RougeRewardBuilder, index_getter

from types import SimpleNamespace
import time
from transformers import BertTokenizerFast
from os.path import join
from joblib import Parallel, delayed
import numpy as np
import pickle
from tqdm import tqdm
from itertools import product, combinations

MAX_SENTS_PER_DOC = 50
MAX_LEN_SENT = 80


def encode_document(document, tokenizer, max_sents_per_doc, max_len_sent, min_len_sent, max_tokens_per_doc):
    """Utility function used to preprocess and tokenize the data

    Args:
        document (list): list of sentences to preprocess and tokenize
        tokenizer (BertTokenizerFast): object method used to preprocess and tokenize

    Return:
        dict: dictionary with keys `input_ids`, `token_type_ids` and `attention_mask`
    """

    # return nothing if `document` is empty
    if not document:
        return

    # concat sentences into document
    result_ = {
        "token_ids": [],
        "token_type_ids": [],
        "mark": [],
        "segs": [],
        "clss": [0],
        "mark_clss": [True],
        "sentence_gap": [],
    }
    preprocessing_stats = {"too_short_sents_skipped": 0, "not_enough_space_sents_skipped": 0, "sents_retained": 0}

    current_sentence = 0
    for seg, sentence in enumerate(document[:max_sents_per_doc]):
        output = tokenizer(
            sentence, add_special_tokens=True, max_length=max_len_sent, truncation=True, return_tensors="pt"
        )
        ids, types, mark = (output["input_ids"][0], output["token_type_ids"][0], output["attention_mask"][0])
        if len(ids) < min_len_sent + 2:
            current_sentence += 1
            preprocessing_stats["too_short_sents_skipped"] += 1
            continue
        if len(result_["token_ids"]) + len(ids.tolist()) > max_tokens_per_doc:
            break
        result_["token_ids"].extend(ids.tolist())
        result_["token_type_ids"].extend(types.tolist())
        result_["mark"].extend(mark.tolist())
        result_["segs"].extend([seg % 2] * len(ids))
        result_["clss"].append(len(result_["segs"]))
        result_["mark_clss"].append(True)
        result_["sentence_gap"].append(current_sentence)

        preprocessing_stats["sents_retained"] += 1

    result_["clss"].pop()
    result_["mark_clss"].pop()

    # padding
    pad_ = max_tokens_per_doc - len(result_["token_ids"])
    result_["token_ids"].extend([PAD] * pad_)
    result_["token_type_ids"].extend([result_["token_type_ids"][-1]] * pad_)
    result_["mark"].extend([0] * pad_)
    result_["segs"].extend([1 - (seg % 2)] * pad_)

    preprocessing_stats["n_sents_doc"] = len(document[:max_sents_per_doc])
    preprocessing_stats["not_enough_space_sents_skipped"] = (
        preprocessing_stats["n_sents_doc"]
        - preprocessing_stats["sents_retained"]
        - preprocessing_stats["too_short_sents_skipped"]
    )
    for key, val in preprocessing_stats.items():
        if key == "n_sents_doc":
            result_[key] = val
        else:
            result_[key] = val / preprocessing_stats["n_sents_doc"]

    return result_


class WrappedCnnDailyMailDatasetBert(CnnDailyMailDatasetBert):
    def tokenized_dataset(self, dataset):
        """Method that tokenizes each document in the train, test and validation dataset

        Args:
            dataset (DatasetDict): dataset that will be tokenized (train, test, validation)

        Returns:
            dict: dataset once tokenized
        """
        if not self.config.bert_cache:  # Used if there's no internet connection
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        else:
            tokenizer = BertTokenizerFast.from_pretrained(
                join(self.config.bert_cache, "tokenizer_save_pretrained", local_files_only=True)
            )

        train_articles = self.get_values("train", "article", dataset, tokenizer)
        test_articles = self.get_values("test", "article", dataset, tokenizer)
        val_articles = self.get_values("val", "article", dataset, tokenizer)
        train_abstracts = self.get_values("train", "abstract", dataset, tokenizer)
        test_abstracts = self.get_values("test", "abstract", dataset, tokenizer)
        val_abstracts = self.get_values("val", "abstract", dataset, tokenizer)

        return {
            "train": (
                dataset["train"]["id"],
                train_articles,
                train_abstracts,
                dataset["train"]["article"],
                dataset["train"]["abstract"],
            ),
            "test": (
                dataset["train"]["id"],
                test_articles,
                test_abstracts,
                dataset["test"]["article"],
                dataset["test"]["abstract"],
            ),
            "val": (
                dataset["val"]["id"],
                val_articles,
                val_abstracts,
                dataset["val"]["article"],
                dataset["val"]["abstract"],
            ),
        }

    def get_values(self, set_, part_, dataset, tokenizer):
        """Utility method used inside `tokenized_dataset`

        Args:
            set_ (str): directory containing json files. Possible choices: 'train', 'test' or 'val'
            part_ (str): Once we've choosen `set_`, select the type of document, i.e.
            'article' or 'abstract'
            dataset (DatasetDict): dataset that will be tokenized (train, test, validation)
            tokenizer (BertTokenizerFast): object method used to preprocess and tokenize

        Returns:
            list: return a list of tokenized documents
        """
        return Parallel(n_jobs=-1)(
            delayed(encode_document)(
                document,
                tokenizer,
                self.config.max_sents_per_doc,
                self.config.max_len_sent,
                self.config.min_len_sent,
                self.config.max_tokens_per_doc,
            )
            for document in dataset[set_][part_]
            if len(document) > MIN_NUM_SEN_PER_DOCUMENT
        )

    def compute_all_preprocessing_stats(self):
        all_sents_retained = []
        all_n_sents_doc = []
        all_not_enough_space_sents_skipped = []
        all_too_short_sents_skipped = []

        for article in self.dataset["train"][1]:
            all_sents_retained.append(article["sents_retained"])
            all_n_sents_doc.append(article["n_sents_doc"])
            all_not_enough_space_sents_skipped.append(article["not_enough_space_sents_skipped"])
            all_too_short_sents_skipped.append(article["too_short_sents_skipped"])

        preprocessing_stats = {}

        preprocessing_stats["all_sents_retained"] = all_sents_retained
        preprocessing_stats["all_n_sents_doc"] = all_n_sents_doc
        preprocessing_stats["all_not_enough_space_sents_skipped"] = all_not_enough_space_sents_skipped
        preprocessing_stats["all_too_short_sents_skipped"] = all_too_short_sents_skipped

        preprocessing_stats["mean_sents_retained"] = np.mean(all_sents_retained)
        preprocessing_stats["mean_n_sents_doc"] = np.mean(all_n_sents_doc)
        preprocessing_stats["mean_not_enough_space_sents_skipped"] = np.mean(all_not_enough_space_sents_skipped)
        preprocessing_stats["mean_too_short_sents_skipped"] = np.mean(all_too_short_sents_skipped)

        return preprocessing_stats

    def compute_rouge_stats(self, data_path):
        rouge_builder = RougeRewardBuilder(data_path)

        all_deltas = []
        all_true_maxs = []

        for article_id, article_details in zip(self.dataset["train"][0], self.dataset["train"][1]):
            # TODO: Ici se trouve le code qui te sera utile pour les cibles binaires.
            #
            # Le scorer de chaque document correspond ici à rouge_builder.init_scorer(article_id, "train")
            # Tu peux accéder à la matrice de scores avec scorer.scores
            # Tu pourras retrouver les 3 meilleures phrases avec l'appel suivant
            # best_avail_idxs = available_summs[available_scores.argmax()]
            scores = rouge_builder.init_scorer(article_id, "train").scores
            available_sents = np.arange(0, len(article_details["clss"])) + np.array(article_details["sentence_gap"])
            available_summs = combinations(available_sents, 3)
            corresponding_idxs = list(map(lambda idxs: index_getter(*idxs), available_summs))
            available_scores = scores[corresponding_idxs]

            # TODO: On veut aussi logger la sous-optimalité pour un sanity check
            # suboptimality = np.mean(scores.max() - available_scores.max()) devrait
            # faire exactement ça.
            all_deltas.append(np.mean(scores.max() - available_scores.max()))
            all_true_maxs.append(np.mean(scores.max()))

        rouge_stats = {}

        rouge_stats["all_deltas"] = all_deltas
        rouge_stats["all_true_maxs"] = all_true_maxs

        rouge_stats["mean_deltas"] = np.mean(all_deltas)
        rouge_stats["std_deltas"] = np.std(all_deltas)
        rouge_stats["mean_true_max"] = np.mean(all_true_maxs)

        return rouge_stats


def get_config(data_path, min_len_sent, max_tokens_per_doc):
    config = dict()

    config["store_data_tokenized"] = False
    config["load_data_tokenized"] = False
    config["data_path"] = data_path
    config["bert_cache"] = ""
    config["max_sents_per_doc"] = MAX_SENTS_PER_DOC
    config["max_len_sent"] = MAX_LEN_SENT
    config["min_len_sent"] = min_len_sent
    config["max_tokens_per_doc"] = max_tokens_per_doc

    return SimpleNamespace(**config)


if __name__ == "__main__":
    all_min_len_sent = list(np.arange(start=0, stop=11, step=1))
    all_max_tokens_per_doc = [512] + list(np.arange(start=600, stop=1600, step=100))

    results = dict()

    for min_len_sent, max_tokens_per_doc in tqdm(
        list(product(all_min_len_sent, all_max_tokens_per_doc)), desc="Processing preprocess configs..."
    ):
        run_key = (min_len_sent, max_tokens_per_doc)
        results[run_key] = dict()

        config = get_config(
            data_path="./data/cnn_dailymail/dev", min_len_sent=min_len_sent, max_tokens_per_doc=max_tokens_per_doc
        )

        dataset = WrappedCnnDailyMailDatasetBert(config)

        # Compile preprocessing stats for whole dataset
        results[run_key]["preprocessing_stats"] = dataset.compute_all_preprocessing_stats()

        # Compile ROUGE deltas stats
        results[run_key]["rouge_stats"] = dataset.compute_rouge_stats(config.data_path)

    with open("preprocessing_results.pck", "wb") as f:
        pickle.dump(results, f)

    print("Done !")
