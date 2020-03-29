import torch
import json
import os


class RougeReward:
    def __init__(self, rouge_path):
        self.rouge_path = rouge_path
        self.loaded_scores = {}

    def init_articles(self, articles, split):
        self.loaded_scores = {}

        ids = [article.id for article in articles]
        reading_dir = os.path.join(self.rouge_path, split)
        for id in ids:
            with open(os.path.join(reading_dir, f"{id}.json"), "rb") as f:
                self.loaded_scores[id] = json.load(f)

    def __call__(self, idxs_by_id, device):
        results = []

        for id, all_idxs in idxs_by_id:
            pass
        results = self.evaluator.get_all_scores(pairs)
        return torch.FloatTensor(results).to(device)

    @staticmethod
    def from_config(self, config):
        return RougeReward(config["rouge_path"])

