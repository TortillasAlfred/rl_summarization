import logging
import torch


class BanditExtractiveSummarizationState:
    def __init__(self, raw_content, raw_abstract, max_episode_len):
        self.raw_content = raw_content
        self.raw_abstract = raw_abstract
        self.max_episode_len = max_episode_len

        self.text_len = len(self.raw_content)
        self.abstract_len = len(self.raw_abstract)
        self.summary_idxs = []
        self.done = False

    def update(self, summary):
        if self.done:
            return
            
        if torch.is_tensor(summary):
            summary = summary.tolist()

        if isinstance(summary, list):
            self.summary_idxs = summary
        else:  # int
            self.summary_idxs.append(summary)

        if len(self.summary_idxs) >= self.max_episode_len:
            self.done = True
