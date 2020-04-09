import logging


class BanditExtractiveSummarizationState:
    def __init__(self, content, raw_content, raw_abstract):
        self.content = content
        self.raw_content = raw_content
        self.raw_abstract = raw_abstract

        self.text_len = len(self.raw_content)
        self.abstract_len = len(self.raw_abstract)
        self.summary_idxs = []
        self.done = False

    def update(self, summary):
        if self.done:
            return

        summary = summary.tolist()

        if any([idx > self.text_len for idx in summary]):
            logging.info(str(self))
            logging.info(summary)

        self.summary_idxs = summary

        if len(self.summary_idxs) == self.abstract_len:
            self.done = True
