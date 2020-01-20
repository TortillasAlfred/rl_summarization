from rouge import Rouge

# We only want the f1-score
Rouge.STATS = ['f']


class RougeReward:
    def __init__(self, *,
                 metrics=['rouge-n', 'rouge-l'],
                 max_n=2,
                 limit_length=False,
                 stemming=False,
                 apply_avg=False):
        self.evaluator = Rouge(metrics=metrics, max_n=max_n,
                               limit_length=limit_length, apply_avg=apply_avg)

    def __call__(self, hypothesis, references):
        hypothesis = [' '.join([' '.join(sent) for sent in h])
                      for h in hypothesis]
        references = [' '.join([' '.join(sent) for sent in r])
                      for r in references]

        scores = self.evaluator.get_scores(hypothesis, references)
        scores = [[score['f'][0] for score in metric]
                  for metric in scores.values()]
        return [(r1 + r2 + rl)/3 for r1, r2, rl in zip(*scores)]


if __name__ == '__main__':
    from dataset import CnnDmDataset
    from dataloader import get_repeating_data_loader

    cnn_dm_dataset = CnnDmDataset('dev')

    loader = get_repeating_data_loader(cnn_dm_dataset, 3, 8)

    reward_function = RougeReward()

    for batch in loader:
        hypothesis = [article.content[:3] for article in batch]
        references = [article.abstract for article in batch]

        print(reward_function(hypothesis, references))
