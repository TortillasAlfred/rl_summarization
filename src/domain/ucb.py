import numpy as np

from scipy.special import softmax


class UCBProcess:
    def __init__(self, ucb_sampling, c_puct):
        self.ucb_sampling = ucb_sampling
        self.c_puct = c_puct

    def __call__(self, scorer):
        n_sents = min(scorer.n_sents, 50)

        if self.ucb_sampling == "fix":
            n_samples = 100
        elif self.ucb_sampling == "linear":
            n_samples = 2 * n_sents + 50
        else:
            raise NotImplementedError(f"{self.ucb_sampling} is not a valid UCB sampling method.")

        return ucb(scorer, self.c_puct, n_samples, n_sents)


class BertUCBProcess:
    def __init__(self, ucb_sampling, c_puct):
        self.ucb_sampling = ucb_sampling
        self.c_puct = c_puct

    def __call__(self, bench_mark):
        sentence_gap, scorer = bench_mark[0], bench_mark[1]

        n_sents = min(scorer.n_sents, 50)

        if self.ucb_sampling == "fix":
            n_samples = 100
        elif self.ucb_sampling == "linear":
            n_samples = 2 * n_sents + 50
        else:
            raise NotImplementedError(
                f"{self.ucb_sampling} is not a valid UCB sampling method."
            )

        return ucb_bert(scorer, self.c_puct, n_samples, n_sents, sentence_gap)


class UCBPriorsProcess:
    def __init__(self, ucb_sampling, c_puct, prior_version, batch_float):
        self.ucb_sampling = ucb_sampling
        self.c_puct = c_puct
        self.prior_version = prior_version
        self.batch_float = batch_float

    def __call__(self, args):
        scorer, action_vals = args
        n_sents = min(scorer.n_sents, 50)

        if self.ucb_sampling == "fix":
            n_samples = 100
        elif self.ucb_sampling == "linear":
            n_samples = 2 * n_sents + 50
        else:
            raise NotImplementedError(f"{self.ucb_sampling} is not a valid UCB sampling method.")

        beta_n = np.log(n_sents + 9) - np.log(9)

        if self.prior_version == "aggro":
            beta = 10 ** (-2 + 2 * self.batch_float)
            beta = min(beta, 100) * beta_n
        elif self.prior_version == "soft":
            beta = 10 ** (-2 + self.batch_float)
            beta = min(beta, 1) * beta_n
        else:
            raise NotImplementedError(f"{self.prior_version} is not a valid Prior version.")

        priors = softmax(action_vals[:n_sents] * beta)

        return ucb(scorer, self.c_puct, n_samples, n_sents, priors)


def ucb(scorer, c_puct, n_samples, n_sents, priors=None):
    if priors is None:
        priors = np.ones((n_sents,)) / n_sents

    n_visits = np.zeros(n_sents, dtype=int)
    q_vals = np.zeros(n_sents, dtype=np.float32)

    for n in range(1, n_samples + 1):
        ucb = q_vals + c_puct * priors * np.sqrt(2 * np.log(n) / n_visits)
        ucb = np.nan_to_num(ucb, nan=np.inf)
        threshold = np.partition(ucb, -3)[-3]
        elligible_idxs = np.argwhere(ucb >= threshold)[:, 0]
        sampled_idxs = np.random.choice(elligible_idxs, 3, replace=False)
        summ_score = scorer(tuple(sampled_idxs))
        q_vals[sampled_idxs] = (summ_score + q_vals[sampled_idxs] * n_visits[sampled_idxs]) / (
            n_visits[sampled_idxs] + 1
        )
        n_visits[sampled_idxs] += 1

    max_score = scorer.scores.max()
    threshold = np.partition(q_vals, -3)[-3]
    elligible_idxs = np.argwhere(q_vals >= threshold)[:, 0]
    best_idxs = np.random.choice(elligible_idxs, 3, replace=False)
    best_score = scorer(tuple(best_idxs))
    ucb_delta = max_score - best_score

    # MinMax scaling
    q_vals = (q_vals - q_vals.min()) / (q_vals.max() - q_vals.min())

    returned_q_vals = np.zeros(50, dtype=np.float32)
    returned_q_vals[:n_sents] = q_vals

    return (returned_q_vals, ucb_delta)


def ucb_bert(scorer, c_puct, n_samples, n_sents, sentence_gap, priors=None):
    n_sents_new = len(sentence_gap)
    if priors is None:
        priors = np.ones((n_sents_new,)) / n_sents_new

    n_visits = np.zeros(n_sents_new, dtype=int)
    q_vals = np.zeros(n_sents_new, dtype=np.float32)

    for n in range(1, n_samples + 1):
        ucb = q_vals + c_puct * priors * np.sqrt(2 * np.log(n) / n_visits)
        ucb = np.nan_to_num(ucb, nan=np.inf)
        threshold = np.partition(ucb, -3)[-3]
        elligible_idxs = np.argwhere(ucb >= threshold)
        elligible_idxs = elligible_idxs[:, 0]
        sampled_idxs = np.random.choice(elligible_idxs, 3, replace=False)

        sentence_gap = np.asarray(sentence_gap)
        real_sampled_idxs = sampled_idxs + sentence_gap[sampled_idxs]

        summ_score = scorer(tuple(real_sampled_idxs))
        q_vals[sampled_idxs] = (
            summ_score + q_vals[sampled_idxs] * n_visits[sampled_idxs]
        ) / (n_visits[sampled_idxs] + 1)
        n_visits[sampled_idxs] += 1

    max_score = scorer.scores.max()
    threshold = np.partition(q_vals, -3)[-3]
    elligible_idxs = np.argwhere(q_vals >= threshold)[:, 0]
    best_idxs = np.random.choice(elligible_idxs, 3, replace=False)
    best_score = scorer(tuple(best_idxs))
    ucb_delta = max_score - best_score

    # MinMax scaling
    q_vals = (q_vals - q_vals.min()) / (q_vals.max() - q_vals.min())

    returned_q_vals = np.zeros(50, dtype=np.float32)
    returned_q_vals[:n_sents_new] = q_vals

    return (returned_q_vals, ucb_delta)
