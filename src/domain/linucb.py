import torch
import numpy as np


class LinUCBProcess:
    def __init__(self, ucb_sampling, c_puct):
        self.ucb_sampling = ucb_sampling
        self.c_puct = c_puct

    def __call__(self, args):
        scorer, ngrams = args
        ngrams = torch.from_numpy(ngrams)
        n_sents = min(scorer.n_sents, 50)

        if self.ucb_sampling == "fix":
            n_samples = 50
        elif self.ucb_sampling == "linear":
            n_samples = 2 * n_sents
        else:
            raise NotImplementedError(f"{self.ucb_sampling} is not a valid UCB sampling method.")

        return linucb(scorer, self.c_puct, n_samples, n_sents, ngrams)


def linucb(scorer, c_puct, n_samples, n_sents, action_vectors, priors=None):
    if priors is None:
        priors = torch.from_numpy(np.ones((n_sents,)) / n_sents)

    action_dim = action_vectors.shape[-1]
    max_score_mean = scorer.scores.max()

    # LinUCB
    A = torch.eye(action_dim, dtype=torch.float32)
    A_inv = torch.eye(action_dim, dtype=torch.float32)
    b = torch.zeros((action_dim, 1), dtype=torch.float32)

    theta_hat = A_inv.mm(b)
    sent_predicted_vals = action_vectors.mm(theta_hat).squeeze()
    for _ in range(n_samples):
        p_t_a = sent_predicted_vals + c_puct * priors * (action_vectors.matmul(A_inv) * action_vectors).sum(-1).sqrt()
        threshold = p_t_a.topk(3)[0][-1]
        elligible_idxs = torch.where(p_t_a >= threshold)[0]
        sampled_idxs = torch.ones_like(elligible_idxs, dtype=torch.float32).multinomial(3)
        idxs = elligible_idxs[sampled_idxs]
        reward = torch.tensor(scorer(tuple(sorted(idxs.tolist()))))

        fv = action_vectors[idxs].mean(0, keepdim=True)
        A += fv.T.mm(fv)
        fv_Vinv = A_inv.mm(fv.T)
        fv_Vinv_fv = fv_Vinv.T.mm(fv.T)
        A_inv -= fv_Vinv.mm(fv_Vinv.T) / (1 + fv_Vinv_fv)
        b += fv.T * reward
        theta_hat = A_inv.mm(b)

        sent_predicted_vals = action_vectors.mm(theta_hat).squeeze()

    _, selected_sents = sent_predicted_vals.topk(3)
    best_score = torch.tensor(scorer(tuple(sorted(selected_sents.tolist()))))
    ucb_delta = max_score_mean - best_score

    # MinMax scaling
    sent_predicted_vals = (sent_predicted_vals - sent_predicted_vals.min()) / (
        sent_predicted_vals.max() - sent_predicted_vals.min()
    )

    returned_q_vals = torch.zeros(50, dtype=torch.float32)
    returned_q_vals[:n_sents] = sent_predicted_vals

    return returned_q_vals, ucb_delta
