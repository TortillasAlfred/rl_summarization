import torch
import numpy as np


class LinSITExpProcess:
    def __init__(
        self, n_samples, c_pucts,
    ):
        self.n_samples = n_samples
        self.c_pucts = c_pucts

    def __call__(self, iterable):
        (sent_contents, priors, scorer, id) = iterable
        results = [
            linsit_exp(sent_contents, priors, scorer, id, c_puct, self.n_samples,)
            for c_puct in self.c_pucts
        ]

        return results


def linsit_exp(
    sent_contents, priors, scorer, id, c_puct, n_samples,
):
    priors = priors[: sent_contents.shape[0]]

    max_rouge, theta_hat_predictions = linsit_exp_episode(
        sent_contents, priors, scorer, n_samples, c_puct,
    )

    key = (c_puct, int(sent_contents.shape[0]), max_rouge, id)

    return (key, theta_hat_predictions)


class LinSITExpPriorsProcess:
    def __init__(self, n_samples, c_puct, tau):
        self.n_samples = n_samples
        self.c_puct = c_puct
        self.tau = tau

    def __call__(self, iterable):
        sent_contents, greedy_priors, prior_choices, scorer, id = iterable
        return [
            linsit_exp_prior(
                sent_contents,
                greedy_prior,
                prior_choices[i],
                scorer,
                id,
                self.c_puct,
                self.tau,
                self.n_samples,
                i,
            )
            for i, greedy_prior in enumerate(greedy_priors)
        ]


def linsit_exp_prior(
    sent_contents,
    greedy_prior,
    prior_choice,
    scorer,
    id,
    c_puct,
    tau,
    n_samples,
    prior_index,
):
    greedy_prior = greedy_prior[: sent_contents.shape[0]]
    unif = torch.ones_like(greedy_prior) / sent_contents.shape[0]

    priors = (1 - tau) * unif + tau * greedy_prior
    priors[priors < 0] = 0
    priors /= priors.sum()

    max_rouge, theta_hat_predictions = linsit_exp_episode(
        sent_contents, priors, scorer, n_samples, c_puct,
    )

    best_idxs = np.argpartition(priors, -3)[-3:]
    prior_max_score = scorer(tuple(best_idxs))
    prior_max_proba = priors[best_idxs].sum()

    key = (
        c_puct,
        int(sent_contents.shape[0]),
        max_rouge,
        id,
        tau,
        prior_index,
        prior_max_score,
        prior_max_proba,
        prior_choice,
    )

    return key, theta_hat_predictions


def linsit_exp_episode(
    action_vectors, priors, scorer, n_samples, c_puct,
):
    action_dim = action_vectors.shape[-1]

    n_visits = torch.zeros(
        (action_vectors.shape[0]), dtype=int, device=action_vectors.device
    )
    theta_predictions = torch.zeros((n_samples,))
    max_score_mean = scorer.scores.max()

    # LinUCB
    A = torch.eye(action_dim, dtype=torch.float32)
    A_inv = torch.eye(action_dim, dtype=torch.float32)
    b = torch.zeros((action_dim, 1), dtype=torch.float32)

    theta_hat = A_inv.mm(b)
    sent_predicted_vals = action_vectors.mm(theta_hat).squeeze()
    for n_updates in range(n_samples):
        p_t_a = (
            sent_predicted_vals
            + c_puct
            * priors
            * (action_vectors.matmul(A_inv) * action_vectors).sum(-1).sqrt()
        )
        p_t_a[n_visits == 0] = float("inf")
        threshold = p_t_a.topk(3)[0][-1]
        elligible_idxs = torch.where(p_t_a >= threshold)[0]
        sampled_idxs = torch.ones_like(elligible_idxs, dtype=torch.float32).multinomial(
            3
        )
        idxs = elligible_idxs[sampled_idxs]
        reward = torch.tensor(scorer(tuple(sorted(idxs.tolist()))))

        n_visits[idxs] += 1
        fv = action_vectors[idxs].mean(0, keepdim=True)
        A += fv.T.mm(fv)
        fv_Vinv = A_inv.mm(fv.T)
        fv_Vinv_fv = fv_Vinv.T.mm(fv.T)
        A_inv -= fv_Vinv.mm(fv_Vinv.T) / (1 + fv_Vinv_fv)
        b += fv.T * reward
        theta_hat = A_inv.mm(b)

        sent_predicted_vals = action_vectors.mm(theta_hat).squeeze()
        _, selected_sents = sent_predicted_vals.topk(3)
        theta_predictions[n_updates] = torch.tensor(
            scorer(tuple(sorted(selected_sents.tolist())))
        )

    return float(max_score_mean), theta_predictions.cpu().numpy()
