import torch
import numpy as np


class LinSITExpProcess:
    def __init__(
        self, n_samples, c_pucts, n_pretraining_steps, device,
    ):
        self.n_samples = n_samples
        self.c_pucts = c_pucts
        self.n_pretraining_steps = n_pretraining_steps
        self.device = device

    def __call__(self, iterable):
        (sent_contents, valid_sentences, priors, scores, id) = iterable
        results = [
            linsit_exp(
                sent_contents.to(self.device),
                valid_sentences.to(self.device),
                priors.to(self.device),
                scores,
                id,
                c_puct,
                self.n_samples,
                self.n_pretraining_steps,
                self.device,
            )
            for c_puct in self.c_pucts
        ]

        return results


def linsit_exp(
    sent_contents,
    valid_sentences,
    priors,
    scores,
    id,
    c_puct,
    n_samples,
    n_pretraining_steps,
    device,
):
    torch.set_grad_enabled(False)
    n_valid_actions = valid_sentences.sum(-1)
    action_vectors = sent_contents[:n_valid_actions]
    action_vectors = action_vectors / action_vectors.norm(dim=-1, keepdim=True).max()

    priors = priors[:n_valid_actions]

    max_rouge, theta_hat_predictions = linsit_exp_episode(
        action_vectors, priors, scores, n_samples, c_puct, device,
    )

    key = (c_puct, n_pretraining_steps, int(n_valid_actions), max_rouge, id)

    return (key, theta_hat_predictions)


class LinSITExpPriorsProcess:
    def __init__(
        self, n_samples, n_pretraining_steps, device,
    ):
        self.n_samples = n_samples
        self.n_pretraining_steps = n_pretraining_steps
        self.device = device

    def __call__(self, iterable):
        (
            (sent_contents, valid_sentences, greedy_priors, prior_choices, scores, id),
            c_puct,
            tau,
        ) = iterable
        return [
            linsit_exp_prior(
                sent_contents.to(self.device),
                valid_sentences.to(self.device),
                greedy_prior.to(self.device),
                prior_choices[i],
                scores,
                id,
                c_puct,
                tau,
                self.n_samples,
                self.n_pretraining_steps,
                self.device,
                i,
            )
            for i, greedy_prior in enumerate(greedy_priors.to(self.device))
        ]


def linsit_exp_prior(
    sent_contents,
    valid_sentences,
    greedy_prior,
    prior_choice,
    scores,
    id,
    c_puct,
    tau,
    n_samples,
    n_pretraining_steps,
    device,
    prior_index,
):
    torch.set_grad_enabled(False)
    n_valid_actions = valid_sentences.sum(-1)
    action_vectors = sent_contents[:n_valid_actions]
    action_vectors = action_vectors / action_vectors.norm(dim=-1, keepdim=True).max()

    greedy_prior = greedy_prior[:n_valid_actions]
    unif = torch.ones_like(greedy_prior) / n_valid_actions

    priors = (1 - tau) * unif + tau * greedy_prior
    priors[priors < 0] = 0
    priors /= priors.sum()

    max_rouge, theta_hat_predictions = linsit_exp_episode(
        action_vectors, priors, scores, n_samples, c_puct, device,
    )

    priors = priors.detach().cpu().numpy()
    best_idxs = np.argpartition(priors, -3)[-3:]
    prior_max_score = scores[tuple(best_idxs)].mean()
    prior_max_proba = priors[best_idxs].sum()

    key = (
        c_puct,
        n_pretraining_steps,
        int(n_valid_actions),
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
    action_vectors, priors, scores, n_samples, c_puct, device,
):
    action_dim = action_vectors.shape[-1]

    theta_predictions = torch.zeros((n_samples,))
    max_score_idx = np.unravel_index(scores.mean(-1).argmax(), scores.shape[:-1])
    max_score = torch.from_numpy(scores[max_score_idx])
    max_score_mean = max_score.mean()

    # LinUCB
    A = torch.eye(action_dim, dtype=torch.float32, device=device)
    A_inv = torch.eye(action_dim, dtype=torch.float32, device=device)
    b = torch.zeros((action_dim, 1), dtype=torch.float32, device=device)
    theta_hat = A_inv.mm(b)

    for n_updates in range(n_samples):
        p_t_a = (
            action_vectors.matmul(theta_hat).squeeze()
            + c_puct
            * priors
            * (action_vectors.matmul(A_inv) * action_vectors).sum(-1).sqrt()
        )
        _, idxs = p_t_a.topk(3)

        reward = torch.tensor(scores[tuple(sorted(idxs.tolist()))], device=device)

        fv = action_vectors[idxs].sum(0, keepdim=True)
        A += fv.T.mm(fv)
        fv_Vinv = A_inv.mm(fv.T)
        fv_Vinv_fv = fv_Vinv.T.mm(fv.T)
        A_inv -= fv_Vinv.mm(fv_Vinv.T) / (1 + fv_Vinv_fv)
        b += fv.T * reward.mean()
        theta_hat = A_inv.mm(b)

        sent_predicted_vals = action_vectors.mm(theta_hat).squeeze()
        _, selected_sents = sent_predicted_vals.topk(3)
        theta_predictions[n_updates] = torch.tensor(
            scores[tuple(sorted(selected_sents.tolist()))], device=device
        ).mean()

    return float(max_score_mean), theta_predictions.to("cpu")
