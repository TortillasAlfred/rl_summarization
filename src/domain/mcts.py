import torch
import math


def article_mcts(
    priors, valid_sentences, n_samples, scores, c_puct, n_sents_per_summary, epsilon
):
    text_lens = valid_sentences.sum(-1).unsqueeze(-1)
    priors = priors[valid_sentences]
    n_visits = torch.zeros_like(priors)
    Q = torch.zeros_like(priors)

    # Add Dirichlet Noise
    epsilon = 0.25
    alpha = text_lens.float() / 100
    alpha = torch.ones_like(priors, dtype=torch.float) * alpha
    noise = torch.distributions.dirichlet.Dirichlet(alpha).sample()
    priors = (1 - epsilon) * priors + epsilon * noise

    for t in range(1, n_samples + 1):
        # Sample
        uct_vals = (
            Q / n_visits
            + priors * torch.sqrt(math.log(t) * 2 / (n_visits + 1)) * c_puct
        )
        _, sampled_actions = uct_vals.topk(k=n_sents_per_summary, dim=-1, sorted=False)

        # Get rewards
        rewards = scores[tuple(sorted(sampled_actions))]
        rewards = torch.tensor(rewards).mean(-1).to(priors.device)

        # Backprop
        n_visits.index_add_(
            -1,
            sampled_actions,
            torch.ones(size=sampled_actions.shape, device=priors.device),
        )
        Q.index_add_(
            -1, sampled_actions, rewards.repeat_interleave(repeats=len(sampled_actions))
        )

    mcts_pure = n_visits / n_visits.sum(-1).unsqueeze(-1)
    out_probs = torch.zeros_like(valid_sentences, dtype=torch.float32)
    out_probs[valid_sentences] = mcts_pure

    return out_probs
