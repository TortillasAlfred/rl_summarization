import torch
import math


def article_mcts(
    priors, valid_sentences, n_samples, scores, c_puct, n_sents_per_summary
):
    valid_sentences = valid_sentences.float()
    n_visits = torch.zeros_like(priors)
    Q = torch.zeros_like(priors)
    text_lens = valid_sentences.sum(-1).unsqueeze(-1)

    for t in range(1, n_samples + 1):
        # Sample
        uct_vals = Q / torch.clamp(n_visits, 1) + text_lens * c_puct * math.pow(
            t, 0.25
        ) * priors / (n_visits + 1).float().pow(0.5)
        uct_vals = uct_vals * valid_sentences
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

    return mcts_pure
