import torch
import math
import itertools
import copy


def bs_mcts(
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


class RLSumMCTSProcess:
    def __init__(self, model, n_samples, c_puct, n_sents_per_summary, epsilon):
        self.model = model
        self.n_samples = n_samples
        self.c_puct = c_puct
        self.n_sents_per_summary = n_sents_per_summary
        self.epsilon = epsilon

    def __call__(self, iterable):
        (sent_contents, doc_contents, state, valid_sentences, scores,) = iterable
        return rlsum_mcts(
            sent_contents,
            doc_contents,
            state,
            valid_sentences,
            scores,
            self.model,
            self.n_samples,
            self.c_puct,
            self.n_sents_per_summary,
            self.epsilon,
        )


class RLSumNode:
    def __init__(self, prior, state, parent=None):
        self.prior = prior
        self.state = state
        self.parent = parent

        self.children = []
        self.n_visits = 0
        self.q_sum = 0.0
        self.expanded = False

    def backprop(self, reward):
        self.n_visits += 1
        self.q_sum += reward

        if self.parent:
            self.parent.backprop(reward)

    def expand(self, priors):
        for i, prior in enumerate(priors.squeeze(0)):
            new_state = copy.deepcopy(self.state)
            new_state.update(i)

            new_node = RLSumNode(prior.unsqueeze(0), new_state, parent=self)
            self.children.append(new_node)

        self.expanded = True


def rlsum_mcts(
    sent_contents,
    doc_contents,
    state,
    valid_sentences,
    scores,
    model,
    n_samples,
    c_puct,
    n_sents_per_summary,
    epsilon,
):
    torch.set_grad_enabled(False)
    text_lens = valid_sentences.sum(-1).unsqueeze(-1)

    alpha = text_lens.float() / 100
    alphas = torch.ones_like(valid_sentences, dtype=torch.float32) * alpha

    prior_producer = PriorsProducer(
        model, sent_contents, doc_contents, valid_sentences, epsilon, alphas
    )

    # Initial Node
    root_node = RLSumNode(prior=torch.zeros((1,), dtype=torch.float32), state=state)

    targets = []

    for _ in range(n_sents_per_summary):
        mcts_probs = rlsum_mcts_episode(
            prior_producer,
            scores,
            n_samples,
            c_puct,
            epsilon,
            root_node,
            valid_sentences.cpu(),
        )

        epsilon = 0.25
        noise = torch.distributions.dirichlet.Dirichlet(alphas).sample().cpu()
        noisy_mcts = (1 - epsilon) * mcts_probs + epsilon * noise
        noisy_mcts = noisy_mcts * valid_sentences.cpu()

        selected_action = torch.distributions.Categorical(noisy_mcts).sample()
        targets.append((root_node.state, mcts_probs, selected_action))

        valid_sentences[selected_action] = 0
        root_node = root_node.children[selected_action]

    return targets


def rlsum_mcts_episode(
    prior_producer, scores, n_samples, c_puct, epsilon, root_node, valid_sentences,
):
    for _ in range(n_samples):
        current_node = root_node

        while not current_node.state.done:
            if not current_node.expanded:
                priors = prior_producer(current_node.state)
                current_node.expand(priors)

            Q = torch.tensor(
                [c.q_sum for c in current_node.children], dtype=torch.float32
            )
            n_visits = torch.tensor(
                [c.n_visits for c in current_node.children], dtype=torch.float32,
            )
            priors = torch.tensor(
                [c.prior for c in current_node.children], dtype=torch.float32
            )

            uct_vals = (
                Q / n_visits
                + priors
                * torch.sqrt(math.log(current_node.n_visits + 1) * 2 / (n_visits + 1))
                * c_puct
            )

            uct_vals[~valid_sentences] = -1

            for idx in current_node.state.summary_idxs:
                uct_vals[idx] = 0.0

            sampled_action = uct_vals.argmax()
            current_node = current_node.children[sampled_action.item()]

        reward = scores[tuple(sorted(current_node.state.summary_idxs))].mean()
        current_node.backprop(reward)

    n_visits = torch.tensor(
        [c.n_visits for c in root_node.children], dtype=torch.float32,
    )
    mcts_pure = n_visits / n_visits.sum(-1).unsqueeze(-1)

    return mcts_pure


class PriorsProducer:
    def __init__(
        self, model, sent_contents, doc_contents, valid_sentences, epsilon, alphas
    ):
        self.model = model
        self.sent_contents = sent_contents.unsqueeze(0)
        self.doc_contents = doc_contents.unsqueeze(0)
        self.valid_sentences = valid_sentences.unsqueeze(0)
        self.epsilon = epsilon
        self.alphas = alphas

    def __call__(self, state):
        action_dist, valid_sents = self.model.produce_affinities(
            self.sent_contents, self.doc_contents, [state], self.valid_sentences,
        )

        # Add Dirichlet Noise
        noise = torch.distributions.dirichlet.Dirichlet(self.alphas).sample()
        priors = (1 - self.epsilon) * action_dist.probs + self.epsilon * noise
        priors = priors * valid_sents

        return priors
