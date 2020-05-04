import torch
import math
import itertools
import copy
import numpy as np


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


class RLSumMCTSPureProcess:
    def __init__(self, n_samples, c_puct, n_sents_per_summary):
        self.n_samples = n_samples
        self.c_puct = c_puct
        self.n_sents_per_summary = n_sents_per_summary

    def __call__(self, iterable):
        (state, valid_sentences, scores,) = iterable
        return rlsum_mcts_pure(
            state,
            valid_sentences,
            scores,
            self.n_samples,
            self.c_puct,
            self.n_sents_per_summary,
        )


class RLSumPureNode:
    def __init__(self, state, parent=None):
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

    def expand(self):
        for i in range(50):
            new_state = copy.deepcopy(self.state)
            new_state.update(i)

            new_node = RLSumPureNode(new_state, parent=self)
            self.children.append(new_node)

        self.expanded = True


def rlsum_mcts_pure(
    state, valid_sentences, scores, n_samples, c_puct, n_sents_per_summary,
):
    torch.set_grad_enabled(False)
    text_lens = valid_sentences.sum(-1).unsqueeze(-1)

    # Initial Node
    root_node = RLSumPureNode(state=state)

    targets = []

    for _ in range(n_sents_per_summary):
        mcts_probs = rlsum_mcts_pure_episode(
            scores, n_samples, c_puct, root_node, valid_sentences.cpu(),
        )

        selected_action = torch.distributions.Categorical(mcts_probs).sample()
        targets.append((root_node.state, mcts_probs, selected_action))

        valid_sentences[selected_action] = 0
        root_node = root_node.children[selected_action]

    return targets


def rlsum_mcts_pure_episode(
    scores, n_samples, c_puct, root_node, valid_sentences,
):
    for _ in range(n_samples):
        current_node = root_node

        while not current_node.state.done:
            if not current_node.expanded:
                current_node.expand()

            Q = torch.tensor(
                [c.q_sum for c in current_node.children], dtype=torch.float32
            )
            n_visits = torch.tensor(
                [c.n_visits for c in current_node.children], dtype=torch.float32,
            )

            uct_vals = (
                Q / n_visits
                + torch.sqrt(math.sqrt(current_node.n_visits + 1) / (n_visits + 1))
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
    n_visits[~valid_sentences] = 0.0

    for idx in root_node.state.summary_idxs:
        n_visits[idx] = 0.0

    mcts_pure = n_visits / n_visits.sum(-1).unsqueeze(-1)

    return mcts_pure


class RLSumValuePureProcess:
    def __init__(self, n_samples, c_puct, n_sents_per_summary):
        self.n_samples = n_samples
        self.c_puct = c_puct
        self.n_sents_per_summary = n_sents_per_summary

    def __call__(self, iterable):
        (state, valid_sentences, scores,) = iterable
        return rlsum_value_pure(
            state,
            valid_sentences,
            scores,
            self.n_samples,
            self.c_puct,
            self.n_sents_per_summary,
        )


class RLSumValuePureNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

        self.children = []
        self.n_visits = 0
        self.q_sum = np.zeros((3,))
        self.expanded = False

    def backprop(self, reward):
        self.n_visits += 1
        self.q_sum += reward

        if self.parent:
            self.parent.backprop(reward)

    def expand(self):
        for i in range(50):
            new_state = copy.deepcopy(self.state)
            new_state.update(i)

            new_node = RLSumValuePureNode(new_state, parent=self)
            self.children.append(new_node)

        self.expanded = True


def rlsum_value_pure(
    state, valid_sentences, scores, n_samples, c_puct, n_sents_per_summary,
):
    torch.set_grad_enabled(False)
    text_lens = valid_sentences.sum(-1).unsqueeze(-1)

    # Initial Node
    root_node = RLSumValuePureNode(state=state)
    level_t_nodes = [root_node]
    level_t_valid = [valid_sentences]

    targets = []

    for t in range(n_sents_per_summary):
        next_level_t_nodes = []
        next_level_t_valid = []

        for root_node, valid_sentences in zip(level_t_nodes, level_t_valid):
            mcts_vals = rlsum_value_pure_episode(
                scores, n_samples, c_puct, root_node, valid_sentences.cpu(), t
            )

            mcts_probs = mcts_vals.mean(-1)
            mcts_probs[mcts_probs > 0] = mcts_probs[mcts_probs > 0].exp()
            mcts_probs = mcts_probs / mcts_probs.sum(-1, keepdim=True)
            selected_actions = torch.distributions.Categorical(mcts_probs).sample((4,))
            targets.append((root_node.state, mcts_vals))

            for action in selected_actions:
                next_valid = valid_sentences.clone()
                next_valid[action] = 0
                next_level_t_valid.append(next_valid)
                next_level_t_nodes.append(root_node.children[action])

        level_t_nodes = next_level_t_nodes
        level_t_valid = next_level_t_valid

    return targets


def rlsum_value_pure_episode(scores, n_samples, c_puct, root_node, valid_sentences, t):
    if t == 0:
        n_samples = n_samples * 4
    elif t == 2:
        n_samples = valid_sentences.sum().item()
    for _ in range(n_samples):
        current_node = root_node

        while not current_node.state.done:
            if not current_node.expanded:
                current_node.expand()

            Q = torch.tensor(
                [c.q_sum for c in current_node.children], dtype=torch.float32
            ).mean(-1)
            n_visits = torch.tensor(
                [c.n_visits for c in current_node.children], dtype=torch.float32,
            )

            uct_vals = (
                Q / n_visits
                + torch.sqrt(math.sqrt(current_node.n_visits + 1) / (n_visits + 1))
                * c_puct
            )

            uct_vals[~valid_sentences] = -1

            for idx in current_node.state.summary_idxs:
                uct_vals[idx] = 0.0

            sampled_action = uct_vals.argmax()
            current_node = current_node.children[sampled_action.item()]

        reward = scores[tuple(sorted(current_node.state.summary_idxs))]
        current_node.backprop(reward)

    Q = torch.tensor([c.q_sum for c in root_node.children], dtype=torch.float32)

    n_visits = torch.tensor(
        [c.n_visits for c in root_node.children], dtype=torch.float32,
    )
    mcts_pure = Q / torch.clamp(n_visits, 1).unsqueeze(-1)

    return mcts_pure


class RLSumOHProcess:
    def __init__(self, n_samples, c_puct, n_sents_per_summary):
        self.n_samples = n_samples
        self.c_puct = c_puct
        self.n_sents_per_summary = n_sents_per_summary

    def __call__(self, iterable):
        (state, valid_sentences, scores,) = iterable
        return rlsum_oh(
            state,
            valid_sentences,
            scores,
            self.n_samples,
            self.c_puct,
            self.n_sents_per_summary,
        )


def rlsum_oh(
    state, valid_sentences, scores, n_samples, c_puct, n_sents_per_summary,
):
    targets = []

    mean_scores = scores[:50, :50, :50].mean(-1)
    best_idx = mean_scores.argmax()
    best_idx = np.asarray(np.unravel_index(best_idx, mean_scores.shape))
    selected_actions = []

    for selected_action in best_idx:
        target_probs = torch.zeros(valid_sentences.shape, dtype=torch.float32)
        for action in best_idx:
            if action not in selected_actions:
                target_probs[action] = 1.0

        targets.append(
            (copy.deepcopy(state), target_probs, torch.tensor(selected_action))
        )
        state.update(selected_action)
        selected_actions.append(selected_action)

    return targets


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
    def __init__(self, prior, state, q_init=0.0, parent=None):
        self.prior = prior
        self.state = state
        self.parent = parent

        self.children = []
        self.n_visits = 0
        self.q_sum = q_init
        self.q_init = q_init
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

            new_node = RLSumNode(
                prior.unsqueeze(0), new_state, parent=self, q_init=self.q_init
            )
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

    prior_producer = RLSumPriorsProducer(
        model, sent_contents, doc_contents, valid_sentences, epsilon, alphas
    )
    q_init = scores[scores > 0].mean()

    # Initial Node
    root_node = RLSumNode(
        prior=torch.zeros((1,), dtype=torch.float32), state=state, q_init=q_init
    )

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
                * torch.sqrt(math.sqrt(current_node.n_visits + 1) / (n_visits + 1))
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
    n_visits[~valid_sentences] = 0.0

    for idx in root_node.state.summary_idxs:
        n_visits[idx] = 0.0

    mcts_pure = n_visits / n_visits.sum(-1).unsqueeze(-1)

    return mcts_pure


class RLSumPriorsProducer:
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

        priors = action_dist.probs * valid_sents

        return priors


class AZSumMCTSProcess:
    def __init__(self, model, n_samples, c_puct, n_sents_per_summary, epsilon):
        self.model = model
        self.n_samples = n_samples
        self.c_puct = c_puct
        self.n_sents_per_summary = n_sents_per_summary
        self.epsilon = epsilon

    def __call__(self, iterable):
        (sent_contents, doc_contents, state, valid_sentences, scores,) = iterable
        return azsum_mcts(
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


class AZSumNode:
    def __init__(self, prior, q_val, state, parent=None):
        self.prior = prior
        self.state = state
        self.parent = parent
        self.q_sum = q_val

        self.children = []
        self.n_visits = 0
        self.expanded = False

    def self_backprop(self):
        self.n_visits += 1

        if self.parent:
            self.parent.backprop(self.q_sum)

    def backprop(self, reward):
        self.n_visits += 1
        self.q_sum += reward

        if self.parent:
            self.parent.backprop(reward)

    def expand(self, priors, q_vals):
        for i, (prior, q_val) in enumerate(zip(priors.squeeze(0), q_vals.squeeze(0))):
            new_state = copy.deepcopy(self.state)
            new_state.update(i)

            new_node = AZSumNode(
                prior.unsqueeze(0).cpu(), q_val.cpu(), new_state, parent=self
            )
            self.children.append(new_node)

        self.expanded = True


def azsum_mcts(
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

    prior_producer = AZSumPriorsProducer(
        model, sent_contents, doc_contents, valid_sentences, epsilon, alphas
    )

    # Initial Node
    root_node = AZSumNode(
        prior=torch.zeros((1,), dtype=torch.float32),
        q_val=torch.zeros((3,), dtype=torch.float32),
        state=state,
    )

    targets = []

    for _ in range(n_sents_per_summary):
        mcts_probs, mcts_vals = azsum_mcts_episode(
            prior_producer,
            scores,
            n_samples,
            c_puct,
            epsilon,
            root_node,
            valid_sentences.cpu(),
        )

        epsilon = 0.25
        val_probs = mcts_vals.mean(-1)
        val_probs = val_probs / val_probs.sum(-1)
        sampling_probs = (mcts_probs + val_probs) / 2
        noise = torch.distributions.dirichlet.Dirichlet(alphas).sample().cpu()

        noisy_mcts = (1 - epsilon) * sampling_probs + epsilon * noise
        noisy_mcts = noisy_mcts * valid_sentences.cpu()

        selected_action = torch.distributions.Categorical(noisy_mcts).sample()
        targets.append(
            (root_node.state, mcts_probs, selected_action, mcts_vals, sampling_probs)
        )

        valid_sentences[selected_action] = 0
        root_node = root_node.children[selected_action]

    return targets


def azsum_mcts_episode(
    prior_producer, scores, n_samples, c_puct, epsilon, root_node, valid_sentences,
):
    if not root_node.expanded:
        priors, vals = prior_producer(root_node.state)
        root_node.expand(priors, vals)
        done = True

    for _ in range(n_samples):
        current_node = root_node
        done = False

        while not done:
            Q = torch.stack([c.q_sum for c in current_node.children]).mean(-1)
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

            if current_node.state.done:
                reward = scores[tuple(sorted(current_node.state.summary_idxs))]
                reward = torch.from_numpy(reward)
                current_node.backprop(reward)
                done = True
            elif not current_node.expanded:
                priors, vals = prior_producer(current_node.state)
                current_node.expand(priors, vals)
                current_node.self_backprop()
                done = True

    n_visits = torch.tensor(
        [c.n_visits for c in root_node.children], dtype=torch.float32,
    )
    mcts_probs = n_visits / n_visits.sum(-1).unsqueeze(-1)
    mcts_probs = torch.nn.functional.softmax(mcts_probs, dim=-1)

    q_vals = torch.stack([c.q_sum for c in root_node.children])
    mcts_vals = q_vals / torch.clamp(n_visits, 1).unsqueeze(-1)

    mcts_probs[~valid_sentences] = 0
    mcts_vals[~valid_sentences] = 0

    for idx in root_node.state.summary_idxs:
        mcts_probs[idx] = 0.0
        mcts_vals[idx] = 0.0

    mcts_probs = mcts_probs / mcts_probs.sum(-1).unsqueeze(-1)

    return mcts_probs, mcts_vals


class AZSumPriorsProducer:
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
        action_dist, valid_sents, action_values = self.model.produce_affinities(
            self.sent_contents, self.doc_contents, [state], self.valid_sentences,
        )

        priors = action_dist.probs * valid_sents

        return priors, action_values
