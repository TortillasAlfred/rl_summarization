import torch
import math
import itertools
import copy
import numpy as np

from itertools import combinations


class RLSumOFULValueProcess:
    def __init__(
        self,
        n_samples,
        lambda_oful,
        alpha_oful,
        action_dim,
        device,
        n_sents_per_summary,
    ):
        self.n_samples = n_samples
        self.lambda_oful = lambda_oful
        self.alpha_oful = alpha_oful
        self.action_dim = action_dim
        self.device = device
        self.n_sents_per_summary = n_sents_per_summary

    def __call__(self, iterable):
        (sent_contents, doc_contents, state, valid_sentences, scores,) = iterable
        return rlsum_oful_value(
            sent_contents.to(self.device),
            doc_contents.to(self.device),
            state,
            valid_sentences.to(self.device),
            scores,
            self.n_samples,
            self.lambda_oful,
            self.alpha_oful,
            self.action_dim,
            self.device,
            self.n_sents_per_summary,
        )


def rlsum_oful_value(
    sent_contents,
    doc_contents,
    state,
    valid_sentences,
    scores,
    n_samples,
    lambda_oful,
    alpha_oful,
    action_dim,
    device,
    n_sents_per_summary,
):
    torch.set_grad_enabled(False)
    n_valid_actions = valid_sentences.sum(-1)
    action_vectors = sent_contents[:n_valid_actions]
    min_norm_actions = action_vectors.norm(dim=-1, keepdim=True).min()
    min_norm = min(doc_contents.norm(), min_norm_actions)
    doc_contents = doc_contents / min_norm
    action_vectors = action_vectors / min_norm

    # Initial Node
    root_node = RLSumOFULValueNode(
        selected_idxs=state.summary_idxs,
        feature_vector=doc_contents,
        n_valid_actions=n_valid_actions,
        parent=None,
        device=device,
    )

    n_samples = min(n_samples, int(n_samples * (int(n_valid_actions) / 20) ** 3))

    theta_hat, regrets = rlsum_value_oful_episode(
        root_node,
        action_vectors,
        scores,
        n_samples,
        n_sents_per_summary,
        device,
        action_dim,
        alpha_oful,
        lambda_oful,
    )

    return theta_hat, regrets


class RLSumOFULValueNode:
    def __init__(self, selected_idxs, feature_vector, n_valid_actions, parent, device):
        self.selected_idxs = selected_idxs
        self.feature_vector = feature_vector
        self.device = device

        self.n_valid_actions = n_valid_actions
        self.mask = torch.ones(n_valid_actions, dtype=bool, device=device)

        for idx in selected_idxs:
            self.mask[idx] = 0
        self.delayed_idxs = self.mask.nonzero()

        self.children = []
        self.expanded = False
        self.parent = parent
        self.n_visits = 0
        self.q_sum = torch.zeros((3,), dtype=torch.float32, device=device)

    def backprop(self, reward, terminal=False):
        self.n_visits += 1
        self.q_sum += reward

        if not terminal:
            children_weights = [c.n_visits + 1 for c in self.children]
            denom = sum(children_weights)
            self.feature_vector = (
                sum(
                    [
                        c.feature_vector * weight
                        for c, weight in zip(self.children, children_weights)
                    ]
                )
                / denom
            )

        if self.parent:
            self.parent.backprop(reward)

    def expand(self, action_vectors):
        for action in self.delayed_idxs:
            new_selected = copy.deepcopy(self.selected_idxs)
            new_selected.append(action.item())

            new_feature_vector = self.feature_vector + action_vectors[action]

            new_node = RLSumOFULValueNode(
                selected_idxs=new_selected,
                feature_vector=new_feature_vector,
                n_valid_actions=self.n_valid_actions,
                parent=self,
                device=self.device,
            )
            self.children.append(new_node)

        self.expanded = True


def rlsum_value_oful_episode(
    root_node,
    action_vectors,
    scores,
    n_samples,
    n_sents_per_summary,
    device,
    action_dim,
    alpha,
    lambda_oful,
):
    A = torch.eye(action_dim, dtype=torch.float32, device=device) * lambda_oful
    A_inv = torch.eye(action_dim, dtype=torch.float32, device=device) / lambda_oful
    b = torch.zeros((action_dim, 1), dtype=torch.float32, device=device)
    theta_hat = A_inv.mm(b)

    regrets = torch.zeros((n_samples,))
    max_score = torch.tensor(scores.mean(-1).max())

    for n_updates in range(n_samples):
        current_node = root_node

        while len(current_node.selected_idxs) < n_sents_per_summary:
            if not current_node.expanded:
                current_node.expand(action_vectors)

            exploit_a = torch.tensor(
                [c.feature_vector.mm(theta_hat) for c in current_node.children]
            )
            explor_a = torch.tensor(
                [
                    alpha
                    * math.sqrt(n_updates / (c.n_visits + 1))
                    * c.feature_vector.mm(A_inv).mm(c.feature_vector.T).sqrt()
                    for c in current_node.children
                ]
            )
            idx = (exploit_a + explor_a).argmax()
            current_node = current_node.children[idx]

        reward = torch.tensor(
            scores[tuple(sorted(current_node.selected_idxs))], device=device
        )
        current_node.backprop(reward, terminal=True)

        fv = current_node.feature_vector
        A += fv.T.mm(fv)
        if n_updates % 25 == 0:
            A_inv = A.inverse()
        else:
            fv_Vinv = A_inv.mm(fv.T)
            fv_Vinv_fv = fv_Vinv.T.mm(fv.T)
            A_inv -= fv_Vinv.mm(fv_Vinv.T) / (1 + fv_Vinv_fv)
        b += fv.T * reward.mean()
        theta_hat = A_inv.mm(b)

        regrets[n_updates] = max_score - reward.mean()

    return theta_hat, regrets


class RLSumOFULWarmupProcess:
    def __call__(self, iterable):
        (state, valid_sentences, scores,) = iterable
        return rlsum_oful_warmup(state, valid_sentences, scores,)


def rlsum_oful_warmup(state, valid_sentences, scores, n_warmup_summs=3):
    torch.set_grad_enabled(False)

    targets = []
    selected_actions = []
    n_sents = valid_sentences.sum().cpu().item()

    all_bases = np.array(
        list(itertools.product(list(range(n_sents)), list(range(n_sents))))
    )

    sampled_bases = np.random.choice(
        range(n_sents ** 2), n_warmup_summs, replace=False,
    )
    sampled_bases = all_bases[sampled_bases]

    for base_idxs in sampled_bases:
        target_state = copy.deepcopy(state)
        target_state.summary_idxs = base_idxs.tolist()
        target_scores = torch.tensor(
            scores[base_idxs[0], base_idxs[1], :50], device=valid_sentences.device
        )
        targets.append((target_state, target_scores))

    return targets
