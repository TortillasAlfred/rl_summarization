from numpy.lib.type_check import nan_to_num
import torch
import math
import itertools
import copy
import numpy as np

from itertools import combinations, chain


class RLSumOFULValueProcess:
    def __init__(
        self,
        n_samples,
        lambda_oful,
        alpha_oful,
        action_dim,
        device,
        n_sents_per_summary,
        scaling_n_samples,
    ):
        self.n_samples = n_samples
        self.lambda_oful = lambda_oful
        self.alpha_oful = alpha_oful
        self.action_dim = action_dim
        self.device = device
        self.n_sents_per_summary = n_sents_per_summary
        self.scaling_n_samples = scaling_n_samples

    def __call__(self, iterable):
        (sent_contents, doc_contents, valid_sentences, scores,) = iterable
        return rlsum_oful_value(
            sent_contents.to(self.device),
            doc_contents.to(self.device),
            valid_sentences.to(self.device),
            scores,
            self.n_samples,
            self.lambda_oful,
            self.alpha_oful,
            self.action_dim,
            self.device,
            self.n_sents_per_summary,
            self.scaling_n_samples,
        )


def rlsum_oful_value(
    sent_contents,
    doc_contents,
    valid_sentences,
    scores,
    n_samples,
    lambda_oful,
    alpha_oful,
    action_dim,
    device,
    n_sents_per_summary,
    scaling_n_samples,
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
        selected_idxs=frozenset(),
        feature_vector=doc_contents,
        n_valid_actions=n_valid_actions,
        device=device,
    )

    theta_hat, max_score, theta_hat_predictions, regrets = rlsum_value_oful_episode(
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
    return (
        theta_hat,
        max_score,
        n_valid_actions,
        theta_hat_predictions,
        regrets,
    )


class RLSumOFULValueNode:
    def __init__(self, selected_idxs, feature_vector, n_valid_actions, device):
        self.selected_idxs = list(selected_idxs)
        self.feature_vector = feature_vector
        self.device = device

        self.n_valid_actions = n_valid_actions
        self.mask = torch.ones(n_valid_actions, dtype=bool, device=device)

        for idx in selected_idxs:
            self.mask[idx] = 0
        self.delayed_idxs = self.mask.nonzero().tolist()

        self.expanded = False
        self.connected = False
        self.n_visits = 0
        self.q_sum = torch.zeros((3,), dtype=torch.float32, device=device)

        self.parents_sets = chain.from_iterable(
            combinations(self.selected_idxs, r)
            for r in reversed(range(len(self.selected_idxs)))
        )
        self.parents_sets = [frozenset(ps) for ps in self.parents_sets]
        self.parents = []
        self.children = []

    def backprop(self, reward, action_vectors, node_dict, terminal=False):
        if not self.connected:
            self.connect(action_vectors, node_dict)

        for parent in self.parents:
            parent.update(reward)

    def connect(self, action_vectors, node_dict):
        set_idxs = frozenset(self.selected_idxs)
        for parent_set in self.parents_sets:
            if parent_set in node_dict:
                parent_node = node_dict[parent_set]
            else:
                actions_removed = set_idxs - parent_set

                parent_feature_vector = self.feature_vector - sum(
                    [action_vectors[action] for action in actions_removed]
                )

                parent_node = RLSumOFULValueNode(
                    selected_idxs=parent_set,
                    feature_vector=parent_feature_vector,
                    n_valid_actions=self.n_valid_actions,
                    device=self.device,
                )
                node_dict[parent_set] = parent_node

            self.parents.append(parent_node)

        self.connected = True

    def update(self, reward, terminal=False):
        self.n_visits += 1
        self.q_sum += reward

        if self.expanded and not terminal:
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

    def expand(self, action_vectors, node_dict):
        for action in self.delayed_idxs:
            new_selected = frozenset(self.selected_idxs + action)

            if new_selected in node_dict:
                new_node = node_dict[new_selected]
            else:
                new_feature_vector = self.feature_vector + action_vectors[action]

                new_node = RLSumOFULValueNode(
                    selected_idxs=new_selected,
                    feature_vector=new_feature_vector,
                    n_valid_actions=self.n_valid_actions,
                    device=self.device,
                )
                node_dict[new_selected] = new_node

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
    node_dict = {frozenset(root_node.selected_idxs): root_node}

    A = torch.eye(action_dim, dtype=torch.float32, device=device) * lambda_oful
    A_inv = torch.eye(action_dim, dtype=torch.float32, device=device) / lambda_oful
    b = torch.zeros((action_dim, 1), dtype=torch.float32, device=device)
    theta_hat = A_inv.mm(b)

    regrets = torch.zeros((n_samples,))
    theta_predictions = torch.zeros((n_samples,))
    max_score_idx = np.unravel_index(scores.argmax(), scores.shape[:-1])
    max_score = torch.from_numpy(scores[max_score_idx])
    max_score_mean = max_score

    for n_updates in range(n_samples):
        current_node = root_node

        while len(current_node.selected_idxs) < n_sents_per_summary:
            if not current_node.expanded:
                current_node.expand(action_vectors, node_dict)

            all_fv_a = torch.stack(
                [c.feature_vector.squeeze() for c in current_node.children]
            )
            all_n_visits = torch.tensor(
                [c.n_visits + 1 for c in current_node.children],
                dtype=float,
                device=device,
            )

            p_t_a = (
                all_fv_a.matmul(theta_hat).squeeze()
                + alpha
                * (n_updates / all_n_visits).sqrt()
                * (all_fv_a.matmul(A_inv) * all_fv_a).sum(-1).sqrt()
            )
            idx = p_t_a.argmax()
            current_node = current_node.children[idx]

        reward = torch.tensor(
            scores[tuple(sorted(current_node.selected_idxs))], device=device
        )
        current_node.backprop(reward, action_vectors, node_dict, terminal=True)

        fv = current_node.feature_vector
        A += fv.T.mm(fv)
        fv_Vinv = A_inv.mm(fv.T)
        fv_Vinv_fv = fv_Vinv.T.mm(fv.T)
        A_inv -= fv_Vinv.mm(fv_Vinv.T) / (1 + fv_Vinv_fv)
        b += fv.T * reward
        theta_hat = A_inv.mm(b)

        regrets[n_updates] = max_score_mean - reward

        sent_predicted_vals = theta_hat.T.mm(action_vectors.T)
        _, selected_sents = sent_predicted_vals.topk(n_sents_per_summary)
        theta_predictions[n_updates] = torch.tensor(
            scores[tuple(sorted(selected_sents.tolist()))], device=device
        )

    return theta_hat, max_score, theta_predictions, regrets


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
