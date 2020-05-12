import torch
import math
import itertools
import copy
import numpy as np
from random import randrange


class RLSumOFULValueProcess:
    def __init__(
        self,
        n_samples,
        lambda_oful,
        delta,
        R,
        S,
        C,
        action_dim,
        device,
        n_sents_per_summary,
    ):
        self.n_samples = n_samples
        self.lambda_oful = lambda_oful
        self.delta = delta
        self.R = R
        self.S = S
        self.C = C
        self.action_dim = action_dim
        self.device = device
        self.n_sents_per_summary = n_sents_per_summary

    def __call__(self, iterable):
        (sent_contents, state, valid_sentences, scores,) = iterable
        return rlsum_oful_value(
            sent_contents.to(self.device),
            state,
            valid_sentences.to(self.device),
            scores,
            self.n_samples,
            self.lambda_oful,
            self.delta,
            self.R,
            self.S,
            self.C,
            self.action_dim,
            self.device,
            self.n_sents_per_summary,
        )


def rlsum_oful_value(
    sent_contents,
    state,
    valid_sentences,
    scores,
    n_samples,
    lambda_oful,
    delta,
    R,
    S,
    C,
    action_dim,
    device,
    n_sents_per_summary,
):
    torch.set_grad_enabled(False)

    n_valid_actions = valid_sentences.sum(-1)
    D = sent_contents[:n_valid_actions]
    D = D / D.norm(dim=-1, keepdim=True)

    # Initial Node
    root_node = RLSumOFULValueNode(
        selected_idxs=state.summary_idxs,
        lambda_oful=lambda_oful,
        action_dim=action_dim,
        n_valid_actions=n_valid_actions,
        parent=None,
        device=device,
    )

    targets = []

    for t in range(n_sents_per_summary):
        target_vals = rlsum_value_oful_episode(
            root_node, scores, n_samples, D, R, delta, S, n_sents_per_summary, device, C
        )

        target_state = copy.deepcopy(state)
        target_state.summary_idxs = root_node.selected_idxs
        targets.append((target_state, target_vals))

        selected_action = target_vals.argmax()
        prev_root = root_node
        root_node = prev_root.children[selected_action]
        del prev_root
        root_node.parent = None

    return targets


class RLSumOFULValueNode:
    def __init__(
        self, selected_idxs, lambda_oful, action_dim, n_valid_actions, parent, device
    ):
        self.selected_idxs = selected_idxs
        self.lambda_oful = lambda_oful
        self.action_dim = action_dim
        self.n_valid_actions = n_valid_actions
        self.device = device

        self.v_t = (
            torch.eye(action_dim, dtype=torch.float32, device=device) * lambda_oful
        )
        self.v_t_inv = (
            torch.eye(action_dim, dtype=torch.float32, device=device) / lambda_oful
        )
        self.det_v_t = lambda_oful ** action_dim
        self.det_v_tau = 0.0
        self.xy = torch.zeros((action_dim, 1), dtype=torch.float32, device=device)
        self.mask = torch.ones(n_valid_actions, dtype=bool, device=device)
        self.theta_hat = torch.zeros(
            (action_dim, 1), dtype=torch.float32, device=device
        )
        for idx in selected_idxs:
            self.mask[idx] = 0
        self.delayed_idxs = self.mask.nonzero()

        self.children = []
        self.expanded = False
        self.parent = parent
        self.n_visits = 0
        self.q_sum = torch.zeros((3,), dtype=torch.float32, device=device)

    def backprop(self, actions, reward, C, terminal=False):
        self.n_visits += 1
        self.q_sum += reward

        if not terminal and len(actions) > 0:
            x_t = actions[-1].unsqueeze(-1)
            actions = actions[:-1]

            self.det_v_t *= 1 + x_t.T.mm(self.v_t_inv).mm(x_t)
            self.v_t_inv -= self.v_t_inv.mm(x_t).mm(x_t.T).mm(self.v_t_inv) / (
                1 + x_t.T.mm(self.v_t_inv).mm(x_t)
            )
            self.v_t += x_t.mm(x_t.T)
            self.xy += x_t * reward.mean()

            if self.det_v_t > (1 + C) * self.det_v_tau:
                self.theta_hat = self.v_t_inv.mm(self.xy)
                self.det_v_tau = copy.deepcopy(self.det_v_t)

        if self.parent:
            self.parent.backprop(actions, reward, C)

    def expand(self):
        for action in self.delayed_idxs:
            new_selected = copy.deepcopy(self.selected_idxs)
            new_selected.append(action.item())

            new_node = RLSumOFULValueNode(
                selected_idxs=new_selected,
                lambda_oful=self.lambda_oful,
                action_dim=self.action_dim,
                n_valid_actions=self.n_valid_actions,
                parent=self,
                device=self.device,
            )
            self.children.append(new_node)

        self.expanded = True

    def select_action(self, D, R, delta, S):
        D_t = D[self.mask]

        if not self.expanded:
            self.expand()

            random_idx = randrange(start=0, stop=D_t.shape[0])

            return random_idx, D_t[random_idx]
        else:
            cf = (
                R
                * math.sqrt(
                    2
                    * math.log(
                        (
                            math.sqrt(self.det_v_t)
                            * math.pow(self.lambda_oful, -self.action_dim / 2)
                        )
                        / delta
                    )
                )
                + math.sqrt(self.lambda_oful) * S
            )

            c = self.theta_hat.T.mm(self.v_t).mm(self.theta_hat) - math.pow(cf, 2)
            alphas = []

            # Parrallel that ?
            for x_it in D_t:
                x_it = x_it.unsqueeze(-1)
                a = x_it.T.mm(self.v_t).mm(x_it)
                b = -(
                    self.theta_hat.T.mm(self.v_t).mm(x_it)
                    + x_it.T.mm(self.v_t).mm(self.theta_hat)
                )

                delt = b.pow(2) - 4 * a * c

                if delt < 0:
                    print("Negative delt !")
                    alphas.append(-math.inf)
                else:
                    alphas.append((-b + math.sqrt(delt)) / (2 * a))

            selected_idx = alphas.index(max(alphas))

            return selected_idx, D_t[selected_idx]


def rlsum_value_oful_episode(
    root_node, scores, n_samples, D, R, delta, S, n_sents_per_summary, device, C
):
    for _ in range(n_samples):
        current_node = root_node
        actions = []

        while len(current_node.selected_idxs) < n_sents_per_summary:
            idx, action = current_node.select_action(D, R, delta, S)
            actions.append(action)
            current_node = current_node.children[idx]

        reward = torch.tensor(
            scores[tuple(sorted(current_node.selected_idxs))], device=device
        )
        current_node.backprop(actions, reward, C, terminal=True)

    targets = D[root_node.mask].mm(root_node.theta_hat)

    return targets
