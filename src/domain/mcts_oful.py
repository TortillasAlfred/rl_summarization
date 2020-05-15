import torch
import math
import itertools
import copy
import numpy as np


class RLSumOFULValueProcess:
    def __init__(
        self,
        n_samples,
        lambda_oful,
        delta,
        R,
        S,
        c_puct,
        action_dim,
        device,
        n_sents_per_summary,
    ):
        self.n_samples = n_samples
        self.lambda_oful = lambda_oful
        self.delta = delta
        self.R = R
        self.S = S
        self.c_puct = c_puct
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
            self.c_puct,
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
    c_puct,
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
            root_node,
            scores,
            n_samples,
            D,
            R,
            delta,
            S,
            n_sents_per_summary,
            device,
            c_puct,
        )

        target_state = copy.deepcopy(state)
        target_state.summary_idxs = root_node.selected_idxs
        targets.append((target_state, target_vals))

        selected_action = target_vals.argmax()
        prev_root = root_node
        root_node = prev_root.children[selected_action]
        del prev_root
        root_node.parent = None

    del root_node

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
        self.log_det_v_t = action_dim * math.log(lambda_oful)
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

    def backprop(self, actions, reward, terminal=False):
        self.n_visits += 1
        self.q_sum += reward

        if not terminal and len(actions) > 0:
            x_t = actions[-1].unsqueeze(-1)
            actions = actions[:-1]

            x_Vinv = self.v_t_inv.mm(x_t)
            x_Vinv_x = x_Vinv.T.mm(x_t)
            self.log_det_v_t += (x_Vinv_x + 1).log().item()
            self.v_t_inv -= x_Vinv.mm(x_Vinv.T) / (1 + x_Vinv_x)
            self.v_t += x_t.mm(x_t.T)
            self.xy += x_t * reward.mean()

            self.theta_hat = self.v_t_inv.mm(self.xy)

        if self.parent:
            self.parent.backprop(actions, reward)

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

    def select_action(self, D, R, delta, S, c_puct):
        D_t = D[self.mask]

        if not self.expanded:
            self.expand()

        cf = (
            R
            * math.sqrt(
                self.log_det_v_t
                - self.action_dim * math.log(self.lambda_oful)
                + 2 * math.log(1 / delta)
            )
            + math.sqrt(self.lambda_oful) * S
        )

        D_Vinv_D_sqrt = (D_t.mm(self.v_t_inv) * D_t).sum(-1).sqrt()
        ucb = D_t.mm(self.theta_hat).squeeze(-1) + c_puct * cf * D_Vinv_D_sqrt

        selected_idx = ucb.argmax()

        return selected_idx, D_t[selected_idx]


def rlsum_value_oful_episode(
    root_node, scores, n_samples, D, R, delta, S, n_sents_per_summary, device, c_puct
):
    for _ in range(n_samples):
        current_node = root_node
        actions = []

        while len(current_node.selected_idxs) < n_sents_per_summary:
            idx, action = current_node.select_action(D, R, delta, S, c_puct)
            actions.append(action)
            current_node = current_node.children[idx]

        reward = torch.tensor(
            scores[tuple(sorted(current_node.selected_idxs))], device=device
        )
        current_node.backprop(actions, reward, terminal=True)

    targets = D[root_node.mask].mm(root_node.theta_hat)

    return targets
