import torch


class Rollouts:
    def __init__(self, *, n_texts, n_sents, n_steps, n_repeats, device):
        self.selected_idxs = torch.zeros(
            n_steps, n_repeats, n_texts, n_sents, dtype=bool).to(device)
        self.entropies = torch.zeros(
            n_steps, n_repeats, n_texts, dtype=float).to(device)
        self.logprobs = torch.zeros(
            n_steps, n_repeats, n_texts, dtype=float).to(device)
        self.values = torch.zeros(
            n_steps, n_repeats, n_texts, dtype=float).to(device)
        self.returns = torch.zeros(
            n_steps + 1, n_repeats, n_texts, dtype=float).to(device)
        self.done_masks = torch.ones(
            n_steps + 1, n_repeats, n_texts, dtype=bool).to(device)
        self.idxs = torch.zeros(
            n_steps, n_repeats, n_texts, dtype=torch.long).to(device)

        self.step = 0
        self.n_steps = n_steps

    def store(self, selected_idxs, values, logprobs, entropies, done_masks, idxs):
        self.selected_idxs[self.step].copy_(selected_idxs)
        self.values[self.step].copy_(values)
        self.logprobs[self.step].copy_(logprobs)
        self.entropies[self.step].copy_(entropies)
        self.done_masks[self.step + 1].copy_(done_masks)
        self.idxs[self.step].copy_(idxs)

        self.step += 1

    def compute_returns(self, rewards, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.n_steps)):
            self.returns[step] = (
                self.returns[step + 1] * gamma * self.done_masks[step + 1]
                + rewards[step]
            )
