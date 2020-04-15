

class DefaultLoggedMetrics:
    def __init__(self, n_states):
        self.entropies = [[] for _ in range(n_states)]
        self.selected_action_probs = [[] for _ in range(n_states)]
        self.max_action_prob = [[] for _ in range(n_states)]
        self.min_action_prob = [[] for _ in range(n_states)]
        self.mean_action_prob = [[] for _ in range(n_states)]
        self.median_action_prob = [[] for _ in range(n_states)]
        self.mcts_entropies = [[] for _ in range(n_states)]
        self.mcts_selected_action_probs = [[] for _ in range(n_states)]
        self.mcts_max_action_prob = [[] for _ in range(n_states)]
        self.mcts_min_action_prob = [[] for _ in range(n_states)]
        self.mcts_mean_action_prob = [[] for _ in range(n_states)]
        self.mcts_median_action_prob = [[] for _ in range(n_states)]
        self.rouge_1 = [[] for _ in range(n_states)]
        self.rouge_2 = [[] for _ in range(n_states)]
        self.rouge_L = [[] for _ in range(n_states)]
        self.rouge_mean = [[] for _ in range(n_states)]

    def log(self, actions, actions_probs, rewards, is_mcts, done):
        for i, (action, reward) in enumerate(zip(actions, rewards)):
            # Policy logging 
            if is_mcts:
                self.mcts_entropies[i].append(actions_probs.entropy()[i])
                self.mcts_max_action_prob[i].append(actions_probs.probs[i].max())
                self.mcts_min_action_prob[i].append(actions_probs.probs[i].min())
                self.mcts_mean_action_prob[i].append(actions_probs.probs[i].mean())
                self.mcts_median_action_prob[i].append(actions_probs.probs[i].median())
                
                if len(action.shape) > 0 and len(action) > 1:
                    self.mcts_selected_action_probs[i].extend(actions_probs.probs[i].index_select(0, action.long()))
                else:
                    self.mcts_selected_action_probs[i].append(actions_probs.probs[i][action])
            else:
                self.entropies[i].append(actions_probs.entropy()[i])
                self.max_action_prob[i].append(actions_probs.probs[i].max())
                self.min_action_prob[i].append(actions_probs.probs[i].min())
                self.mean_action_prob[i].append(actions_probs.probs[i].mean())
                self.median_action_prob[i].append(actions_probs.probs[i].median())
                
                if len(action.shape) > 1:
                    self.selected_action_probs[i].extend(actions_probs.probs[i].index_select(0, action.long()))
                else:
                    self.selected_action_probs[i].append(actions_probs.probs[i][action])

            if done:
                # Rewards
                self.rouge_1[i].append(reward[0])
                self.rouge_2[i].append(reward[1])
                self.rouge_L[i].append(reward[2])
                self.rouge_mean[i].append(reward.mean())