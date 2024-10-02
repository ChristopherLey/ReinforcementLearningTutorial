from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class PPOMemory:
    def __init__(self, batch_size: int):
        self.states = []
        self.actions = []
        self.log_probabilities = []
        self.rewards = []
        self.values = []
        self.is_terminals = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probabilities),
            np.array(self.values),
            np.array(self.rewards),
            np.array(self.is_terminals),
            batches
        )

    def store_memory(self, state, action, log_probability, value, reward, is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.log_probabilities.append(log_probability)
        self.values.append(value)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.log_probabilities[:]
        del self.values[:]


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir="tmp/ppo"):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = Path(chkpt_dir) / "actor_ppo.pth"
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        action_distribution = self.actor(state)
        action_distribution = Categorical(action_distribution)

        return action_distribution

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir="tmp/ppo"):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = Path(chkpt_dir) / "critic_ppo.pth"
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent:
    def __init__(
            self,
            n_actions,
            input_dims,
            gamma=0.99,
            alpha=0.0003,
            gae_lambda=0.95,
            policy_clip=0.2,
            batch_size=64,
            horizon=2048,
            n_epochs=10
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.horizon = horizon
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, log_probability, value, reward, is_terminal):
        self.memory.store_memory(state, action, log_probability, value, reward, is_terminal)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        action_distribution = self.actor(state)
        value = self.critic(state)

        action = action_distribution.sample()
        log_probability = action_distribution.log_prob(action)

        return action.item(), log_probability.item(), value.item()

    def learn(self):
        for _ in range(self.n_epochs):
            state_array, action_array, old_log_prob_array, values_array, \
                reward_array, is_terminal_array, batches = self.memory.generate_batches()

            values = torch.tensor(values_array).to(self.actor.device)
            advantages = np.zeros(len(reward_array), dtype=np.float32)

            for t in range(len(reward_array) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_array) - 1):
                    a_t += discount * (reward_array[k] + self.gamma * values[k + 1] * (1 - int(is_terminal_array[k]))
                                       - values[k])     # This differs from the video but is the same as the paper
                    discount *= self.gamma * self.gae_lambda

                advantages[t] = a_t

            advantages = torch.tensor(advantages).to(self.actor.device)

            for batch in batches:
                states = torch.tensor(state_array[batch], dtype=torch.float).to(self.actor.device)
                old_log_probs = torch.tensor(old_log_prob_array[batch]).to(self.actor.device)
                actions = torch.tensor(action_array[batch]).to(self.actor.device)

                new_action_distributions = self.actor(states)
                critic_values = self.critic(states).flatten()

                new_log_probs = new_action_distributions.log_prob(actions)
                probability_ratio = torch.exp(new_log_probs - old_log_probs)

                advantage = advantages[batch]
                weighted_probability = probability_ratio * advantage
                weighted_probability_clipped = torch.clamp(
                    probability_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage

                actor_loss = -torch.min(weighted_probability, weighted_probability_clipped).mean()

                returns = advantage + values[batch]
                critic_loss = ((returns - critic_values) ** 2).mean()

                total_loss = actor_loss + 0.5*critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()