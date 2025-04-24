import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# from torch.utils.tensorboard import SummaryWriter

from net import Net, NetContinousActions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOContinuous:
    def __init__(self, action_space, observation_space, gamma, episode_batch_size,
                 actor_learning_rate, critic_learning_rate, lambda_=0.95, eps_clip=0.2,
                 writer=None, K_epochs=4, minibatch_size=64, c1=0.5, c2=0.01):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps = eps_clip
        self.writer = writer
        self.K_epochs = K_epochs
        self.minibatch_size = minibatch_size
        self.c1 = c1
        self.c2 = c2

        self.episode_batch_size = episode_batch_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        hidden_size = 128
        obs_size = np.prod(self.observation_space.shape)
        act_dim = self.action_space.shape[0]

        self.actor = NetContinousActions(obs_size, hidden_size, act_dim)
        self.critic = Net(obs_size, hidden_size, 1)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        self.current_episode = []
        self.episode_reward = 0
        self.n_eps = 0
        self.total_steps = 0

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        state_tensor = state_tensor.view(state_tensor.size(0), -1)
        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        action_clipped = torch.clamp(action, min=self.action_space.low[0], max=self.action_space.high[0])
        return action_clipped.squeeze().cpu().numpy().astype(np.float32).flatten(), log_prob.cpu().detach()

    def compute_gae(self, rewards, terminals, values, next_values):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - terminals[t]) * next_values[t] - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - terminals[t]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32)

    def compute_joint_loss(self, states, actions, advantages, returns, old_log_probs):
        means, stds = self.actor(states)
        dist = torch.distributions.Normal(means, stds)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        values = self.critic(states).squeeze()

        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
        L_clip = torch.min(ratio * advantages, clipped_ratio * advantages)

        L_value = (returns - values).pow(2)

        if self.writer:
            self.writer.add_scalar("policy/entropy", entropy.mean(), self.n_eps)

        loss = -(L_clip - self.c1 * L_value + self.c2 * entropy).mean()
        return loss

    def update(self, state, action, reward, terminated, next_state):
        # Prepare transition
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device).view(1, -1)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
        done_tensor = torch.tensor([terminated], dtype=torch.float32).to(device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device).view(1, -1)

        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            old_log_prob = dist.log_prob(action_tensor).sum(dim=-1)

        self.current_episode.append((
            state_tensor.squeeze(0), action_tensor.squeeze(0),
            reward_tensor, done_tensor,
            next_state_tensor.squeeze(0), old_log_prob
        ))

        self.episode_reward += reward
        self.total_steps += 1

        if terminated:
            self.n_eps += 1
            if self.writer:
                self.writer.add_scalar("reward", self.episode_reward, self.n_eps)
            self.episode_reward = 0

            if self.n_eps % self.episode_batch_size == 0:
                self._train_on_batch()
                self.current_episode = []

    def _train_on_batch(self):
        states, actions, rewards, dones, next_states, old_log_probs = zip(*self.current_episode)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.cat(rewards)
        dones = torch.cat(dones)
        next_states = torch.stack(next_states)
        old_log_probs = torch.stack(old_log_probs)

        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            advantages = self.compute_gae(rewards, dones, values, next_values).to(values.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + values

        dataset = TensorDataset(states, actions, advantages, returns, old_log_probs)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        for _ in range(self.K_epochs):
            for minibatch in dataloader:
                s_mb, a_mb, adv_mb, ret_mb, old_lp_mb = [x.to(device) for x in minibatch]

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                loss = self.compute_joint_loss(s_mb, a_mb, adv_mb, ret_mb, old_lp_mb)
                loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()

        if self.writer:
            self.writer.add_scalar("loss/actor", loss.item(), self.n_eps)
        
        self.last_joint_loss = loss.item()

