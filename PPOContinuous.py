import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from net import Net, NetContinousActions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOContinuous:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        episode_batch_size,
        actor_learning_rate,
        critic_learning_rate,
        lambda_=0.95,
        eps_clip=0.2,
        writer=None,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps = eps_clip
        self.writer = writer

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
        self.scores = []
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

    def compute_ppo_loss(self):
        states, actions, rewards, terminals, next_states, old_log_probs = tuple(
            [torch.cat(data) for data in zip(*self.current_episode)]
        )

        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            advantages = self.compute_gae(rewards, terminals, values, next_values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.to(states.device)

        means, stds = self.actor(states)
        dist = torch.distributions.Normal(means, stds)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
        ppo_clip_obj = torch.min(ratio * advantages, clipped_ratio * advantages)

        entropy = dist.entropy().sum(dim=-1)  # sum over action dimensions

        if self.writer:
            self.writer.add_scalar("policy/entropy", dist.entropy().mean(), self.n_eps)

        return (ppo_clip_obj + 0.01 * entropy).sum().unsqueeze(0) 

    def update_critic(self, state, reward, done, next_state):
        value = self.critic(state)
        with torch.no_grad():
            next_value = (1 - done) * self.critic(next_state)
            target = reward + self.gamma * next_value
        loss = nn.MSELoss()(value, target)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        if self.writer:
            self.writer.add_scalar("loss/critic", loss.item(), self.total_steps)

    def update(self, state, action, reward, terminated, next_state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        state_tensor = state_tensor.view(state_tensor.size(0), -1)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
        done_tensor = torch.tensor([terminated], dtype=torch.float32).to(device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        next_state_tensor= next_state_tensor.view(next_state_tensor.size(0), -1)

        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            old_log_prob = dist.log_prob(action_tensor).sum(dim=-1)

        transition = (
            state_tensor,
            action_tensor,
            reward_tensor,
            done_tensor,
            next_state_tensor,
            old_log_prob,
        )

        self.current_episode.append(transition)
        self.update_critic(state_tensor, reward_tensor, terminated, next_state_tensor)

        self.episode_reward += reward
        self.total_steps += 1

        if terminated:
            self.n_eps += 1
            if self.writer:
                self.writer.add_scalar("reward", self.episode_reward, self.n_eps)
            self.episode_reward = 0

            self.scores.append(self.compute_ppo_loss())
            self.current_episode = []

            if self.n_eps % self.episode_batch_size == 0:
                self.actor_optimizer.zero_grad()
                loss = -torch.cat(self.scores).sum() / self.episode_batch_size
                loss.backward()
                self.actor_optimizer.step()

                if self.writer:
                    self.writer.add_scalar("loss/actor", loss.item(), self.n_eps)

                self.scores = []

