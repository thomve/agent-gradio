import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
ENV_ID = "Ant-v5"
HIDDEN_SIZE = 128
LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
BATCH_SIZE = 64
EPOCHS = 10
STEPS_PER_UPDATE = 2048
TOTAL_TIMESTEPS = 1_000_000

env = gym.make(ENV_ID, render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_low = float(env.action_space.low[0])
act_high = float(env.action_space.high[0])


# Neural Network for Actor and Critic
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(HIDDEN_SIZE, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor_mean(x), self.critic(x)

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        mean, _ = self.forward(obs)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(act_low, act_high).cpu().numpy(), log_prob.item()

    def evaluate_actions(self, obs, actions):
        mean, value = self.forward(obs)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy, value.squeeze(-1)


model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)


def compute_gae(rewards, values, dones, next_value):
    advantages = []
    gae = 0
    values = values + [next_value]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + GAMMA * LAMBDA * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def ppo_update(obs, actions, log_probs_old, returns, advantages):
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(EPOCHS):
        idx = np.random.permutation(len(obs))
        for start in range(0, len(obs), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_idx = idx[start:end]
            batch_obs = obs[batch_idx]
            batch_actions = actions[batch_idx]
            batch_log_probs_old = log_probs_old[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]

            log_probs, entropy, values = model.evaluate_actions(batch_obs, batch_actions)
            ratio = torch.exp(log_probs - batch_log_probs_old)

            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = ((batch_returns - values) ** 2).mean()
            entropy_bonus = entropy.mean()

            loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# Training loop
obs, _ = env.reset()
episode_reward = 0
timestep = 0

while timestep < TOTAL_TIMESTEPS:
    obs_buf, actions_buf, rewards_buf, dones_buf, log_probs_buf, values_buf = [], [], [], [], [], []

    for _ in range(STEPS_PER_UPDATE):
        action, log_prob = model.get_action(obs)
        value = model.forward(torch.tensor(obs, dtype=torch.float32).to(device))[1].item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_buf.append(obs)
        actions_buf.append(action)
        rewards_buf.append(reward)
        dones_buf.append(float(done))
        log_probs_buf.append(log_prob)
        values_buf.append(value)

        obs = next_obs
        episode_reward += reward
        timestep += 1

        if done:
            print(f"Episode reward: {episode_reward:.2f}")
            obs, _ = env.reset()
            episode_reward = 0
        

    next_value = model.forward(torch.tensor(obs, dtype=torch.float32).to(device))[1].item()
    advantages, returns = compute_gae(rewards_buf, values_buf, dones_buf, next_value)

    ppo_update(obs_buf, actions_buf, log_probs_buf, returns, advantages)

env.close()
