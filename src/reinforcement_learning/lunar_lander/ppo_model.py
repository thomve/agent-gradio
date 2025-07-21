import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 64
UPDATE_EPOCHS = 4
STEPS_PER_UPDATE = 2048
TOTAL_TIMESTEPS = 100_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

    def get_action(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze()

    def evaluate_actions(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value.squeeze()


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def store(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value):
        returns, advs = [], []
        gae = 0
        values = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + GAMMA * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - self.dones[t]) * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[t])
        self.returns = returns
        self.advs = advs

    def get_batches(self):
        indices = np.arange(len(self.obs))
        np.random.shuffle(indices)
        for start in range(0, len(self.obs), BATCH_SIZE):
            end = start + BATCH_SIZE
            idx = indices[start:end]
            yield (
                torch.tensor(np.array(self.obs)[idx], dtype=torch.float32).to(device),
                torch.tensor(np.array(self.actions)[idx], dtype=torch.int64).to(device),
                torch.tensor(np.array(self.log_probs)[idx], dtype=torch.float32).to(device),
                torch.tensor(np.array(self.returns)[idx], dtype=torch.float32).to(device),
                torch.tensor(np.array(self.advs)[idx], dtype=torch.float32).to(device),
            )

    def clear(self):
        self.__init__()

env = gym.make("LunarLander-v3")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

model = ActorCritic(obs_dim, act_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
buffer = RolloutBuffer()

obs, _ = env.reset()
ep_return = 0
timesteps = 0

while timesteps < TOTAL_TIMESTEPS:
    buffer.clear()
    for _ in range(STEPS_PER_UPDATE):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, _, value = model.get_action(obs_tensor)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = truncated or terminated
        buffer.store(obs, action.item(), reward, done, log_prob.item(), value.item())

        obs = next_obs
        ep_return += reward
        timesteps += 1

        if done:
            print(f"Episode return: {ep_return:.1f}")
            obs, _ = env.reset()
            ep_return = 0

    with torch.no_grad():
        last_obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        _, _, _, last_value = model.get_action(last_obs_tensor)
    buffer.compute_returns_and_advantages(last_value.item())

    # PPO update
    for _ in range(UPDATE_EPOCHS):
        for batch in buffer.get_batches():
            obs_b, act_b, old_logp_b, ret_b, adv_b = batch
            new_logp, entropy, value = model.evaluate_actions(obs_b, act_b)

            ratio = torch.exp(new_logp - old_logp_b)
            clipped_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_b
            policy_loss = -torch.min(ratio * adv_b, clipped_adv).mean()
            value_loss = ((ret_b - value) ** 2).mean()
            entropy_bonus = entropy.mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

env.close()
