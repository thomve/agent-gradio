import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from collections import deque
import ale_py

gym.register_envs(ale_py)

# Hyperparameters
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.1
LEARNING_RATE = 2.5e-4
ENT_COEF = 0.01
VF_COEF = 0.5
BATCH_SIZE = 64
STEPS_PER_UPDATE = 1024
EPOCHS = 4
MAX_FRAMES = 1_000_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Preprocess frame: grayscale + resize
def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84))
    return obs / 255.0


# Frame stacker for partial observability
class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        self.frames = deque([obs] * self.k, maxlen=self.k)
        return np.stack(self.frames, axis=0)

    def append(self, obs):
        self.frames.append(obs)
        return np.stack(self.frames, axis=0)


# PPO Network: CNN encoder + LSTM + actor/critic heads
class PPOActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.lstm = nn.LSTM(conv_out_size, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv(o)
        return int(np.prod(o.size()))

    def forward(self, x, lstm_state=None, dones=None):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        conv_out = self.conv(x)
        conv_out = conv_out.view(batch_size, seq_len, -1)

        if lstm_state is None:
            h, c = [torch.zeros(1, batch_size, 512).to(DEVICE) for _ in range(2)]
        else:
            h, c = lstm_state

        if dones is not None:
            # Reset hidden state at episode boundaries
            for i, done in enumerate(dones):
                if done:
                    h[:, i].zero_()
                    c[:, i].zero_()

        lstm_out, (h, c) = self.lstm(conv_out, (h, c))
        logits = self.actor(lstm_out)
        values = self.critic(lstm_out).squeeze(-1)

        return logits, values, (h, c)


# Storage buffer for PPO
class PPOBuffer:
    def __init__(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []

    def store(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value):
        advs, returns = [], []
        gae = 0
        values = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + GAMMA * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + GAMMA * LAM * (1 - self.dones[t]) * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[t])
        self.advs = advs
        self.returns = returns

    def get(self):
        return map(np.array, [self.obs, self.actions, self.log_probs, self.advs, self.returns])

    def clear(self):
        self.__init__()


# Main training loop
def train():
    env = gym.make("ALE/Pong-v5")
    num_actions = env.action_space.n
    stacker = FrameStack(4)

    model = PPOActorCritic((4, 84, 84), num_actions).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = PPOBuffer()

    obs, _ = env.reset()
    obs = preprocess(obs)
    obs = stacker.reset(obs)
    lstm_state = None
    done = False
    total_steps = 0

    while total_steps < MAX_FRAMES:
        obs_batch = []
        for _ in range(STEPS_PER_UPDATE):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, value, lstm_state = model(obs_tensor, lstm_state, [done])
                probs = torch.softmax(logits[:, -1], dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            next_obs_proc = preprocess(next_obs)
            obs_stack = stacker.append(next_obs_proc)

            buffer.store(obs, action.item(), reward, done, log_prob.item(), value[:, -1].item())

            obs = obs_stack
            total_steps += 1

            if done:
                print(f"[Step {total_steps}] Episode finished.")
                obs = preprocess(env.reset()[0])
                obs = stacker.reset(obs)
                lstm_state = None

        # Compute advantages and returns
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            _, last_value, _ = model(obs_tensor, lstm_state, [done])
        buffer.compute_returns_and_advantages(last_value[:, -1].item())

        # Optimize policy
        obs_arr, actions, old_log_probs, advs, returns = buffer.get()
        buffer.clear()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(EPOCHS):
            idxs = np.arange(len(obs_arr))
            np.random.shuffle(idxs)
            for start in range(0, len(obs_arr), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idx = idxs[start:end]

                obs_batch = torch.tensor(obs_arr[batch_idx], dtype=torch.float32).unsqueeze(1).to(DEVICE)
                actions_batch = torch.tensor(actions[batch_idx], dtype=torch.int64).to(DEVICE)
                old_logp_batch = torch.tensor(old_log_probs[batch_idx], dtype=torch.float32).to(DEVICE)
                adv_batch = torch.tensor(advs[batch_idx], dtype=torch.float32).to(DEVICE)
                ret_batch = torch.tensor(returns[batch_idx], dtype=torch.float32).to(DEVICE)

                logits, values, _ = model(obs_batch)
                logits = logits[:, -1]
                values = values[:, -1]

                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions_batch)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_logp_batch)
                clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                policy_loss = -torch.min(ratio * adv_batch, clipped_ratio * adv_batch).mean()

                value_loss = (ret_batch - values).pow(2).mean()
                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Trained up to frame {total_steps}, loss: {loss.item():.4f}")

    env.close()


if __name__ == "__main__":
    train()
