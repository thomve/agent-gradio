import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from threading import Thread
import time

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

training = False
rewards = []
episode_data = []
episodes_trained = 0
max_memory = 10000
batch_size = 64
max_limit = 10000

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = deque(maxlen=max_memory)
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

def sample_memory():
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions),
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(next_states, dtype=torch.float32),
        torch.tensor(dones, dtype=torch.float32),
    )

def train_dqn(update_callback=None):
    global training, rewards, episodes_trained, epsilon

    for episode in range(max_limit):
        if not training:
            break

        state = env.reset()[0]
        done = False
        ep_reward = 0

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    action = policy_net(torch.tensor(state).float().unsqueeze(0)).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            ep_reward += reward

            if len(memory) >= batch_size:
                s, a, r, ns, d = sample_memory()
                q_vals = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    max_next_q = target_net(ns).max(1)[0]
                    target = r + gamma * max_next_q * (1 - d)
                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            done = terminated or truncated

        rewards.append(ep_reward)
        episode_data.append({"episode": episode, "reward": ep_reward})
        episodes_trained += 1

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if update_callback:
            update_callback()

        time.sleep(0.05)

def start_training():
    global training
    training = True
    Thread(target=train_dqn, kwargs={"update_callback": None}).start()
    return "Training started."

def stop_training():
    global training
    training = False
    return "Training stopped."

def reset_training():
    global policy_net, target_net, memory, rewards, episodes_trained, epsilon, episode_data
    stop_training()
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    memory.clear()
    rewards.clear()
    episodes_trained = 0
    epsilon = 1.0
    episode_data = []
    return "Training reset."

def get_data():
    return pd.DataFrame(episode_data)