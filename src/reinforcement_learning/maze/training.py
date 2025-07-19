import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

from agent import DQN
from environment import MazeEnv


def train_agent():
    env = MazeEnv(size=10)
    print("The maze is\n", env.maze)
    num_actions = 4
    model = DQN(input_dim=10, num_actions=num_actions)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = deque(maxlen=10000)

    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    for episode in range(500):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0

        for t in range(200):  # Max steps per episode
            if random.random() < epsilon:
                action = random.randint(0, num_actions - 1)
            else:
                with torch.no_grad():
                    q_vals = model(state.unsqueeze(0))
                    action = torch.argmax(q_vals).item()

            next_state, reward, done = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            memory.append((state, action, reward, next_state_tensor, done))
            state = next_state_tensor
            total_reward += reward

            if done:
                break

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards)
                next_states = torch.stack(next_states)
                dones = torch.tensor(dones)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = model(next_states).max(1)[0]
                target_q = rewards + gamma * next_q_values * (1 - dones.int())

                loss = F.mse_loss(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")


if __name__ == "__main__":
    train_agent()