"""
The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them off at one of four locations.
There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue) in the 5x5 grid world. The taxi starts off at a random square and the passenger at one of the designated locations.
The goal is move the taxi to the passenger’s location, pick up the passenger, move to the passenger’s desired destination, and drop off the passenger. Once the passenger is dropped off, the episode ends.
The player receives positive rewards for successfully dropping-off the passenger at the correct location. Negative rewards for incorrect attempts to pick-up/drop-off passenger and for each step where another reward is not received.
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict
import random

env = gym.make("Taxi-v3")
n_states = env.observation_space.n
n_actions = env.action_space.n

alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor
initial_epsilon = 1.
min_epsilon = 0.05
decay_rate = 0.995
n_planning_steps = 10
n_episodes = 10000


Q = np.zeros((n_states, n_actions))
model = defaultdict(lambda: (0, 0))  # (next_state, reward)
seen_state_actions = set()

# Epsilon-greedy policy
def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])


for episode in range(n_episodes):
    state = env.reset()[0]
    epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** episode))  # Exploration rate
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update Q-table with real experience
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state][best_next_action]
        Q[state][action] += alpha * (td_target - Q[state][action])

        # Update model with observed transition
        model[(state, action)] = (next_state, reward)
        seen_state_actions.add((state, action))

        # Planning: simulate experience from model
        for _ in range(n_planning_steps):
            sim_state, sim_action = random.choice(list(seen_state_actions))
            sim_next_state, sim_reward = model[(sim_state, sim_action)]
            sim_best_next_action = np.argmax(Q[sim_next_state])
            sim_td_target = sim_reward + gamma * Q[sim_next_state][sim_best_next_action]
            Q[sim_state][sim_action] += alpha * (sim_td_target - Q[sim_state][sim_action])

        state = next_state

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean([np.max(Q[s]) for s in range(n_states)])
            print(f"Episode {episode + 1} - Avg Max Q-value: {avg_reward:.3f}")

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1} finished")

def evaluate_agent():
    wins = 0
    for _ in range(100):
        state = env.reset()[0]
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done and reward == 1.0:
                wins += 1
    print(f"Agent won {wins}/100 episodes")

evaluate_agent()
