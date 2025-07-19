import numpy as np
import random

class MazeEnv:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.maze = np.zeros((self.size, self.size), dtype=np.int32)
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)

        # Randomly generate walls
        wall_percentage = 0.25
        for _ in range(int(self.size ** 2 * wall_percentage)):
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (x, y) != self.start and (x, y) != self.goal:
                self.maze[x][y] = 1

        self.position = self.start
        return self._get_obs()

    def _get_obs(self):
        obs = np.copy(self.maze)
        obs[self.position] = 3  # mark agent
        obs[self.goal] = 2      # mark goal
        return obs

    def step(self, action):
        # Actions: 0=up, 1=down, 2=left, 3=right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        nx, ny = self.position[0] + dx, self.position[1] + dy

        if 0 <= nx < self.size and 0 <= ny < self.size and self.maze[nx][ny] != 1:
            self.position = (nx, ny)

        done = self.position == self.goal
        reward = 10 if done else -0.1  # Encourage faster solutions
        return self._get_obs(), reward, done
