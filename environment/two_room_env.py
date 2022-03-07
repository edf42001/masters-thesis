from typing import Tuple
import numpy as np

from effects.effect import JointEffect
from environment.environment import Environment


class TwoRoomEnv(Environment):
    # Actions (How to track options?)
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    NUM_ACTIONS = 4

    # Room dimensions, start and goal locations
    WIDTH = 20
    HEIGHT = 10

    # State arities are the size of each dimension
    STATE_ARITIES = [WIDTH, HEIGHT]

    GOAL = [17, 7]
    START = [2, 2]

    def __init__(self):
        """Creates the room grid, bottom right is 0, 0"""
        self.agent_pos = self.START  # Agent starts here

        # Define a wall going down the middle of the room, with a gap in the middle
        # Although numpy arrays use a row, col (y, x) we will work with them as though they are x, y
        self.walls = np.zeros((self.WIDTH, self.HEIGHT), dtype='bool')
        self.walls[10, 0:4] = True
        self.walls[10, 6:self.HEIGHT] = True

    def step(self, action: int) -> Tuple[int, float]:
        """Step the environment with a primitive action, and return next state, done, reward"""

        # Move agent
        self.agent_pos = self.compute_next_location(self.agent_pos[0], self.agent_pos[1], action)

        # Check if at goal
        done = (self.agent_pos == self.GOAL)
        reward = 1 if done else 0

        return self.get_state(), reward

    def compute_next_location(self, x: int, y: int, action: int) -> Tuple[int, int]:
        """Deterministically compute next location"""

        # Check for each action, if the agent moves off the screen, or if there is a wall there,
        # and if so, don't move, otherwise do move
        if action == self.UP:
            if y + 1 < self.HEIGHT and not self.walls[x][y+1]:
                return x, y+1
        elif action == self.RIGHT:
            if x + 1 < self.WIDTH and not self.walls[x+1][y]:
                return x+1, y
        elif action == self.DOWN:
            if y - 1 >= 0 and not self.walls[x][y-1]:
                return x, y-1
        elif action == self.LEFT:
            if x - 1 >= 0 and not self.walls[x-1][y]:
                return x-1, y
        else:
            print("BAD ACTION: {}".format(action))

        return x, y

    def restart(self):
        """Resets the environment to the beginning of an episode"""
        self.agent_pos = self.START

    def end_of_episode(self, state: int = None) -> bool:
        # Can optionally take in a state, for now just return if agent pos is at goal
        # Careful, tuples are different from lists!
        return self.agent_pos[0] == self.GOAL[0] and self.agent_pos[1] == self.GOAL[1]

    def get_state(self) -> int:
        """Returns flattened (number from 1 to N) state"""

        # These multipliers are called arities
        return self.agent_pos[0] * self.HEIGHT + self.agent_pos[1]

    def visualize(self):
        self.visualize_state(self.agent_pos)

    def visualize_state(self, curr_state):
        x, y = curr_state  # Unpack x and y from agent location
        # For visualization (TODO: See if storing as global variable is faster)
        print('')
        for row in range(self.HEIGHT - 1, -1, -1):
            line = "|"

            # Generate scene from walls, agent, and empty space
            for col in range(self.WIDTH):
                if row == y and col == x:
                    line += "@"
                elif row == self.GOAL[1] and col == self.GOAL[0]:
                    line += 'O'
                elif self.walls[col, row]:
                    line += "X"
                else:
                    line += "."
            line += "|"
            print(line)
        print('')
