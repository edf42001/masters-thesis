import logging
import numpy as np
import random
from typing import List, Tuple, Union
import cv2

from effects.utils import get_effects
from effects.effect import JointEffect, NoChange
from environment.environment import Environment


class SokobanWorld(Environment):

    # Environment constants
    SIZE_X = 5
    SIZE_Y = 5

    # Actions of agent
    A_NORTH = 0
    A_EAST = 1
    A_SOUTH = 2
    A_WEST = 3
    NUM_ACTIONS = 4

    # State variables
    S_X = 0
    S_Y = 1
    S_BLOCK1_X = 2
    S_BLOCK1_Y = 3
    S_BLOCK2_X = 4
    S_BLOCK2_Y = 5
    NUM_ATT = 6

    # Each x/y variable has 5 possibilities
    STATE_ARITIES = [SIZE_X, SIZE_Y] * 3

    # Rewards
    R_DEFAULT = -1
    R_GOAL = 3
    R_DONE = 10

    # Object descriptions
    OB_GUY = 0
    OB_BLOCK = 1
    OB_COUNT = [1, 2]
    OB_ARITIES = [2, 2]

    # Conditions
    NUM_COND = 7  # touch_N/E/S/W, on(Agent,Passenger), in(Agent,Passenger), on(Agent,Dest)

    # Outcomes (non-standard OO implementation)
    # All possible outcomes are results of successful actions or no change
    O_NO_CHANGE = NUM_ACTIONS

    ACTION_NAMES = ['North', 'East', 'South', 'West']
    ATT_NAMES = ["x", "y", "block1x", "block1y", "block2x", "block2y"]

    def __init__(self, stochastic=True):
        self.stochastic: bool = stochastic

        # Add walls to the map. Walls are full squares on the map
        self.walls = [(0, 4), (1, 3), (3, 1), (4, 0)]

        # List of goal locations
        self.goals = [(0, 3), (4, 1)]

        # Object instance in state information
        self.generate_object_maps()

        self.curr_state: List[int] = None
        self.last_action: int = None
        self.last_reward: float = None

        # Restart to begin episode
        self.restart()

        # For testing purposes only: Set initial state
        self.curr_state = [1, 1, 2, 1, 3, 3]

    def end_of_episode(self, state: int = None) -> bool:
        """Check if the episode has ended"""
        state = self.get_factored_state(state) if state else self.curr_state

        # When all blocks are on goals, the episode has ended
        block1 = (state[self.S_BLOCK1_X], state[self.S_BLOCK2_X])
        block2 = (state[self.S_BLOCK2_X], state[self.S_BLOCK2_Y])

        return block1 in self.goals and block2 in self.goals

    def restart(self):
        """Reset state variables to begin new episode"""
        self.curr_state = [1, 1, 2, 1, 3, 3]

    def get_condition(self, state) -> List[bool]:
        """
        Convert state vars to OO conditions
        This basically enables the perfect knowledge assumption, that the agent knows the conditions for every state
        This sortof makes sense, because you can just look at it as a human
        """
        # Convert flat state to factored state
        if isinstance(state, (int, np.integer)):
            state = self.get_factored_state(state)

        conditions = [False] * self.NUM_COND
        pos = (state[self.S_X], state[self.S_Y])

        # Check if taxi is at passenger location but passenger is not picked up
        if state[self.S_PASS] < self.NUM_LOCATIONS:
            at_pass = pos == self.locations[state[self.S_PASS]]
        else:
            at_pass = False

        # Touch_N/E/S/W, on(Agent,Dest), in(Agent,Pass), on(Agent,Pass)
        conditions[0] = pos in self.walls['N']
        conditions[1] = pos in self.walls['E']
        conditions[2] = pos in self.walls['S']
        conditions[3] = pos in self.walls['W']
        conditions[4] = pos == self.locations[state[self.S_DEST]]
        conditions[5] = at_pass
        conditions[6] = state[self.S_PASS] == self.NUM_LOCATIONS

        return conditions

    def step(self, action: int) -> Union[List[JointEffect], List[int]]:
        """Stochastically apply action to environment"""
        x, y, block1x, block1y, block2x, block2y = self.curr_state
        next_x, next_y, next_block1x, next_block1y, next_block2x, next_block2y = x, y, block1x, block1y, block2x, block2y
        blocks = [(block1x, block1y), (block2x, block2y)]

        self.last_action = action

        # TODO: see how dallan did dynamic duplicate objects

        
        # # Assign reward
        # if next_passenger == self.NUM_LOCATIONS + 1:
        #     self.last_reward = self.R_SUCCESS
        # else:
        #     self.last_reward = self.R_DEFAULT

        # Calculate effects
        # Get all possible JointEffects that could have transformed the current state into the next state
        # observation = eff_joint(self.curr_state, next_state)

        # Instead of the above, lets return the actual effects for each attribute
        # Or an empty dict for "failure state"
        observation = dict()
        # if self.curr_state != next_state:
        #     for att in range(self.NUM_ATT):
        #         # The passenger state is categorical: at a specific place, or not
        #         # at that place
        #         is_bool = att in [self.S_PASS]
        #         observation[att] = get_effects(att, self.curr_state, next_state, is_bool=is_bool)
        #
        #         # if not observation[att]:
        #         #     observation[att] = [NoChange()]
        #
        # # Update current state
        # self.curr_state = next_state

        return observation

    def compute_next_loc(self, x: int, y: int, action: int) -> Tuple[int, int]:
        """Deterministically return the result of taking an action"""
        pos = (x, y)
        if action == self.A_NORTH:
            if pos not in self.walls['N']:
                return x, y + 1
        elif action == self.A_EAST:
            if pos not in self.walls['E']:
                return x + 1, y
        elif action == self.A_SOUTH:
            if pos not in self.walls['S']:
                return x, y - 1
        elif action == self.A_WEST:
            if pos not in self.walls['W']:
                return x - 1, y

        return x, y

    def get_reward(self, state: int, next_state: int, action: int) -> float:
        """Get reward of arbitrary state"""
        factored_s = self.get_factored_state(next_state)

        # Assign reward if in goal state
        if factored_s[self.S_PASS] == self.NUM_LOCATIONS + 1:
            return self.R_SUCCESS
        else:
            return self.R_DEFAULT

    def unreachable_state(self, from_state: int, to_state: int) -> bool:
        """
        If a next state is unreachable from the current state, we can ignore it for the purpose of value iteration
        For example, if the current state has destination = 1, we can never reach a state destination = 2.
        Destination is static during an episode.
        """
        from_factored = self.get_factored_state(from_state)
        to_factored = self.get_factored_state(to_state)

        # If the destinations are different, this is unreachable:
        if from_factored[self.S_DEST] != to_factored[self.S_DEST]:
            return True

        # The passenger pickup location is represented as 0-N-1 if the passenger is
        # at the location, N if in the taxi, and N+1 if the episode has ended and
        # the passenger has been dropped off. THe only unreachable states is if the
        # pickup locations differ, i.e. both states have passenger at a pickup location
        # and they are not the same.
        # What to do about end of episode state?
        elif from_factored[self.S_PASS] < len(self.locations) and \
             to_factored[self.S_PASS] < len(self.locations) and \
             from_factored[self.S_PASS] != to_factored[self.S_PASS]:
            return True
        else:
            return False

    def get_interacting_objects(self, state: int):
        """Returns a representation of all objects that are interacting with each other and how in the current env"""
        state = self.get_factored_state(state)
        x, y, passenger, dest = state

        pos = (x, y)
        pass_in_taxi = passenger >= len(self.locations)
        pass_loc = self.locations[passenger] if not pass_in_taxi else (0, 0)
        dest_loc = self.locations[dest]

        interactions = dict()

        interactions["wall"] = dict()
        interactions["wall"]["touchN"] = pos in self.walls['N']
        interactions["wall"]["touchE"] = pos in self.walls['E']
        interactions["wall"]["touchS"] = pos in self.walls['S']
        interactions["wall"]["touchW"] = pos in self.walls['W']
        interactions["wall"]["on"] = False
        interactions["wall"]["in"] = False

        interactions["pass"] = dict()
        interactions["pass"]["touchN"] = not pass_in_taxi and (x, y+1) == pass_loc
        interactions["pass"]["touchE"] = not pass_in_taxi and (x+1, y) == pass_loc
        interactions["pass"]["touchS"] = not pass_in_taxi and (x, y-1) == pass_loc
        interactions["pass"]["touchW"] = not pass_in_taxi and (x-1, y) == pass_loc
        interactions["pass"]["on"] = not pass_in_taxi and (x, y) == pass_loc
        interactions["pass"]["in"] = pass_in_taxi

        interactions["dest"] = dict()
        interactions["dest"]["touchN"] = (x, y+1) == dest_loc
        interactions["dest"]["touchE"] = (x+1, y) == dest_loc
        interactions["dest"]["touchS"] = (x, y-1) == dest_loc
        interactions["dest"]["touchW"] = (x-1, y) == dest_loc
        interactions["dest"]["on"] = (x, y) == dest_loc
        interactions["dest"]["in"] = False

        return interactions

    def visualize(self):
        # self.draw_taxi(self.curr_state, delay=1)
        self.visualize_state(self.curr_state)

    def visualize_state(self, state):
        x, y, block1x, block1y, block2x, block2y = state

        blocks = [(block1x, block1y), (block2x, block2y)]

        # Coordinate transform for printing
        goals = [(y, 2*x+1) for (x, y) in self.goals]
        walls = [(y, 2*x+1) for (x, y) in self.walls]
        blocks = [(y, 2*x+1) for (x, y) in blocks]

        ret = ""
        ret += '-----------\n'
        for i in range(self.SIZE_Y):
            for j in range(2*self.SIZE_X):
                iy = self.SIZE_Y - i - 1  # Flip y axis vertically

                if iy == y and j == 2 * x + 1:
                    ret += "O"
                elif (iy, j) in blocks:
                    ret += "b"
                elif (iy, j) in goals:
                    ret += "."
                elif (iy, j) in walls:
                    ret += "X"
                else:
                    ret += " "
            ret += '\n'
        ret += '-----------\n'

        # Do the display
        print(ret)

    def draw_world(self, state, delay=100):
        GRID_SIZE = 100
        WIDTH = self.SIZE_X
        HEIGHT = self.SIZE_Y

        # Have to name these guy_x or they get overwritten by the variables in the for loops
        guy_x, guy_y, block1x, block1y, block2x, block2y = state

        blocks = [(block1x, block1y), (block2x, block2y)]

        # Blank white square
        img = 255 * np.ones((HEIGHT * GRID_SIZE, WIDTH * GRID_SIZE, 3))

        # Draw blocks
        for (x, y) in blocks:
            color = [0, 0, 0.8]
            bottom_left_x = x * GRID_SIZE
            bottom_left_y = (HEIGHT - y) * GRID_SIZE
            cv2.rectangle(img, (bottom_left_x, bottom_left_y), (bottom_left_x + GRID_SIZE, bottom_left_y - GRID_SIZE),
                          thickness=-1, color=color)

        # Draw walls
        for (x, y) in self.walls:
            color = [0.05, 0, 0]
            bottom_left_x = x * GRID_SIZE
            bottom_left_y = (HEIGHT - y) * GRID_SIZE
            cv2.rectangle(img, (bottom_left_x, bottom_left_y), (bottom_left_x + GRID_SIZE, bottom_left_y - GRID_SIZE),
                          thickness=-1, color=color)

        # Draw goals
        for (x, y) in self.goals:
            color = [0.2, 0.7, 0]
            bottom_left_x = x * GRID_SIZE
            bottom_left_y = (HEIGHT - y) * GRID_SIZE
            cv2.rectangle(img, (bottom_left_x, bottom_left_y), (bottom_left_x + GRID_SIZE, bottom_left_y - GRID_SIZE),
                          thickness=-1, color=color)

        # Mark Guy with small circle
        cv2.circle(img, (int((guy_x + 0.5) * GRID_SIZE), int((HEIGHT - (guy_y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.4),
                   thickness=-1, color=[0, 0.4, 0.3])

        #
        #     if pos in self.walls['N']:
        #         cv2.line(img, (x * GRID_SIZE, (HEIGHT - y-1) * GRID_SIZE), ((x+1) * GRID_SIZE, (HEIGHT - y-1) * GRID_SIZE),
        #                  thickness=3, color=[0, 0, 0])
        #     if pos in self.walls['E']:
        #         cv2.line(img, ((x+1) * GRID_SIZE, (HEIGHT - y) * GRID_SIZE), ((x+1) * GRID_SIZE, (HEIGHT - y-1) * GRID_SIZE),
        #                  thickness=3, color=[0, 0, 0])
        #     if pos in self.walls['S']:
        #         cv2.line(img, (x * GRID_SIZE, (HEIGHT - y) * GRID_SIZE), ((x+1) * GRID_SIZE, (HEIGHT - y) * GRID_SIZE),
        #                  thickness=3, color=[0, 0, 0])
        #     if pos in self.walls['W']:
        #         cv2.line(img, (x * GRID_SIZE, (HEIGHT - y) * GRID_SIZE), (x * GRID_SIZE, (HEIGHT - y-1) * GRID_SIZE),
        #                  thickness=3, color=[0, 0, 0])

        cv2.imshow("Sokoban", img)
        cv2.waitKey(delay)

    def get_rmax(self) -> float:
        """The maximum reward available in the environment"""
        return self.R_SUCCESS + 5  # In my implementation, this value is 15

    def get_action_map(self) -> dict:
        """Returns the real action map for debugging purposes"""
        return self.action_map
