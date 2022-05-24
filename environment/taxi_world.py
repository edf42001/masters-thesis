import logging
import numpy as np
import random
from typing import List, Tuple, Union
import cv2

from effects.utils import eff_joint, get_effects
from effects.effect import JointEffect, NoChange
from environment.environment import Environment


class TaxiWorld(Environment):

    # Environment constants
    SIZE_X = 5
    SIZE_Y = 5
    NUM_LOCATIONS = 4

    # Actions of agent
    A_NORTH = 0
    A_EAST = 1
    A_SOUTH = 2
    A_WEST = 3
    A_PICKUP = 4
    A_DROPOFF = 5
    NUM_ACTIONS = 6

    # State variables
    S_X = 0
    S_Y = 1
    S_PASS = 2
    S_DEST = 3
    NUM_ATT = 4
    STATE_ARITIES = [SIZE_X, SIZE_Y, NUM_LOCATIONS + 2, NUM_LOCATIONS]

    # Stochastic modification to actions
    MOD = [-1, 0, 1]
    P_PROB = [0.1, 0.8, 0.1]

    # Rewards
    R_DEFAULT = -0.5
    R_SUCCESS = 10

    # Object descriptions
    OB_TAXI = 0
    OB_PASS = 1
    OB_DEST = 2
    OB_COUNT = [1, 1, 1]
    OB_ARITIES = [2, 1, 1]

    # Conditions
    NUM_COND = 7  # touch_N/E/S/W, on(Agent,Passenger), in(Agent,Passenger), on(Agent,Dest)
    MAX_PARENTS = 3

    # Outcomes (non-standard OO implementation)
    # All possible outcomes are results of successful actions or no change
    O_NO_CHANGE = NUM_ACTIONS

    # For visualization
    lines = ['|   |     |',
             '|   |     |',
             '|         |',
             '| |   |   |',
             '| |   |   |']

    ACTION_NAMES = ['North', 'East', 'South', 'West', 'Pickup', 'Dropoff']
    ATT_NAMES = ["x", "y", "pass", "dest"]

    def __init__(self, stochastic=True, shuffle_actions=False):
        self.stochastic: bool = stochastic

        # Add walls to the map
        # For each direction, stores which positions, (x, y), have a wall in that direction
        self.walls = {
            'N': [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
            'E': [(0, 0), (0, 1), (1, 3), (1, 4), (2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
            'S': [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            'W': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (2, 3), (2, 4), (3, 0), (3, 1)]
        }

        # List of possible pickup/dropoff locations
        self.locations = [(0, 4), (4, 4), (3, 0), (0, 0)]

        # Object instance in state information
        self.generate_object_maps()

        self.curr_state: List[int] = None
        self.last_action: int = None
        self.last_outcome: int = None
        self.last_reward: float = None

        # Restart to begin episode
        self.restart()

        # For testing purposes only:
        # Taxi position and pickup / dropoff
        self.curr_state = [0, 2, 3, 1]

        # An action map, for if the actions are shuffled around. Used for learning action mappings
        self.action_map = {i: i for i in range(self.NUM_ACTIONS)}
        if shuffle_actions:
            actions = list(range(self.NUM_ACTIONS))
            random.shuffle(actions)
            self.action_map = {i: actions[i] for i in range(self.NUM_ACTIONS)}

    def end_of_episode(self, state: int = None) -> bool:
        """Check if the episode has ended"""
        state = self.get_factored_state(state) if state else self.curr_state
        return state[self.S_PASS] == self.NUM_LOCATIONS + 1

    def restart(self, init_state=None):
        """Reset state variables to begin new episode"""
        if init_state:
            self.curr_state = [0, 1, init_state[0], init_state[1]]
        else:
            # Taxi starts at (0, 1)
            # Randomly choose passenger and destination locations
            passenger, destination = random.sample([0, 1, 2, 3], 2)
            self.curr_state = [0, 1, passenger, destination]

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

        # Touch_N/E/S/W, on(Agent,Pass), in(Agent,Pass), on(Agent,Dest)
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
        x, y, passenger, destination = self.curr_state
        next_x, next_y, next_passenger = x, y, passenger

        # Lookup new action in action map
        action = self.action_map[action]

        self.last_action = action

        # Movement action
        if action <= 3:
            if self.stochastic:
                # Randomly change action according to stochastic property of env
                modification = np.random.choice(self.MOD, p=self.P_PROB)
                action = (action + modification) % 4
            next_x, next_y = self.compute_next_loc(x, y, action)
        # Pickup action
        elif action == 4:
            pos = (x, y)
            # Check if taxi already holds passenger
            if passenger == self.NUM_LOCATIONS:
                next_passenger = self.NUM_LOCATIONS
            # Check if taxi is on correct pickup location
            elif passenger < self.NUM_LOCATIONS and pos == self.locations[passenger]:
                next_passenger = self.NUM_LOCATIONS
        # Dropoff action
        else:
            pos = (x, y)
            # Check if passenger is in taxi and taxi is on the destination
            if passenger == self.NUM_LOCATIONS and pos == self.locations[destination]:
                next_passenger = self.NUM_LOCATIONS + 1

        # Make updates to state
        # Destination status does not change
        next_state = [next_x, next_y, next_passenger, destination]

        # Assign reward
        if next_passenger == self.NUM_LOCATIONS + 1:
            self.last_reward = self.R_SUCCESS
        else:
            self.last_reward = self.R_DEFAULT

        # Calculate effects
        # Get all possible JointEffects that could have transformed the current state into the next state
        # observation = eff_joint(self.curr_state, next_state)

        # Instead of the above, lets return the actual effects for each attribute
        # Or an empty dict for "failure state"
        observation = dict()
        if self.curr_state != next_state:
            for att in range(self.NUM_ATT):
                # The passenger state is categorical: at a specific place, or not
                # at that place
                is_bool = att in [self.S_PASS]
                observation[att] = get_effects(att, self.curr_state, next_state, is_bool=is_bool)

                # if not observation[att]:
                #     observation[att] = [NoChange()]

        # Update current state
        self.curr_state = next_state

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

    def visualize(self):
        self.draw_taxi(self.curr_state, delay=1)
        # self.visualize_state(self.curr_state)

    def visualize_state(self, curr_state):
        x, y, passenger, dest = curr_state
        dest_x, dest_y = self.locations[dest]
        lines = self.lines
        taxi = '@' if passenger == len(self.locations) else 'O'

        pass_x, pass_y = -1, -1
        if passenger < len(self.locations):
            pass_x, pass_y = self.locations[passenger]

        ret = ""
        ret += '-----------\n'
        for i, line in enumerate(lines):
            for j, c in enumerate(line):
                iy = self.SIZE_Y - i - 1  # Flip y axis vertically
                if iy == y and j == 2 * x + 1:
                    ret += taxi
                elif iy == dest_y and j == 2 * dest_x + 1:
                    ret += "g"
                elif iy == pass_y and j == 2 * pass_x + 1:
                    ret += "p"
                else:
                    ret += c
            ret += '\n'
        ret += '-----------\n'

        # Do the display
        print(ret)

    def draw_taxi(self, state, delay=100):
        GRID_SIZE = 100
        WIDTH = self.SIZE_X
        HEIGHT = self.SIZE_Y

        x, y, passenger, dest = state

        # Blank white square
        img = 255 * np.ones((HEIGHT * GRID_SIZE, WIDTH * GRID_SIZE, 3))

        # Draw pickup and dropoff zones
        colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 128, 128]]
        for loc, color in zip(self.locations, colors):
            bottom_left_x = loc[0] * GRID_SIZE
            bottom_left_y = (HEIGHT - loc[1]) * GRID_SIZE
            cv2.rectangle(img, (bottom_left_x, bottom_left_y), (bottom_left_x + GRID_SIZE, bottom_left_y - GRID_SIZE),
                          thickness=-1, color=color)

        # Mark goal with small circle
        goal_x = self.locations[dest][0]
        goal_y = self.locations[dest][1]
        cv2.circle(img, (int((goal_x + 0.5) * GRID_SIZE), int((HEIGHT - (goal_y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.05),
                   thickness=-1, color=[0, 0, 0])

        # Draw taxi
        cv2.circle(img, (int((x + 0.5) * GRID_SIZE), int((HEIGHT - (y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.3),
                   thickness=-1, color=[0, 0, 0])

        # Draw passenger
        if passenger >= len(self.locations):
            pass_x = x
            pass_y = y
        else:
            pass_x = self.locations[passenger][0]
            pass_y = self.locations[passenger][1]

        cv2.circle(img, (int((pass_x + 0.5) * GRID_SIZE), int((HEIGHT - (pass_y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.2),
                   thickness=-1, color=[0.5, 0.5, 0.5])

        # Draw horizontal and vertical walls
        for i in range((self.SIZE_X + 1) * (self.SIZE_Y + 1)):
            x = i % (self.SIZE_X + 1)
            y = int(i / (self.SIZE_Y + 1))

            pos = (x, y)

            if pos in self.walls['N']:
                cv2.line(img, (x * GRID_SIZE, (HEIGHT - y-1) * GRID_SIZE), ((x+1) * GRID_SIZE, (HEIGHT - y-1) * GRID_SIZE),
                         thickness=3, color=[0, 0, 0])
            if pos in self.walls['E']:
                cv2.line(img, ((x+1) * GRID_SIZE, (HEIGHT - y) * GRID_SIZE), ((x+1) * GRID_SIZE, (HEIGHT - y-1) * GRID_SIZE),
                         thickness=3, color=[0, 0, 0])
            if pos in self.walls['S']:
                cv2.line(img, (x * GRID_SIZE, (HEIGHT - y) * GRID_SIZE), ((x+1) * GRID_SIZE, (HEIGHT - y) * GRID_SIZE),
                         thickness=3, color=[0, 0, 0])
            if pos in self.walls['W']:
                cv2.line(img, (x * GRID_SIZE, (HEIGHT - y) * GRID_SIZE), (x * GRID_SIZE, (HEIGHT - y-1) * GRID_SIZE),
                         thickness=3, color=[0, 0, 0])

        cv2.imshow("Taxi World", img)
        cv2.waitKey(delay)

    def get_rmax(self) -> float:
        """The maximum reward available in the environment"""
        return self.R_SUCCESS + 5  # In my implementation, this value is 15

    def get_action_map(self) -> dict:
        """Returns the real action map for debugging purposes"""
        return self.action_map
