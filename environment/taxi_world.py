
import numpy as np
import random
from typing import List, Tuple, Union

# from effects.utils import eff_joint TODO
from effects.effect import JointEffect
from environment.environment import Environment
from environment.hierarchy.taxi_hierarchy import TaxiHierarchy


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
    R_DEFAULT = -1
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
             '|         |',
             '|         |',
             '| |   |   |',
             '| |   |   |']
    actions = ['North', 'East', 'South', 'West', 'Pickup', 'Dropoff']

    def __init__(self, stochastic=True, use_outcomes=True):
        self.stochastic: bool = stochastic
        self.use_outcomes: bool = use_outcomes

        self.dynamic_objects = False

        # Add walls to the map
        # For each direction, stores which positions, (x, y), have a wall in that direction
        self.walls = {
            'N': [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            'E': [(0, 3), (0, 4), (1, 0), (2, 3), (2, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
            'S': [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
            'W': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 0), (3, 3), (3, 4)]
        }

        # List of possible pickup/dropoff locations
        self.locations = [(0, 0), (0, 4), (3, 4), (4, 0)]

        # Hierarchical decomposition
        self.hierarchy = TaxiHierarchy(self)

        # Object instance in state information
        # self.generate_object_maps() TODO

        self.curr_state: List[int] = None
        self.last_action: int = None
        self.last_outcome: int = None
        self.last_reward: float = None
        self.restart()

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
        """Convert state vars to O-O conditions"""
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
        conditions[4] = at_pass
        conditions[5] = state[self.S_PASS] == self.NUM_LOCATIONS
        conditions[6] = pos == self.locations[state[self.S_DEST]]

        return conditions

    def step(self, action: int) -> Union[List[JointEffect], List[int]]:
        """Stochastically apply action to environment"""
        x, y, passenger, destination = self.curr_state
        next_x, next_y, next_passenger = x, y, passenger

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

        # Calculate outcome or effect
        if self.use_outcomes:
            observation = [self.O_NO_CHANGE] * self.NUM_ATT
            # Only one attribute changes at a time in this environment
            if x != next_x:
                observation[self.S_X] = self.A_WEST if next_x < x else self.A_EAST
            elif y != next_y:
                observation[self.S_Y] = self.A_NORTH if next_y < y else self.A_SOUTH
            elif passenger != next_passenger:
                observation[self.S_PASS] = self.A_DROPOFF if passenger == self.NUM_LOCATIONS else self.A_PICKUP
            observation = tuple(observation)
        else:
            # Get all possible JointEffects that could have transformed the current state into the next state
            observation = eff_joint(self.curr_state, next_state)

        # Update current state
        self.curr_state = next_state

        return observation

    def compute_next_loc(self, x: int, y: int, action: int) -> Tuple[int, int]:
        """Deterministically return the result of taking an action"""
        pos = (x, y)
        if action == self.A_NORTH:
            if pos not in self.walls['N']:
                return x, y - 1
        elif action == self.A_EAST:
            if pos not in self.walls['E']:
                return x + 1, y
        elif action == self.A_SOUTH:
            if pos not in self.walls['S']:
                return x, y + 1
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

    def apply_outcome(self, state: int, outcome: List[int]) -> Union[int, np.ndarray]:
        """Compute next state given an outcome"""
        if all(o == self.O_NO_CHANGE for o in outcome):
            return state

        factored_s = self.get_factored_state(state)

        # Only one attribute changes at a time in this environment
        if outcome[self.S_X] == self.A_EAST:
            factored_s[self.S_X] += 1
        elif outcome[self.S_X] == self.A_WEST:
            factored_s[self.S_X] -= 1
        elif outcome[self.S_Y] == self.A_NORTH:
            factored_s[self.S_Y] -= 1
        elif outcome[self.S_Y] == self.A_SOUTH:
            factored_s[self.S_Y] += 1
        elif outcome[self.S_PASS] == self.A_PICKUP:
            factored_s[self.S_PASS] = self.NUM_LOCATIONS
        elif outcome[self.S_PASS] == self.A_DROPOFF:
            factored_s[self.S_PASS] = self.NUM_LOCATIONS + 1

        try:
            return self.get_flat_state(factored_s)
        except ValueError:
            # Outcome returned illegal state
            return state

    def visualize(self):
        self.visualize_state(self.curr_state)

    def visualize_state(self, curr_state):
        x, y, passenger, dest = curr_state
        lines = self.lines
        taxi = '@' if passenger == len(self.locations) else 'O'

        print('-----------')
        for line in lines[:y]:
            print(line)
        print(lines[y][:2 * x + 1] + taxi + lines[y][2 * x + 2:])
        for line in lines[y + 1:]:
            print(line)
        print('-----------')
