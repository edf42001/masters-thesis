import logging
import numpy as np
import random
from typing import List, Tuple, Union

# from effects.utils import eff_joint TODO
from effects.effect import JointEffect
from environment.environment import Environment


class DoorWorld(Environment):
    """
    A one dimensional world with a switch and a door that needs to be opened to let a taxi through
    """

    # Environment constants
    SIZE_X = 11

    # Actions of agent (parameterized for taxi)
    A_LEFT = 0
    A_RIGHT = 1
    NUM_ACTIONS = 2

    # State variables
    S_X1 = 0  # Taxi1 x
    S_X2 = 1  # Taxi2 x
    S_DOOR = 2 # Door open or close
    NUM_ATT = 3  # TODO: why is this named this

    # Each taxi can be anywhere along the line
    # The door can be open or closed
    STATE_ARITIES = [SIZE_X, SIZE_X, 2]

    # # Stochastic modification to actions
    # MOD = [-1, 0, 1]
    # P_PROB = [0.1, 0.8, 0.1]

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
    NUM_COND = 9  # touch(L/R wall), touch(L/R door), touch(L/R goal), touch(L/R switch), open(door)
    MAX_PARENTS = 4  # This doesn't do anything but I think this number is accurate

    # Outcomes (non-standard OO implementation)
    # All possible outcomes are results of successful actions or no change
    O_NO_CHANGE = NUM_ACTIONS

    actions = ['Left', 'Right']

    def __init__(self, stochastic=True, use_outcomes=True):
        self.stochastic: bool = stochastic
        self.use_outcomes: bool = use_outcomes

        self.dynamic_objects = False

        # Add walls to the map
        # Stores x location of wall, 0 indexed, to the left of the cell
        self.walls = [0, 5, 11]

        # Location of switch
        self.switch = 1

        # Location of door
        self.door = 8

        # Location of goal
        self.goal = 10

        # Object instance in state information
        # self.generate_object_maps() TODO

        self.curr_state: List[int] = None
        self.last_action: int = None
        self.last_outcome: int = None
        self.last_reward: float = None
        self.restart()

    def end_of_episode(self, state: int = None) -> bool:
        """Check if the episode has ended"""
        # Extract state variables from state integer if passed as option
        state = self.get_factored_state(state) if state else self.curr_state

        # Either taxi has made it to the goal
        return state[0] == self.goal or state[1] == self.goal

    def restart(self, init_state=None):
        """Reset state variables to begin new episode"""
        if init_state:
            self.curr_state = init_state
        else:
            # Taxis start at 3 and 6, door is closed
            self.curr_state = [3, 6, 0]

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

    def step(self, action: int, target=None) -> Union[List[JointEffect], List[int]]:
        """Stochastically apply action to environment"""
        x1, x2, door_open = self.curr_state
        next_x1, next_x2, next_door_open = x1, x2, door_open

        self.last_action = action

        # Left action
        if action == 0:
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

    def compute_next_loc(self, x: int, action: int) -> int:
        """Deterministically return the result of mving, given that there are walls"""
        if action == self.A_LEFT:
            if x in self.walls:
                return x
            else:
                return x - 1
        elif action == self.A_RIGHT:
            if (x + 1) in self.walls:
                return x
            else:
                return x + 1
        else:
            logging.error("Unknown action {}".format(action))  # TODO: logging.logerror()
            return x

    def get_reward(self, state: int, next_state: int, action: int) -> float:
        """Get reward of arbitrary state"""
        factored_ns = self.get_factored_state(next_state)

        # Assign reward if any taxi made it to the goal state
        if factored_ns[self.S_X1] == self.goal or factored_ns[self.S_X2] == self.goal:
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
        x1, x2, door_open = curr_state
        print(x1, x2, door_open)
        drawing = ""
        for i in range(self.SIZE_X):
            if i in self.walls:
                drawing += "|"

            if i == x1 or i == x2:
                drawing += "t"
            elif i == self.door:
                if door_open:
                    drawing += "o"
                else:
                    drawing += "d"
            elif i == self.switch:
                drawing += "x"
            elif i == self.goal:
                drawing += "g"
            else:
                drawing += "-"

        drawing += "|"
        print(drawing)
