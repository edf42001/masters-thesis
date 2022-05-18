import numpy as np
from typing import List, Tuple, Union

from effects.utils import eff_joint
from effects.effect import JointEffect
from environment import Environment
from environment.hierarchy.heist_hier import HeistHierarchy


class Heist(Environment):
    # Environment constants
    SIZE_X = 5
    SIZE_Y = 5

    # Actions of agent
    A_NORTH = 0
    A_EAST = 1
    A_SOUTH = 2
    A_WEST = 3
    A_PICKUP = 4
    A_UNLOCK = 5
    NUM_ACTIONS = 6

    # State variables
    S_X = 0
    S_Y = 1
    S_KEY_1 = 2
    S_KEY_2 = 3
    S_KEY_3 = 4
    S_KEY_4 = 5
    S_KEY_5 = 6
    S_LOCK_1 = 7
    S_LOCK_2 = 8
    S_LOCK_3 = 9
    S_GEM = 10
    NUM_ATT = 11

    # Agent can be anywhere in grid
    # Keys are existing, not existing, or held by agent
    # Locks are locked or unlocked
    # Gem is not held or held
    S_ARITIES = [SIZE_X, SIZE_Y] + [3] * 5 + [2] * 4

    # Stochastic modification to actions
    MOD = [-1, 0, 1]
    P_PROB = [0.1, 0.8, 0.1]

    # Rewards
    R_DEFAULT = -1
    R_UNLOCK = 5
    R_SUCCESS = 10

    # Object descriptions
    OB_AGENT = 0
    OB_KEY = 1
    OB_LOCK = 2
    OB_GEM = 3
    OB_COUNT = [1, 5, 3, 1]
    OB_ARITIES = [2, 1, 1, 1]

    # Conditions
    # touch(N/E/S/W wall), touch(N/E/S/W lock), on(key/gem), hold(key/gem)
    NUM_COND = 12
    MAX_PARENTS = 3

    # Outcomes (non-standard OO implementation)
    O_NO_CHANGE = NUM_ACTIONS

    def __init__(self, stochastic=True, use_outcomes=True):
        self.stochastic = stochastic
        self.use_outcomes = use_outcomes

        self.dynamic_objects = True

        # Add walls to the map
        # For each direction, stores which positions, (x, y), have a wall in that direction
        #   x 0 1 2 3 4
        # y   _ _ _ _ _
        # 0  | |_ _    |
        # 1  |  _    | |
        # 2  | | | | |_|
        # 3  | |_ _| | |
        # 4  |_ _|_ _ _|
        self.walls = {
            'N': {(0, 0), (1, 0), (1, 1), (1, 2), (1, 4), (2, 0), (2, 1), (2, 4), (3, 0), (4, 0), (4, 3)},
            'E': {(0, 0), (0, 2), (0, 3), (1, 2), (1, 4), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1),
                  (4, 2), (4, 3), (4, 4)},
            'S': {(0, 4), (1, 0), (1, 1), (1, 3), (1, 4), (2, 0), (2, 3), (2, 4), (3, 4), (4, 2), (4, 4)},
            'W': {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 2), (1, 3), (2, 2), (2, 4), (3, 2), (3, 3),
                  (4, 1), (4, 2), (4, 3)}
        }

        # List of possible key locations
        self.keys = [(0, 0), (1, 0), (1, 2), (2, 4), (4, 2)]

        # List of lock locations
        self.locks = [(0, 2), (0, 3), (0, 4)]

        # Location of gem
        self.gem = (1, 4)

        # Hierarchical decomposition
        self.hierarchy = HeistHierarchy(self)

        # Object instance and class in state information
        self.generate_object_maps()

        # Set RL variables
        self.curr_state: List[int] = None
        self.last_action: int = None
        self.last_reward: float = None
        self.restart()

    def EOE(self, state: int = None) -> bool:
        """Check if the episode has ended"""
        state = self.get_factored_state(state) if state else self.curr_state
        return state[self.S_GEM] == 1

    def restart(self, init_state=None):
        """Reset state variables to begin new episode"""
        if init_state:
            self.curr_state = init_state
        else:
            # Agent starts at (2, 1)
            # Randomly choose 3 of 5 possible key locations
            # All locks begin locked
            # Gem begins not held
            key_idx = np.random.choice(5, size=3, replace=False)
            self.curr_state = [2, 1] + [int(i in key_idx) for i in range(5)] + [1, 1, 1, 0]

    def get_condition(self, state):
        """Convert state vars to O-O conditions, return grounded instances for each True condition"""
        # Convert flat state to factored state
        if isinstance(state, (int, np.integer)):
            state = self.get_factored_state(state)

        conditions = [False] * self.NUM_COND
        groundings = [None] * self.NUM_COND

        pos = (state[self.S_X], state[self.S_Y])

        # touch(N/E/S/W wall): indexes 0 through 3
        conditions[0] = pos in self.walls['N']
        conditions[1] = pos in self.walls['E']
        conditions[2] = pos in self.walls['S']
        conditions[3] = pos in self.walls['W']

        # Get the status of each key/lock/gem
        keys = state[self.S_KEY_1: self.S_KEY_5 + 1]
        locks = state[self.S_LOCK_1: self.S_LOCK_3 + 1]
        gem = state[self.S_GEM]

        # Get the (possibly illegal) surrounding locations
        surroundings = [(pos[0], pos[1] - 1),
                        (pos[0] + 1, pos[1]),
                        (pos[0], pos[1] + 1),
                        (pos[0] - 1, pos[1])]

        # touch(N/E/S/W lock): indexes 4 through 7
        touch_lock_base_idx = 4
        # Check if a lock exists at surrounding locations
        for i, s in enumerate(surroundings):
            try:
                # Check that a locked lock (value 1) exists at s
                lock_idx = self.locks.index(s)
                if locks[lock_idx] == 1:
                    lock_state_idx = self.S_LOCK_1 + lock_idx
                    conditions[touch_lock_base_idx + i] = True
                    groundings[touch_lock_base_idx + i] = self.state_index_instance_map[lock_state_idx]
                    break
            except ValueError:
                # s is not a valid location for a lock
                pass

        # on(key): index 8
        try:
            # Check that an available key (value 1) exists at pos
            key_idx = self.keys.index(pos)
            if keys[key_idx] == 1:
                key_state_idx = self.S_KEY_1 + key_idx
                conditions[8] = True
                groundings[8] = self.state_index_instance_map[key_state_idx]
        except ValueError:
            # pos is not a valid location for a key
            pass

        # on(gem): index 9
        conditions[9] = pos == self.gem and gem == 0  # There is only one gem so we don't really need to ground it

        # hold(key): index 10
        try:
            key_idx = keys.index(2)
            key_state_idx = self.S_KEY_1 + key_idx
            conditions[10] = True
            groundings[10] = self.state_index_instance_map[key_state_idx]
        except ValueError:
            # There are no held keys
            pass

        # hold(gem): index 11
        conditions[11] = gem == 1  # There is only one gem so we don't really need to ground it

        return conditions, groundings

    def perform_action(self, action: int) -> List[JointEffect]:
        """Stochastically apply action to environment"""
        state = self.curr_state
        x, y, keys, locks, gem = \
            state[self.S_X], state[self.S_Y], state[self.S_KEY_1: self.S_KEY_5 + 1], \
            state[self.S_LOCK_1: self.S_LOCK_3 + 1], state[self.S_GEM]
        next_x, next_y, next_keys, next_locks, next_gem = x, y, keys.copy(), locks.copy(), gem

        pos = (x, y)

        self.last_action = action

        # Movement action
        if action <= 3:
            if self.stochastic:
                # Randomly change action according to stochastic property of env
                modification = np.random.choice(self.MOD, p=self.P_PROB)
                action = (action + modification) % 4

            next_x, next_y = self.compute_next_loc(x, y, locks, action)
        # Pickup action
        elif action == 4:
            # If holding key already, no change
            if 2 in keys:
                pass
            # Pickup a gem
            elif pos == self.gem:
                next_gem = 1
            # Pickup a key
            else:
                try:
                    key_idx = self.keys.index(pos)
                    if keys[key_idx] == 1:
                        next_keys[key_idx] = 2
                except ValueError:
                    # No key to pick up
                    pass
        # Unlock action
        else:
            # If not holding a key, no change
            if 2 not in keys:
                pass
            # Otherwise, check that a locked lock exists in the surrounding
            # location and that no wall exists between the agent and the lock
            else:
                # Get the (possibly illegal) surrounding locations N/E/S/W
                surroundings = [(pos[0], pos[1] - 1),
                                (pos[0] + 1, pos[1]),
                                (pos[0], pos[1] + 1),
                                (pos[0] - 1, pos[1])]

                for direction, walls in enumerate(self.walls.values()):
                    if pos in walls:
                        continue
                    try:
                        lock_idx = self.locks.index(surroundings[direction])
                        if locks[lock_idx] == 1:
                            # Unlock lock and consume held key
                            next_locks[lock_idx] = 0
                            next_keys[keys.index(2)] = 0
                            break
                    except ValueError:
                        # No lock
                        continue

        # Create next state
        next_state = [next_x, next_y] + next_keys + next_locks + [next_gem]

        # Assign reward
        if next_gem > gem:
            self.last_reward = self.R_SUCCESS
        elif next_locks.count(1) < locks.count(1):
            self.last_reward = self.R_UNLOCK
        else:
            self.last_reward = self.R_DEFAULT

        # Calculate outcome or effect
        if self.use_outcomes:
            observation = [self.O_NO_CHANGE] * self.NUM_ATT

            # Movement and picking up the gem can only affect one attribute
            # Keys and locks can change at the same time, but only one of each
            if x != next_x:
                observation[self.S_X] = self.A_WEST if next_x < x else self.A_EAST
            elif y != next_y:
                observation[self.S_Y] = self.A_NORTH if next_y < y else self.A_SOUTH
            elif gem != next_gem:
                observation[self.S_GEM] = self.A_PICKUP
            else:
                for i, (key, next_key) in enumerate(zip(keys, next_keys)):
                    if key != next_key:
                        observation[self.S_KEY_1 + i] = self.A_PICKUP if next_key == 2 else self.A_UNLOCK
                        break
                for i, (lock, next_lock) in enumerate(zip(locks, next_locks)):
                    if lock != next_lock:
                        observation[self.S_LOCK_1 + i] = self.A_UNLOCK
                        break
        else:
            # Get all possible JointEffects that could have transformed the current state into the next state
            observation = eff_joint(self.curr_state, next_state)

        # Update current state
        self.curr_state = next_state

        return observation

    def compute_next_loc(self, x: int, y: int, locks: List[int], action: int) -> Tuple[int, int]:
        """Deterministically return the result of taking an action"""
        pos = (x, y)
        if action == self.A_NORTH:
            next_pos = (x, y - 1)
            is_wall = pos in self.walls['N']
        elif action == self.A_EAST:
            next_pos = (x + 1, y)
            is_wall = pos in self.walls['E']
        elif action == self.A_SOUTH:
            next_pos = (x, y + 1)
            is_wall = pos in self.walls['S']
        else:
            next_pos = (x - 1, y)
            is_wall = pos in self.walls['W']

        if is_wall or (next_pos in self.locks and locks[self.locks.index(next_pos)] == 1):
            return x, y
        else:
            return next_pos

    def get_reward(self, state: int, next_state: int, action: int) -> float:
        """Get reward of arbitrary state"""
        factored_s = self.get_factored_state(state)
        factored_ns = self.get_factored_state(next_state)

        if factored_ns[self.S_GEM] > factored_s[self.S_GEM]:
            # The gem was picked up
            return self.R_SUCCESS
        if sum(factored_ns[self.S_LOCK_1: self.S_LOCK_3 + 1]) < sum(factored_s[self.S_LOCK_1: self.S_LOCK_3 + 1]):
            # The number of locked locks decreased
            return self.R_UNLOCK
        else:
            return self.R_DEFAULT

    def apply_outcome(self, state: int, outcome: List[int]) -> Union[int, np.ndarray]:
        """Compute next state given an outcome"""
        if all(o == self.O_NO_CHANGE for o in outcome):
            return state

        factored_s = self.get_factored_state(state)

        # Movement and picking up the gem can only affect one attribute
        # Keys and locks can change at the same time, but only one of each
        if outcome[self.S_X] == self.A_EAST:
            factored_s[self.S_X] += 1
        elif outcome[self.S_X] == self.A_WEST:
            factored_s[self.S_X] -= 1
        elif outcome[self.S_Y] == self.A_NORTH:
            factored_s[self.S_Y] -= 1
        elif outcome[self.S_Y] == self.A_SOUTH:
            factored_s[self.S_Y] += 1
        elif outcome[self.S_GEM] == self.A_PICKUP:
            factored_s[self.S_GEM] = 1
        else:
            for key_idx in range(self.S_KEY_1, self.S_KEY_5 + 1):
                if outcome[key_idx] == self.A_PICKUP:
                    factored_s[key_idx] = 2
                    break
                elif outcome[key_idx] == self.A_UNLOCK:
                    factored_s[key_idx] = 0
                    break
            for lock_idx in range(self.S_LOCK_1, self.S_LOCK_3 + 1):
                if outcome[lock_idx] == self.A_UNLOCK:
                    factored_s[lock_idx] = 0
                    break

        try:
            return self.get_flat_state(factored_s)
        except ValueError:
            # Outcome returned illegal state
            return state

    def visualize(self):
        pass
