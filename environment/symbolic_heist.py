import numpy as np
import cv2
from typing import List, Tuple, Union, Dict

from effects.utils import eff_joint
from effects.effect import JointEffect, EffectType, Effect, JointNoEffect
from environment.environment import Environment
from symbolic_stochastic_domains.predicates_and_objects import Taxi2D, Key2D, Lock2D, Wall2D,\
    Gem2D, Predicate, PredicateType


class SymbolicHeist(Environment):
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
    # Locks are locked (1) or unlocked (0)
    # Gem is not held or held
    STATE_ARITIES = [SIZE_X, SIZE_Y] + [3] * 5 + [2] * 4

    # Stochastic modification to actions
    MOD = [-1, 0, 1]
    P_PROB = [0.1, 0.8, 0.1]

    # Rewards
    R_DEFAULT = -1
    R_UNLOCK = 5
    R_SUCCESS = 10

    # Object descriptions
    OB_TAXI = 0
    OB_KEY = 1
    OB_LOCK = 2
    OB_GEM = 3
    OB_COUNT = [1, 5, 3, 1]
    OB_ARITIES = [2, 1, 1, 1]

    OB_NAMES = ["taxi", "key", "lock", "gem"]
    ACTION_NAMES = ['Up', 'Right', 'Down', 'Left', 'Pickup', 'Unlock']
    ATT_NAMES = [["x", "y"], ["state"], ["state"], ["state"]]

    # For each predicate type, defines which objects are valid for each argument
    # This is basically just (taxi, everything else) except for the ones that are just params?
    PREDICATE_MAPPINGS = {
        PredicateType.TOUCH_LEFT2D: [[OB_TAXI], [OB_KEY, OB_LOCK, OB_GEM]],
        PredicateType.TOUCH_RIGHT2D: [[OB_TAXI], [OB_KEY, OB_LOCK, OB_GEM]],
        PredicateType.TOUCH_UP2D: [[OB_TAXI], [OB_KEY, OB_LOCK, OB_GEM]],
        PredicateType.TOUCH_DOWN2D: [[OB_TAXI], [OB_KEY, OB_LOCK, OB_GEM]],
        PredicateType.ON2D: [[OB_TAXI], [OB_KEY, OB_LOCK, OB_GEM]],
        PredicateType.IN: [[OB_TAXI], [OB_KEY, OB_GEM]],
        # PredicateType.OPEN: [[OB_LOCK]]
    }

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
        # 4  | |_ _    |
        # 3  |  _    | |
        # 2  | | | | |_|
        # 1  | |_ _| | |
        # 0  |_ _|_ _ _|
        self.walls = {
            'N': {(1, 2), (0, 4), (3, 4), (1, 0), (2, 3), (2, 4), (4, 1), (2, 0), (1, 4), (1, 3), (4, 4)},
            'E': {(1, 2), (3, 2), (0, 4), (3, 1), (3, 3), (4, 0), (1, 0), (2, 1), (4, 1), (2, 2), (4, 2),
                  (0, 1), (4, 3), (0, 2), (4, 4)},
            'S': {(1, 1), (4, 0), (1, 0), (2, 4), (2, 0), (2, 1), (0, 0), (4, 2), (1, 4), (3, 0), (1, 3)},
            'W': {(0, 3), (1, 2), (3, 2), (0, 4), (1, 1), (3, 1), (1, 4), (2, 2), (2, 0), (4, 1), (0, 0),
                  (4, 2), (0, 1), (4, 3), (0, 2)}
        }

        # List of possible key locations
        self.keys = [(0, 4), (1, 4), (1, 2), (2, 0), (4, 2)]

        # List of lock locations
        self.locks = [(0, 2), (0, 1), (0, 0)]

        # Location of gem
        self.gem = (1, 0)

        # Object instance and class in state information
        self.generate_object_maps()

        # Set RL variables
        self.curr_state: List[int] = None
        self.last_action: int = None
        self.last_reward: float = None
        self.restart()

    def end_of_episode(self, state: int = None) -> bool:
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

    def get_object_list(self, state: int):
        state = self.get_factored_state(state)

        taxi = (state[self.S_X], state[self.S_Y])
        keys = state[self.S_KEY_1: self.S_KEY_5 + 1]
        locks = state[self.S_LOCK_1: self.S_LOCK_3 + 1]
        gem_state = state[self.S_GEM]  # TODO: need gem held

        objects = []

        objects.append(Taxi2D("taxi", taxi))

        # Instead of having str(i), we could just have key and then a mapping saying which key is the one that is true
        for i, (key_state, location) in enumerate(zip(keys, self.keys)):
            objects.append(Key2D("key", location, key_state))

        for i, (lock_open, location) in enumerate(zip(locks, self.locks)):
            objects.append(Lock2D("lock", location, lock_open == 0))  # A locked lock has a value of 1

        objects.append(Gem2D("gem", self.gem, gem_state))
        objects.append(Wall2D("wall", self.walls))

        return objects

    def get_literals(self, state: int) -> Tuple[List[Predicate], Dict[Predicate, int]]:
        """Converts state to the literals from that state"""

        # Get object list from the current state
        objects = self.get_object_list(state)

        bindings = dict()  # Dictionary that for each predicate that references an object, which object?
        properties = dict()  # Dictionary for object properties. Basically, if we have a lock that is open, which lock?

        predicates = []

        # Find the start and end indices for each object type
        ob_index_range_map = np.cumsum(self.OB_COUNT)

        for p_type, mappings in self.PREDICATE_MAPPINGS.items():
            # if len(mappings) == 1:
            #     # One arity predicate
            #     objects1 = mappings[0]
            #     for ob_id in objects1:
            #         # This is duplicate code. Anything we can do about that?
            #         found = False
            #         for ob_idx in range(ob_index_range_map[ob_id-1], ob_index_range_map[ob_id]):
            #             pred = Predicate.create(p_type, objects[ob_idx], objects[ob_idx])
            #             if pred.value:
            #                 predicates.append(pred)
            #                 bindings[pred] = ob_idx
            #                 found = True
            #                 break
            #         if not found:
            #             # Any will do
            #             predicates.append(Predicate.create(p_type, objects[ob_idx], objects[ob_idx]))  # note duplicate
            # else:
            # Create predicates combining every object in the first list with every object from the second
            objects1 = mappings[0]
            objects2 = mappings[1]
            for ob1_id in objects1:  # Could do the same here only have single taxi for now though
                for ob2_id in objects2:
                    # Find if any object makes the condition true. Return true then, otherwise, false
                    # TODO: But wait, what about the fact that multiple locks can be open at once? Does
                    # it not matter because they can't be in the same place at once? I think that might be it
                    found = False
                    for ob2_idx in range(ob_index_range_map[ob2_id-1], ob_index_range_map[ob2_id]):
                        pred = Predicate.create(p_type, objects[ob1_id], objects[ob2_idx])
                        if pred.value:
                            predicates.append(pred)
                            bindings[pred] = ob2_idx
                            found = True

                            # If we found a valid predicate and it has an object with a property in it, record it
                            # mapping from predicate referencing the object to the other predicate
                            # Would it be easier for predicate objects to have ids for each object as well as names?
                            if type(objects[ob2_idx]) is Lock2D:
                                properties[pred] = Predicate.create(PredicateType.OPEN, objects[ob2_idx], objects[ob2_idx])
                            break
                    if not found:
                        # Any will do. Will always be false.
                        predicates.append(Predicate.create(p_type, objects[ob1_id], objects[ob2_idx]))

        # TODO: Do walls need to be handled separately. Also technically this needs all the types, including on and in
        for p_type in [PredicateType.TOUCH_LEFT2D, PredicateType.TOUCH_RIGHT2D,
                       PredicateType.TOUCH_UP2D, PredicateType.TOUCH_DOWN2D]:
            predicates.append(Predicate.create(p_type, objects[self.OB_TAXI], objects[-1]))  # Objects -1 is the wall

        return predicates, bindings, properties

    def get_condition(self, state: int):
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

    def step(self, action: int) -> List[JointEffect]:
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
            # Get the correct effect type for each attribute. This is pretty good, but it would be better if it
            # was on a per class basis, that is then mapped. So lets map the attribute to a class,
            # and then we can reverse it using the groundings

            correct_types = [EffectType.INCREMENT] * 2 + [EffectType.SET_TO_NUMBER] * 8
            effects = []
            atts = []
            for att, e_type in enumerate(correct_types):
                if self.curr_state[att] != next_state[att]:
                    effects.append(Effect.create(e_type, self.curr_state[att], next_state[att]))

                    class_id = self.state_index_class_map[att]  #  Whoops, this doesn't take into account the taxi has two variables
                    class_att_idx = self.state_index_class_index_map[att]
                    # Convert the class and att idx to a string. (For viewing only, this probably makes the code slower)
                    identifier = f"{self.OB_NAMES[class_id]}.{self.ATT_NAMES[class_id][class_att_idx]}"

                    atts.append(identifier)

            if len(effects) == 0:
                observation = JointNoEffect()
            else:
                observation = JointEffect(atts, effects)

        # Update current state
        self.curr_state = next_state

        return observation

    def compute_next_loc(self, x: int, y: int, locks: List[int], action: int) -> Tuple[int, int]:
        """Deterministically return the result of taking an action"""
        pos = (x, y)
        if action == self.A_NORTH:
            next_pos = (x, y + 1)
            is_wall = pos in self.walls['N']
        elif action == self.A_EAST:
            next_pos = (x + 1, y)
            is_wall = pos in self.walls['E']
        elif action == self.A_SOUTH:
            next_pos = (x, y - 1)
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
        self.draw_world(self.get_flat_state(self.curr_state), delay=1)

    def draw_world(self, state: int, delay=100):
        state = self.get_factored_state(state)

        GRID_SIZE = 100
        WIDTH = self.SIZE_X
        HEIGHT = self.SIZE_Y

        # x, y, passenger, dest = state
        taxi_x = state[self.S_X]
        taxi_y = state[self.S_Y]

        # Blank white square
        img = 255 * np.ones((HEIGHT * GRID_SIZE, WIDTH * GRID_SIZE, 3))

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

        # Draw locks (need to be before taxi so taxi is shown on top)
        lock_values = state[self.S_LOCK_1:self.S_LOCK_3+1]
        for lock, value in zip(self.locks, lock_values):
            inset = 0.05
            bottom_left_x = lock[0] * GRID_SIZE + int(inset * GRID_SIZE)
            bottom_left_y = (HEIGHT - lock[1]) * GRID_SIZE - int(inset * GRID_SIZE)
            top_right_x = bottom_left_x + int((1 - 2 * inset) * GRID_SIZE)
            top_right_y = bottom_left_y - int((1 - 2 * inset) * GRID_SIZE)

            if value == 1:  # Closed lock
                color = [0, 0, 0.8]
            else:  # Open lock
                color = [0.8, 0.8, 1]

            cv2.rectangle(img, (bottom_left_x, bottom_left_y), (top_right_x, top_right_y), thickness=-1, color=color)

        # Draw taxi
        cv2.circle(img, (int((taxi_x + 0.5) * GRID_SIZE), int((HEIGHT - (taxi_y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.3),
                   thickness=-1, color=[0, 0, 0])

        # Draw gem
        gem_x, gem_y = self.gem
        cv2.circle(img, (int((gem_x + 0.5) * GRID_SIZE), int((HEIGHT - (gem_y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.25),
                   thickness=-1, color=[0.8, 0, 0])

        # Draw keys
        key_values = state[self.S_KEY_1:self.S_KEY_5+1]
        for key, value in zip(self.keys, key_values):
            if value == 0:  # Non existent key
                continue

            key_x, key_y = key

            if value == 1:  # Key on the ground
                cv2.circle(img, (int((key_x + 0.5) * GRID_SIZE), int((HEIGHT - (key_y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.25),
                           thickness=-1, color=[0, 0.7, 0.7])
            else:  # Key in the taxi
                cv2.circle(img, (int((taxi_x + 0.5) * GRID_SIZE), int((HEIGHT - (taxi_y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.2),
                           thickness=-1, color=[0, 0.7, 0.7])

        cv2.imshow("Heist World", img)
        cv2.waitKey(delay)
