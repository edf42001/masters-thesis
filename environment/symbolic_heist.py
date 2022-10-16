import numpy as np
import cv2
from typing import List, Tuple, Dict

from effects.effect import JointEffect, EffectType, Effect, JointNoEffect
from environment.environment import Environment
from symbolic_stochastic_domains.predicates_and_objects import Taxi, Key, Lock, Wall,\
    Gem, Predicate, PredicateType
from symbolic_stochastic_domains.predicate_tree import PredicateTree
from common.utils.utils import random_string_generator


# TODO: Lots of duplicate code here
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
    # Keys are held by agent (0) existing (1), or not existing (2)
    # Locks are locked (1) or unlocked (0)
    # Gem is held (0) or not held (1)
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
        PredicateType.TOUCH_LEFT: [[OB_TAXI], [OB_KEY, OB_LOCK, OB_GEM]],
        PredicateType.TOUCH_RIGHT: [[OB_TAXI], [OB_KEY, OB_LOCK, OB_GEM]],
        PredicateType.TOUCH_UP: [[OB_TAXI], [OB_KEY, OB_LOCK, OB_GEM]],
        PredicateType.TOUCH_DOWN: [[OB_TAXI], [OB_KEY, OB_LOCK, OB_GEM]],
        PredicateType.ON: [[OB_TAXI], [OB_KEY, OB_LOCK, OB_GEM]],
        PredicateType.IN: [[OB_TAXI], [OB_KEY, OB_GEM]],
        # PredicateType.OPEN: [[OB_LOCK]]
    }

    def __init__(self, stochastic=True, shuffle_object_names=False):
        self.stochastic = stochastic

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

        # Initialize object name map to anonymize object identities
        self.object_name_map = None
        if shuffle_object_names:
            self.object_name_map = {}
            for ob in self.OB_NAMES + ['wall']:
                self.object_name_map[ob] = random_string_generator(5)
            self.object_name_map['taxi'] = 'taxi'  # Except for taxi, taxi is base object

        self.restart()

    def end_of_episode(self, state: int = None) -> bool:
        """Check if the episode has ended"""
        state = self.get_factored_state(state) if state else self.curr_state
        return state[self.S_GEM] == 0

    def restart(self, init_state=None):
        """Reset state variables to begin new episode"""
        if init_state:
            self.curr_state = init_state
        else:
            # Agent starts at (2, 1)
            # Randomly choose 3 of 5 possible key locations (1 if exists, 2 if not exists)
            # All locks (3) begin locked (1)
            # Gem begins not held (1)
            key_idx = np.random.choice(5, size=3, replace=False)
            self.curr_state = [2, 1] + [1 if i in key_idx else 2 for i in range(5)] + [1, 1, 1, 1]

    def get_object_list(self, state: int):
        state = self.get_factored_state(state)

        taxi = (state[self.S_X], state[self.S_Y])
        keys = state[self.S_KEY_1: self.S_KEY_5 + 1]
        locks = state[self.S_LOCK_1: self.S_LOCK_3 + 1]
        gem_state = state[self.S_GEM]  # TODO: need gem held

        objects = []

        objects.append(Taxi("taxi", taxi))

        # Instead of having str(i), we could just have key and then a mapping saying which key is the one that is true
        for i, (key_state, location) in enumerate(zip(keys, self.keys)):
            objects.append(Key("key", location, key_state))

        for i, (lock_open, location) in enumerate(zip(locks, self.locks)):
            objects.append(Lock("lock", location, lock_open == 0))  # A locked lock has a value of 1

        objects.append(Gem("gem", self.gem, gem_state))
        objects.append(Wall("wall", self.walls))

        return objects

    def anonymize_name(self, ob_name):
        if self.object_name_map:
            return self.object_name_map[ob_name]

        return ob_name

    def get_literals(self, state: int) -> Tuple[PredicateTree, Dict]:
        """Converts state to the literals from that state using variables to refer to objects"""

        # Get object list from the current state
        objects = self.get_object_list(state)

        # Create a tree. Add the taxi node, as that will always exist
        tree = PredicateTree()
        tree.add_node("taxi0")

        # To create unique ids for each object, but that don't depend on which object is which
        object_reference_counts = {name: 0 for name in self.OB_NAMES}  # Like {'taxi': 0, 'key': 1, 'lock': 0, 'gem': 0}
        # Object id to incremental id, i.e {3: 0, 4: 1, 0: 0} (3 and 4 are both keys, so one is key0 and one is key2)
        ob_index_name_map = dict()

        # Find the start and end indices for each object type, so we can iterate over all objects of a type
        ob_index_range_map = np.cumsum(self.OB_COUNT)

        for p_type, mappings in self.PREDICATE_MAPPINGS.items():
            objects1 = mappings[0]  # The mappings define which objects can be together
            objects2 = mappings[1]
            for ob1_id in objects1:  # Could do the same here only have single taxi for now though
                for ob2_id in objects2:
                    # Find if any object makes the condition true. Only add predicates for true values
                    # Check every object of that type
                    for ob2_idx in range(ob_index_range_map[ob2_id-1], ob_index_range_map[ob2_id]):
                        pred = Predicate.create(p_type, objects[ob1_id], objects[ob2_idx], walls=self.walls)
                        if pred.value:
                            # Convert the objects to unique variable names
                            # If we have already encountered this object, reuse the name.
                            # I think because I hardcode taxi0, it never needs to lookup taxi0 here
                            if ob1_id in ob_index_name_map:
                                new_name1 = ob_index_name_map[ob1_id]
                            else:
                                # Extract the class name, and append the id to the end of it
                                ob_name = objects[ob1_id].name
                                new_name1 = self.anonymize_name(ob_name) + str(object_reference_counts[ob_name])
                                object_reference_counts[ob_name] += 1  # So the next will have a new name
                                # Store this in the name map so if we encounter it again we can refer to it
                                ob_index_name_map[ob1_id] = new_name1

                            if ob2_idx in ob_index_name_map:
                                new_name2 = ob_index_name_map[ob2_idx]
                                assert False, "Just checking if this ever runs"
                            else:
                                # Extract the class name, and append the id to the end of it
                                ob_name = objects[ob2_idx].name
                                new_name2 = self.anonymize_name(ob_name) + str(object_reference_counts[ob_name])
                                object_reference_counts[ob_name] += 1  # So the next will have a new name
                                # Store this in the name map so if we encounter it again we can refer to it
                                ob_index_name_map[ob2_idx] = new_name2

                            # Recreate the object, but swap the names out for the variables
                            # node = Node(self.OB_NAMES[ob2_id], new_name2[-1])  # Extract just the id for the node
                            # tree.base_object.add_edge(Edge(p_type, node))
                            tree.add_node(new_name2)
                            tree.add_edge("taxi0", new_name2, p_type)

                            # Handle properties separately
                            if type(objects[ob2_idx]) is Lock:
                                pred = Predicate.create(PredicateType.OPEN, objects[ob2_idx], objects[ob2_idx])
                                if pred.value:
                                    # node.add_edge(Edge(PredicateType.OPEN, Node("lock", new_name2[-1])))
                                    tree.add_property(new_name2, PredicateType.OPEN, True)
                                    # tree.add_node(new_name2 + "_property")
                                    # tree.add_edge(new_name2, new_name2 + "_property", PredicateType.OPEN)

                            # Only one object can satisfy the condition at a time, so no need to keep searching
                            break

        # Walls are currently handled separately because they are static
        # TODO: Do walls need to be handled separately. Also technically this needs all the types, including on and in

        for p_type in [PredicateType.TOUCH_LEFT, PredicateType.TOUCH_RIGHT,
                       PredicateType.TOUCH_UP, PredicateType.TOUCH_DOWN]:
            pred = Predicate.create(p_type, objects[self.OB_TAXI], objects[-1], walls=self.walls)  # Objects -1 is the wall
            if pred.value:
                wall_name = self.anonymize_name("wall") + "0"
                if wall_name not in tree.node_lookup:
                    tree.add_node(wall_name)
                tree.add_edge("taxi0", wall_name, p_type)
                # tree.base_object.add_edge(Edge(p_type, node))

        # # Note: The taxi is referenced by the action (MoveLeft(taxi0)) for example. We need to add this in or
        # # it won't be able to refer to the taxi in instances when there are no other predicates
        if 0 not in ob_index_name_map:  # Manually put the taxi in there if it is not
            ob_index_name_map[0] = "taxi0"  # Taxi is object 0

        # TODO: Could also have the predicates just use "taxi" instead of "taxi0", but then have a dict for
        # each predicate to it's mappings: {Predicate: ["taxi0", "key1"]}, which might help with hashing
        return tree, ob_index_name_map

        # # If we found a valid predicate and it has an object with a property in it, record it
        # # mapping from predicate referencing the object to the other predicate
        # # Would it be easier for predicate objects to have ids for each object as well as names?

    def step(self, action: int) -> Tuple[PredicateTree, JointEffect]:
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
            if 0 in keys:
                pass
            # Pickup a gem
            elif pos == self.gem:
                next_gem = 0
            # Pickup a key
            else:
                try:
                    key_idx = self.keys.index(pos)
                    if keys[key_idx] == 1:
                        next_keys[key_idx] = 0
                except ValueError:
                    # No key to pick up
                    pass
        # Unlock action
        else:
            # If not holding a key, no change
            if 0 not in keys:
                pass
            # Otherwise, check that a locked lock exists in the surrounding
            # location and that no wall exists between the agent and the lock
            else:
                # Get the (possibly illegal) surrounding locations N/E/S/W
                surroundings = [(pos[0], pos[1] + 1),
                                (pos[0] + 1, pos[1]),
                                (pos[0], pos[1] - 1),
                                (pos[0] - 1, pos[1])]

                for direction, walls in enumerate(self.walls.values()):
                    if pos in walls:
                        continue
                    try:
                        lock_idx = self.locks.index(surroundings[direction])
                        if locks[lock_idx] == 1:
                            # Unlock lock and consume held key
                            next_locks[lock_idx] = 0
                            next_keys[keys.index(0)] = 2
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

        # Get the correct effect type for each attribute. This is pretty good, but it would be better if it
        # was on a per class basis, that is then mapped. So lets map the attribute to a class,
        # and then we can reverse it using the groundings

        # In order to specify which object's attributes are changing, we need to know the variable groundings
        # for this state. Thus, we get the literals in here, and return them from the step function
        # As part of the observation
        tree, ob_id_name_map = self.get_literals(self.get_flat_state(self.curr_state))  # predicate_to_ob_map

        correct_types = [EffectType.INCREMENT] * 2 + [EffectType.SET_TO_NUMBER] * 9
        effects = []
        atts = []
        obs_grounding = dict()  # class name to class unique identifier
        unique_name_to_ob_id = dict()
        for att, e_type in enumerate(correct_types):
            if self.curr_state[att] != next_state[att]:
                effects.append(Effect.create(e_type, self.curr_state[att], next_state[att]))

                #  Whoops, this doesn't take into account the taxi has two variables
                class_id = self.state_index_class_map[att]  # This one maps the att to the type of object
                class_instance_id = self.state_index_instance_map[att]  # This one maps the att to the specific object
                class_att_idx = self.state_index_class_index_map[att]  # If an object has many atts, this is which one

                # Convert the class and att idx to a string. (For viewing only, this probably makes the code slower)
                # identifier = f"{ob_id_name_map[class_instance_id]}.{self.ATT_NAMES[class_id][class_att_idx]}"
                # identifier = f"{self.OB_NAMES[class_id]}{ob_id_name_map[class_instance_id]}.{self.ATT_NAMES[class_id][class_att_idx]}"
                # obs_grounding[self.OB_NAMES[class_id]] = ob_id_name_map[class_instance_id]
                unique_name_to_ob_id[ob_id_name_map[class_instance_id]] = class_instance_id

                # We want to use deictic references to refer to objects. First we use the unique identifier to get the
                # corresponding node in the tree
                unique_name = ob_id_name_map[class_instance_id]
                node = tree.node_lookup[unique_name]
                # For now, assume there is only one path towards every object. i.e, no loops
                # (except for walls, but those are static so it doesn't matter, their properties will never change)
                # Make an exception for taxi, the taxi is already at the root of the tree. So there is nothing to chain
                # We remove the numbers ([:-1]) from here, because those are not the defining feature, the defining feature
                # is the relationship between the objects
                if len(node.to_edges) > 0:
                    to_edge = node.to_edges[0]
                    from_object_name = to_edge.from_node.object_name
                    identifier = f"{from_object_name[:-1]}-{str(to_edge)[:-1]}.{self.ATT_NAMES[class_id][class_att_idx]}"
                else:
                    identifier = f"{unique_name[:-1]}.{self.ATT_NAMES[class_id][class_att_idx]}"

                atts.append(identifier)

        if len(effects) == 0:
            observation = JointNoEffect()
        else:
            observation = JointEffect(atts, effects)

        # Update current state
        self.curr_state = next_state

        return tree, observation, unique_name_to_ob_id

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

        if factored_ns[self.S_GEM] < factored_s[self.S_GEM]:
            # The gem was picked up
            return self.R_SUCCESS
        if sum(factored_ns[self.S_LOCK_1: self.S_LOCK_3 + 1]) < sum(factored_s[self.S_LOCK_1: self.S_LOCK_3 + 1]):
            # The number of locked locks decreased
            return self.R_UNLOCK
        else:
            return self.R_DEFAULT

    def visualize(self, delay=100):
        self.draw_world(self.get_flat_state(self.curr_state), delay=delay)

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
            if value == 2:  # Non existent key
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

    def get_object_names(self):
        return [self.anonymize_name(ob_name) for ob_name in self.OB_NAMES + ["wall"]]
