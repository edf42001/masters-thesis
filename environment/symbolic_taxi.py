import numpy as np
import random
from typing import List, Tuple, Union, Dict
import cv2

from effects.effect import JointEffect, EffectType, Effect, JointNoEffect
from environment.environment import Environment

from symbolic_stochastic_domains.predicates_and_objects import Taxi, Wall, Passenger, Destination,\
    PredicateType, Predicate
from symbolic_stochastic_domains.predicate_tree import PredicateTree
from common.utils.utils import random_string_generator


class SymbolicTaxi(Environment):

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
    ACTION_NAMES = ["Up", "Right", "Down", "Left", "Pickup", "Dropoff"]

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
    OB_NAMES = ["taxi", "pass", "dest"]

    ATT_NAMES = [["x", "y"], ["state"], ["state"]]

    # For each predicate type, defines which objects are valid for each argument
    # This is basically just (taxi, everything else) except for the ones that are just params?
    PREDICATE_MAPPINGS = {
        PredicateType.TOUCH_LEFT: [[OB_TAXI], [OB_PASS, OB_DEST]],
        PredicateType.TOUCH_RIGHT: [[OB_TAXI], [OB_PASS, OB_DEST]],
        PredicateType.TOUCH_UP: [[OB_TAXI], [OB_PASS, OB_DEST]],
        PredicateType.TOUCH_DOWN: [[OB_TAXI], [OB_PASS, OB_DEST]],
        PredicateType.ON: [[OB_TAXI], [OB_PASS, OB_DEST]],
        PredicateType.IN: [[OB_TAXI], [OB_PASS, OB_DEST]],
        # PredicateType.OPEN: [[OB_LOCK]]
    }

    # For visualization
    lines = ['|   |     |',
             '|   |     |',
             '|         |',
             '| |   |   |',
             '| |   |   |']

    def __init__(self, stochastic=True, shuffle_actions=False, shuffle_object_names=False):
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
        self.last_reward: float = None

        # Initialize object name map to anonymize object identities
        self.object_name_map = None
        if shuffle_object_names:
            self.object_name_map = {}
            for ob in self.OB_NAMES + ['wall']:
                self.object_name_map[ob] = random_string_generator(5)
            self.object_name_map['taxi'] = 'taxi'  # Except for taxi, taxi is base object

        # Restart to begin episode
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
            # Update so set to 0 indicates pickup
            # Passenger has 0 = pickup, but destination is normal (starts at 0)
            passenger, destination = random.sample([1, 2, 3, 4], 2)
            destination = destination - 1
            self.curr_state = [0, 1, passenger, destination]

    def anonymize_name(self, ob_name):
        if self.object_name_map:
            return self.object_name_map[ob_name]

        return ob_name

    def get_object_list(self, state: int):
        state = self.get_factored_state(state)

        taxi = (state[self.S_X], state[self.S_Y])
        passenger = state[self.S_PASS]
        destination = state[self.S_DEST]

        objects = []
        objects.append(Taxi("taxi", taxi))
        objects.append(Passenger("pass", self.locations[passenger-1] if 0 < passenger < len(self.locations)+1 else None, passenger))
        objects.append(Destination("dest", self.locations[destination], destination))
        objects.append(Wall("wall", self.walls))

        return objects

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
                            # If we have already encountered this object, reuse the name. Why is this not used?
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
            # Check if taxi is on correct pickup location and not already held
            if 0 < passenger < self.NUM_LOCATIONS+1 and pos == self.locations[passenger-1]:
                next_passenger = 0
        # Dropoff action
        else:
            pos = (x, y)
            # Check if passenger is in taxi and taxi is on the destination
            if passenger == 0 and pos == self.locations[destination]:
                next_passenger = self.NUM_LOCATIONS + 1

        # Make updates to state
        # Destination status does not change
        next_state = [next_x, next_y, next_passenger, destination]

        # Assign reward
        if next_passenger == self.NUM_LOCATIONS + 1:
            self.last_reward = self.R_SUCCESS
        else:
            self.last_reward = self.R_DEFAULT

        tree, ob_id_name_map = self.get_literals(self.get_flat_state(self.curr_state))  # predicate_to_ob_map

        correct_types = [EffectType.INCREMENT] * 2 + [EffectType.SET_TO_NUMBER] * 2
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
        elif 0 < from_factored[self.S_PASS] < len(self.locations)+1 and \
             0 < to_factored[self.S_PASS] < len(self.locations)+1 and \
             from_factored[self.S_PASS] != to_factored[self.S_PASS]:
            return True
        else:
            return False

    def visualize(self, delay=100):
        self.draw_world(self.get_flat_state(self.curr_state), delay=delay)

    def visualize_state(self, curr_state):
        x, y, passenger, dest = curr_state
        dest_x, dest_y = self.locations[dest]
        lines = self.lines
        taxi = '@' if passenger == 0 else 'O'

        pass_x, pass_y = -1, -1
        if 0 < passenger < self.NUM_LOCATIONS+1:
            pass_x, pass_y = self.locations[passenger - 1]

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

    def draw_world(self, state, delay=100):
        state = self.get_factored_state(state)

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
        if passenger == 0 or passenger == self.NUM_LOCATIONS+1:
            pass_x = x
            pass_y = y
        else:
            pass_x = self.locations[passenger-1][0]
            pass_y = self.locations[passenger-1][1]

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

    def get_object_names(self):
        return [self.anonymize_name(ob_name) for ob_name in self.OB_NAMES + ["wall"]]
