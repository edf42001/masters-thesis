import logging
import numpy as np
import random
from typing import List, Tuple, Union
from enum import Enum


# from effects.utils import eff_joint TODO
from effects.effect import JointEffect, Effect, EffectType, JointNoEffect
from environment.environment import Environment
from symbolic_stochastic_domains.predicates_and_objects import Predicate, PredicateType, Taxi, Wall, Door, Switch, Goal


class SymbolicDoorWorld(Environment):
    """
    A one dimensional world with a switch and a door that needs to be opened to let a taxi through
    |-x-t-d-g|
    """

    # Environment constants
    SIZE_X = 8

    # Actions of agent
    A_LEFT = 0
    A_RIGHT = 1
    NUM_ACTIONS = 2

    # State variables
    S_X1 = 0  # Taxi1 x
    S_DOOR_OPEN = 1  # Door open or close
    NUM_ATT = 2

    # The taxi can be anywhere along the line
    # The door can be open or closed
    STATE_ARITIES = [SIZE_X, 2]

    # Rewards
    R_DEFAULT = -1
    R_SUCCESS = 10

    # Object descriptions
    OB_TAXI = 0
    OB_WALL = 1
    OB_SWITCH = 2
    OB_DOOR = 3
    OB_GOAL = 4
    OB_COUNT = [1, 1, 1, 1, 1]
    OB_ARITIES = [1, 0, 0, 1, 0]  # How many state variables does each object contribute?
    OB_NAMES = ["taxi", "wall", "switch", "door", "goal"]

    actions = ['Left', 'Right']

    # For each predicate type, defines which objects are valid for each argument
    # This is basically just (taxi, everything else) except for the ones that are just params?
    PREDICATE_MAPPINGS = {
        PredicateType.TOUCH_LEFT: [[OB_TAXI], [OB_WALL, OB_SWITCH, OB_GOAL, OB_DOOR]],
        PredicateType.TOUCH_RIGHT: [[OB_TAXI], [OB_WALL, OB_SWITCH, OB_GOAL, OB_DOOR]],
        PredicateType.ON: [[OB_TAXI], [OB_WALL, OB_SWITCH, OB_GOAL, OB_DOOR]],
        PredicateType.OPEN: [[OB_DOOR]]
    }

    # Things defines up here are for the domain as a whole. Things below are for a specific instance of that domain

    def __init__(self, stochastic=False):
        self.stochastic: bool = stochastic

        # Taxi object
        self.taxi = Taxi(x=3, name="taxi")

        # Add walls to the map
        # Stores x location of wall, 0 indexed, to the left of the cell
        self.walls = [0, 8]

        # Location of switch
        self.switch = 1

        # Location of door
        self.door = 5

        # Location of goal
        self.goal = 7

        # Chance for action to do nothing
        self.no_move_probability = 0.3

        # Object instance in state information
        self.generate_object_maps()
        print(self.instance_index_map)
        print(self.state_index_instance_map)
        print(self.state_index_class_map)
        print(self.state_index_class_index_map)
        # For testing, stop here
        # import sys
        # sys.exit(1)

        self.curr_state: List[int] = None
        self.last_action: int = None
        self.last_reward: float = None
        self.restart()

    def end_of_episode(self, state: int = None) -> bool:
        """Check if the episode has ended"""
        # Extract state variables from state integer if passed as option
        state = self.get_factored_state(state) if state else self.curr_state

        # Taxi has made it to the goal
        return state[0] == self.goal

    def restart(self, init_state=None):
        """Reset state variables to begin new episode"""
        if init_state:
            self.curr_state = init_state
        else:
            # Taxi starts at 3, door is closed
            self.curr_state = [3, 0]

    def get_object_list(self, state: int):
        state = self.get_factored_state(state)
        x, open = state

        # Convert state to a list of objects
        return [Taxi(x=x, name="taxi"), Wall(locations=self.walls, name="wall"), Switch(x=self.switch, name="switch"),
                Door(x=self.door, open=open, name="door"), Goal(x=self.goal, name="goal")]

    def get_literals(self, state: int) -> List[Predicate]:
        """Converts state to the literals from that state"""

        # Get object list from the current state
        objects = self.get_object_list(state)

        predicates = []

        for p_type, mappings in self.PREDICATE_MAPPINGS.items():
            # TODO: need to convert class variable to a range maybe
            if len(mappings) == 1:
                # One arity predicate
                objects1 = mappings[0]
                for ob_id in objects1:
                    predicates.append(Predicate.create(p_type, objects[ob_id], objects[ob_id]))
            else:
                # Create predicates combining every object in the first list with every object from the second
                objects1 = mappings[0]
                objects2 = mappings[1]
                for ob1_id in objects1:
                    for ob2_id in objects2:
                        predicates.append(Predicate.create(p_type, objects[ob1_id], objects[ob2_id]))

        # TODO: huh?
        return predicates

    def step(self, action: int) -> Union[List[JointEffect], List[int]]:
        """Stochastically or deterministically apply action to environment"""
        x1, door_open = self.curr_state
        next_x1, next_door_open = x1, door_open

        self.last_action = action

        # If stochastic, there is a chance for no movement to occur
        if self.stochastic and random.random() < self.no_move_probability:
            pass
        # Left action
        elif action == 0:
            next_x1 = self.compute_next_loc(x1, action)
        # Right action
        elif action == 1:
            next_x1 = self.compute_next_loc(x1, action)
        else:
            logging.error("Unknown action {}".format(action))

        # Check if any taxi has pressed the switch
        if next_x1 == self.switch:
            next_door_open = True

        # Make updates to state
        next_state = [next_x1, next_door_open]

        # Assign reward if any taxi made it to the goal state
        self.last_reward = self.get_reward(self.get_flat_state(self.curr_state), self.get_flat_state(next_state), action)

        # Calculate outcomes
        att_list = []
        effect_list = []
        if next_x1 != x1:
            att_list.append("taxi.x")
            effect_list.append(Effect.create(EffectType.INCREMENT, x1, next_x1))
        if next_door_open != door_open:
            att_list.append("door.open")
            effect_list.append(Effect.create(EffectType.SET_TO_NUMBER, door_open, next_door_open))

        if len(att_list) == 0:
            observation = JointNoEffect()
        else:
            observation = JointEffect(att_list=att_list, eff_list=effect_list)

        self.curr_state = next_state

        return observation

    def compute_next_loc(self, x: int, action: int) -> int:
        """Deterministically return the result of moving, given that there are walls and doors"""

        # Because am trying to use objects but state is numbers, need to convert to numbers
        # Replaced self.walls below with these walls.
        # Perhaps do the state in numbers, but have associated objects in a dictionary?
        # Or, a mapping from state to numbers, with the object maps

        if action == self.A_LEFT:
            # If bump wall or door, don't move
            # TODO: door.x is because this is an object not a variable anymore
            if x in self.walls or ((x - 1) == self.door and not self.curr_state[self.S_DOOR_OPEN]):
                return x
            else:
                return x - 1
        elif action == self.A_RIGHT:
            # If bump wall or door, don't move
            if (x + 1) in self.walls or ((x + 1) == self.door and not self.curr_state[self.S_DOOR_OPEN]):
                return x
            else:
                return x + 1
        else:
            logging.error("Unknown action {}".format(action))
            return x

    def get_reward(self, state: int, next_state: int, action: int) -> float:
        """Get reward of arbitrary state"""
        factored_ns = self.get_factored_state(next_state)

        # Assign reward if any taxi made it to the goal state
        if factored_ns[self.S_X1] == self.goal:
            return self.R_SUCCESS
        else:
            return self.R_DEFAULT

    def apply_effect(self, state: int, effect: JointEffect) -> Union[int, np.ndarray]:
        # TODO: because I am doing this hackily, need to convert the strings in my joint effect to state variables

        if type(effect) is JointNoEffect:
            return state

        factored_s = self.get_factored_state(state)
        for att, change in effect.value.items():
            if att == "taxi.x":
                factored_s[0] = change.apply_to(factored_s[0])
            else:
                factored_s[1] = 1 if change.apply_to(factored_s[1]) else 0

        try:
            state = self.get_flat_state(factored_s)
            return state
        # This can happen when the taxi predicts a movement out of bounds, for example
        except ValueError:
            # Effect returned illegal state
            logging.error(f"Effect {effect} returned illegal state")
            return state

    def visualize(self) -> str:
        return self.visualize_state(self.curr_state)

    def visualize_state(self, curr_state) -> str:
        x1, door_open = curr_state

        # Incrementally draw - for empty, d for closed door, o open, x for switch and g for goal
        drawing = ""
        for i in range(self.SIZE_X):
            if i in self.walls:
                drawing += "|"

            if i == x1:
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

        drawing += "|"  # Wall at end
        return drawing
