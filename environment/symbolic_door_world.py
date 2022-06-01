import logging
import numpy as np
import random
from typing import List, Tuple, Union
from enum import Enum


# from effects.utils import eff_joint TODO
from effects.effect import JointEffect, Effect, EffectType, JointNoEffect
from environment.environment import Environment


class SymbolicObject:
    """Represents a generic object in a symbolic world"""
    def __init__(self, name=""):
        self.name = name

    def __repr__(self):
        # Print the object type plus our unique id
        return f"{type(self).__name__}({self.name})"


class Taxi(SymbolicObject):
    def __init__(self, name="", x=0):
        super().__init__(name)

        self.x = x


class Door(SymbolicObject):
    def __init__(self, name="", x=0, open=False):
        super().__init__(name)

        self.x = x
        self.open = open


class Switch(SymbolicObject):
    def __init__(self, name="", x=0):
        super().__init__(name)

        self.x = x


class Goal(SymbolicObject):
    def __init__(self, name="", x=0):
        super().__init__(name)

        self.x = x


class Wall(SymbolicObject):
    def __init__(self, name="", x=0):
        super().__init__(name)

        self.x = x


class PredicateType(Enum):
    TOUCH_LEFT = 0
    TOUCH_RIGHT = 1
    ON = 2
    OPEN = 3


class Predicate:
    """
    Grounded predicates have a name, object references, and truth value

    """
    type = None
    object1 = ""
    object2 = ""
    value = False

    hash = 0

    def __repr__(self):
        """For example, predicate on with args block1, table that is false will be ~on(block1, table)"""
        squiggle = "" if self.value else "~"
        name = type(self).__name__

        return f"{squiggle}{name}({self.object1}, {self.object2})"

    @staticmethod
    def create(p_type: PredicateType, o1: SymbolicObject, o2: SymbolicObject):
        """Factory method for creating predicates of specific type"""
        if p_type == PredicateType.TOUCH_LEFT:
            return TouchLeft(o1, o2)
        elif p_type == PredicateType.TOUCH_RIGHT:
            return TouchRight(o1, o2)
        elif p_type == PredicateType.ON:
            return On(o1, o2)
        elif p_type == PredicateType.OPEN:
            return Open(o1, o2)
        else:
            raise ValueError(f'Unrecognized effect type: {p_type}')

    def copy(self):
        # TODO: this is silly
        ret = type(self)(Door(), Door())
        ret.type = self.type
        ret.object1 = self.object1
        ret.object2 = self.object2
        ret.value = self.value
        ret.hash = self.hash
        return ret

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return (
            self.type == other.type and
            self.value == other.value and
            self.object1 == other.object1 and
            self.object2 == other.object2
        )


class TouchLeft(Predicate):
    # Perhaps each predicate can have a creator function which will create the object
    # thent he real constructor can take in the strings, but can call the evaluate function for the objects
    def __init__(self, object1: SymbolicObject, object2: SymbolicObject):
        self.type = PredicateType.TOUCH_LEFT
        self.value = self.evaluate(object1, object2)
        self.object1 = object1.name
        self.object2 = object2.name
        self.hash = hash((self.type, self.value, self.object1, self.object2))

    def evaluate(self, o1, o2):
        if type(o2) is Wall:
            # Because walls are between cells, they are treated separately
            return o1.x == o2.x
        else:
            return o1.x == o2.x+1


class TouchRight(Predicate):
    def __init__(self, object1: SymbolicObject, object2: SymbolicObject):
        self.type = PredicateType.TOUCH_RIGHT
        self.value = self.evaluate(object1, object2)
        self.object1 = object1.name
        self.object2 = object2.name
        self.hash = hash((self.type, self.value, self.object1, self.object2))

    @staticmethod
    def evaluate(o1, o2):
        # Because a wall position is defined left of the cell, this same equation works for them
        return o1.x == o2.x-1


class On(Predicate):
    def __init__(self, object1: SymbolicObject, object2: SymbolicObject):
        self.type = PredicateType.ON
        self.value = self.evaluate(object1, object2)
        self.object1 = object1.name
        self.object2 = object2.name
        self.hash = hash((self.type, self.value, self.object1, self.object2))

    @staticmethod
    def evaluate(o1, o2):
        # Taxi can't be on a wall
        if type(o2) is Wall:
            return False
        else:
            return o1.x == o2.x


class Open(Predicate):
    def __init__(self, object1: SymbolicObject, object2: SymbolicObject):
        self.type = PredicateType.OPEN
        self.value = self.evaluate(object1, object2)
        self.object1 = object1.name
        self.object2 = object2.name
        self.hash = hash((self.type, self.value, self.object1, self.object2))

    @staticmethod
    def evaluate(o1, o2):
        return o1.open


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
    OB_DOOR = 1
    OB_DEST = 2
    OB_COUNT = [1, 1, 1]
    # OB_ARITIES = [2, 1, 1]

    actions = ['Left', 'Right']

    def __init__(self, stochastic=False):
        self.stochastic: bool = stochastic

        # Taxi object
        self.taxi = Taxi(x=3, name="taxi")

        # Add walls to the map
        # Stores x location of wall, 0 indexed, to the left of the cell
        self.walls = [Wall(x=0, name="wall1"), Wall(x=8, name="wall2")]

        # Location of switch
        self.switch = Switch(x=1, name="switch")

        # Location of door
        self.door = Door(x=5, open=False, name="door")

        # Location of goal
        self.goal = Goal(x=7, name="goal")

        # Chance for action to do nothing
        self.no_move_probability = 0.3

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

        # Taxi has made it to the goal
        # TODO: Have to change this to .x because goal is now an object
        return state[0] == self.goal.x

    def restart(self, init_state=None):
        """Reset state variables to begin new episode"""
        if init_state:
            self.curr_state = init_state
        else:
            # Taxi starts at 3, door is closed
            self.curr_state = [3, 0]

            # TODO Have to change these as well
            self.taxi.x = 3
            self.door.open = False

    def get_literals(self, state: int) -> List[Predicate]:
        """Converts state to the literals from that state"""

        # Convert flat state to factored state
        state = self.get_factored_state(state)

        # Convert state to objects
        x, open = state
        taxi = Taxi(name="taxi", x=x)
        door = Door(name="door", x=self.door.x, open=open)

        # Switch, goal, and walls never changed, so they do not need to be converted. Neither does door.x
        non_taxi_objects = [self.switch, self.goal, door] + self.walls

        predicates = []

        for p_type in [PredicateType.TOUCH_LEFT, PredicateType.TOUCH_RIGHT, PredicateType.ON]:
            for object in non_taxi_objects:
                predicates.append(Predicate.create(p_type, taxi, object))

        # TODO: huh?
        predicates.append(Predicate.create(PredicateType.OPEN, door, door))
        return predicates

    def get_condition(self, state) -> List[bool]:
        """Convert state vars to OO conditions"""
        # Convert flat state to factored state
        if isinstance(state, (int, np.integer)):
            state = self.get_factored_state(state)

        # How to deal with parameterized actions?
        # (I.E., the fact we have two taxis?)

        conditions = [False] * self.NUM_COND
        x1 = state[self.S_X1]
        door_open = state[self.S_DOOR_OPEN]

        # touch(L/R wall), touch(L/R door), touch(L/R goal), touch(L/R switch), open(door)
        # conditions[0] = pos in self.walls['N']
        conditions[0] = x1 in self.walls
        conditions[1] = (x1 + 1) in self.walls
        conditions[2] = (x1 - 1) == self.door
        conditions[3] = (x1 + 1) == self.door
        conditions[4] = (x1 - 1) == self.goal
        conditions[5] = (x1 + 1) == self.goal
        conditions[6] = (x1 - 1) == self.switch
        conditions[7] = (x1 + 1) == self.switch
        conditions[8] = bool(door_open)

        return conditions

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
            self.taxi.x = next_x1  # Adding in stuff to update the object values
        # Right action
        elif action == 1:
            next_x1 = self.compute_next_loc(x1, action)
            self.taxi.x = next_x1  # Adding in stuff to update the object values
        else:
            logging.error("Unknown action {}".format(action))

        # Check if any taxi has pressed the switch
        # TODO: next_x1 is a number, but switch is an object, so I have to add .x
        if next_x1 == self.switch.x:
            next_door_open = True
            self.door.open = True  # Adding in stuff to update the object values

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
        walls = [wall.x for wall in self.walls]

        if action == self.A_LEFT:
            # If bump wall or door, don't move
            # TODO: door.x is because this is an object not a variable anymore
            if x in walls or ((x - 1) == self.door.x and not self.curr_state[self.S_DOOR_OPEN]):
                return x
            else:
                return x - 1
        elif action == self.A_RIGHT:
            # If bump wall or door, don't move
            if (x + 1) in walls or ((x + 1) == self.door.x and not self.curr_state[self.S_DOOR_OPEN]):
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
        # TODO: need .x in here because goal is now a object
        if factored_ns[self.S_X1] == self.goal.x:
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

        walls = [wall.x for wall in self.walls]
        door = self.door.x
        goal = self.goal.x
        switch = self.switch.x

        # Incrementally draw - for empty, d for closed door, o open, x for switch and g for goal
        drawing = ""
        for i in range(self.SIZE_X):
            if i in walls:
                drawing += "|"

            if i == x1:
                drawing += "t"
            elif i == door:
                if door_open:
                    drawing += "o"
                else:
                    drawing += "d"
            elif i == switch:
                drawing += "x"
            elif i == goal:
                drawing += "g"
            else:
                drawing += "-"

        drawing += "|"  # Wall at end
        return drawing
