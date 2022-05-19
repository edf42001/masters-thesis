from typing import List, Union
import numpy as np
import logging

from effects.effect import JointEffect
from environment.hierarchy.hierarchy import Hierarchy


class Environment:
    """Base class for any environment the agent could be working in"""
    # TODO: Figure this stuff out
    # From https://github.com/rail-cwru/hoomdp

    stochastic = None
    use_outcomes = None
    dynamic_objects = None

    MAX_PARENTS = None
    NUM_ACTIONS = None
    NUM_ATT = None
    NUM_COND = None
    R_SUCCESS = None
    STATE_ARITIES = None
    O_NO_CHANGE = None
    OB_ARITIES = None
    OB_COUNT = None

    # Names, to help debugging
    ACTION_NAMES = None
    ATT_NAMES = None

    instance_index_map = {}
    state_index_instance_map = {}
    state_index_class_map = {}
    state_index_class_index_map = {}

    eval_states = []

    hierarchy = None

    curr_state: List[int] = None
    last_action: int = None
    last_outcome: int = None
    last_reward: float = None

    def end_of_episode(self, state: int = None) -> bool:
        raise NotImplementedError()

    def restart(self):
        raise NotImplementedError()

    def get_condition(self, state: Union[int, List[int]]) -> List[bool]:
        raise NotImplementedError()

    def step(self, action: int) -> Union[List[JointEffect], List[int]]:
        raise NotImplementedError()

    def get_reward(self, state: int, next_state: int, action: int) -> float:
        raise NotImplementedError()

    def apply_outcome(self, state: int, outcome: List[int]) -> int:
        raise NotImplementedError()

    def visualize(self):
        pass

    def get_eval_states(self):
        return self.eval_states

    def get_factored_state(self, flat_state: int) -> List[Union[int, float]]:
        return list(np.unravel_index(flat_state, shape=self.STATE_ARITIES))

    def get_flat_state(self, factored_state: List[Union[int, float]]) -> int:
        return np.ravel_multi_index(factored_state, self.STATE_ARITIES)

    def get_last_reward(self) -> float:
        return self.last_reward

    def get_state(self) -> int:
        return self.get_flat_state(self.curr_state)

    def apply_effect(self, state: int, effect: JointEffect) -> Union[int, np.ndarray]:
        if effect.is_empty():
            return state
        try:
            factored_s = self.get_factored_state(state)
            effect.apply_to(factored_s)
            return self.get_flat_state(factored_s)
        except ValueError:
            # Effect returned illegal state
            logging.error(f"Effect {effect} returned illegal state")
            return state

    def get_num_actions(self) -> int:
        return self.NUM_ACTIONS

    def get_num_attributes(self) -> int:
        return self.NUM_ATT

    def get_attribute_arities(self) -> List[int]:
        return self.STATE_ARITIES

    def get_condition_size(self) -> int:
        return self.NUM_COND

    def get_rmax(self) -> float:
        """The maximum reward available in the environment"""
        return self.R_SUCCESS

    def get_num_states(self) -> int:
        return int(np.prod(self.STATE_ARITIES))

    def get_max_parents(self) -> int:
        return self.MAX_PARENTS

    def get_hierarchy(self) -> Hierarchy:
        return self.hierarchy

    def get_action_name(self, action: int):
        """Maps int action to string name"""
        return self.ACTION_NAMES[action]

    def get_att_name(self, attribute: int):
        """Maps int attribute to string name"""
        return self.ATT_NAMES[attribute]

    def generate_object_maps(self) -> None:
        """
        Generate useful maps for dealing with classes and object attributes in state
        instance_index_map: map instance number (unique object ID) to start and end index for attributes
        state_index_instance_map: map state variable to instance number of object that it belongs to
        state_index_class_map: map state variable to class of object that it belongs to
        state_index_class_index_map:  map state variable to index in corresponding class definition
        """
        # Expand object arities to list of arities for each object instance in env
        instance_arities = [arity for arity, count in zip(self.OB_ARITIES, self.OB_COUNT) for _ in range(count)]
        instance_classes = [cl for cl, count in enumerate(self.OB_COUNT) for _ in range(count)]

        # Iteration variables
        instance_num = 0
        base_idx, next_base_idx = 0, instance_arities[0]
        self.instance_index_map[0] = base_idx

        for idx in range(self.NUM_ATT):
            # Shift iteration variables when state variable belongs to new object instance
            if idx == next_base_idx:
                instance_num += 1
                base_idx = next_base_idx
                next_base_idx += instance_arities[instance_num]
                self.instance_index_map[instance_num] = (base_idx, next_base_idx)
            # Associate state variable with current object instance, class, and class index
            self.state_index_instance_map[idx] = instance_num
            self.state_index_class_map[idx] = instance_classes[instance_num]
            self.state_index_class_index_map[idx] = (idx - base_idx)
