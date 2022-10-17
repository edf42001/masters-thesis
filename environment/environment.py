from typing import List, Union, Tuple, Dict
import numpy as np
import logging

from effects.effect import EffectType, Effect
from symbolic_stochastic_domains.predicate_tree import PredicateTree
from symbolic_stochastic_domains.symbolic_classes import Outcome, DeicticReference


class Environment:
    """Base class for any environment the agent could be working in"""
    # From https://github.com/rail-cwru/hoomdp

    stochastic = None

    NUM_ACTIONS = None
    NUM_ATT = None
    R_SUCCESS = None
    STATE_ARITIES = None
    OB_ARITIES = None
    OB_COUNT = None

    # Names, to help debugging
    ACTION_NAMES = None
    ATT_NAMES = None
    OB_NAMES = None

    instance_index_map = {}
    state_index_instance_map = {}
    state_index_class_map = {}
    state_index_class_index_map = {}
    instance_class_map = {}

    eval_states = []

    curr_state: List[int] = None
    last_action: int = None
    last_reward: float = None

    def end_of_episode(self, state: int = None) -> bool:
        raise NotImplementedError()

    def restart(self):
        raise NotImplementedError()

    def get_condition(self, state: Union[int, List[int]]) -> List[bool]:
        raise NotImplementedError()

    def step(self, action: int) -> Tuple[PredicateTree, Outcome, dict]:
        raise NotImplementedError()

    def get_reward(self, state: int, next_state: int, action: int) -> float:
        raise NotImplementedError()

    def apply_outcome(self, state: int, outcome: List[int]) -> int:
        raise NotImplementedError()

    def unreachable_state(self, from_state, to_state) -> bool:
        raise NotImplementedError()

    def visualize(self, delay=100):
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

    def apply_effect(self, state: int, effect: Outcome) -> Union[int, np.ndarray]:
        if effect.is_no_effect():
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

    def get_rmax(self) -> float:
        """The maximum reward available in the environment"""
        return self.R_SUCCESS

    def get_num_states(self) -> int:
        return int(np.prod(self.STATE_ARITIES))

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
        instance_class_map: map instance number (unique object ID) to class of object that it is
        """
        # Expand object arities to list of arities for each object instance in env
        instance_arities = [arity for arity, count in zip(self.OB_ARITIES, self.OB_COUNT) for _ in range(count)]
        instance_classes = [cl for cl, count in enumerate(self.OB_COUNT) for _ in range(count)]

        # Iteration variables
        instance_num = 0
        base_idx, next_base_idx = 0, instance_arities[0]
        self.instance_index_map[0] = (base_idx, next_base_idx)  # This used to be not a tuple, seemed like a bug

        for idx in range(self.NUM_ATT):
            # print(f"Instance num: {instance_num}, base_idx: {base_idx}, next_base_idx: {next_base_idx}, idx: {idx}")
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

        self.instance_class_map = {i: c for i, c in enumerate(instance_classes)}

    def get_literals(self, state: int) -> Tuple[PredicateTree, Dict]:
        pass

    def get_observation_and_tree(self, next_state: List[int], correct_types: List[EffectType]) -> Tuple[PredicateTree, Outcome, dict]:
        # Get the correct effect type for each attribute. This is pretty good, but it would be better if it
        # was on a per class basis, that is then mapped. So lets map the attribute to a class,
        # and then we can reverse it using the groundings

        # In order to specify which object's attributes are changing, we need to know the variable groundings
        # for this state. Thus, we get the literals in here, and return them from the step function
        # As part of the observation
        tree, ob_id_name_map = self.get_literals(self.get_flat_state(self.curr_state))

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
                att_name = self.ATT_NAMES[class_id][class_att_idx]
                if len(node.to_edges) > 0:
                    to_edge = node.to_edges[0]
                    from_name = to_edge.from_node.object_name
                    to_name = to_edge.to_node.object_name
                    identifier = DeicticReference(from_name, to_edge.type, to_name, att_name)
                else:
                    from_name = unique_name[:-1]
                    identifier = DeicticReference(from_name, None, "", att_name)

                atts.append(identifier)

        if len(effects) == 0:
            observation = Outcome([], [], no_effect=True)
        else:
            observation = Outcome(atts, effects)

        return tree, observation, unique_name_to_ob_id
