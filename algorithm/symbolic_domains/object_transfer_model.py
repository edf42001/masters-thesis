from typing import List

from common.structures import Transition

from symbolic_stochastic_domains.symbolic_classes import Example, Outcome
from symbolic_stochastic_domains.object_transfer import get_possible_object_assignments,\
    determine_possible_object_maps, determine_transition_given_action


class ObjectTransferModel:
    """Tracks interactions with the world with Examples and Experience"""

    def __init__(self, env, previous_ruleset):
        self.env = env

        self.num_actions = self.env.get_num_actions()
        self.num_atts = self.env.NUM_ATT

        # Learned rules from taxi world but with different object names
        self.previous_ruleset = previous_ruleset

        # List of known object names without taxi and with wall (wall is static so is not in the list normally)
        self.prior_object_names = env.OB_NAMES.copy()
        self.prior_object_names.append("wall")
        self.prior_object_names.remove("taxi")

        self.possible_assignments = set()  # Possible objects that could be other objects or not

        self.solved = False  # Whether the object_map is now one to one

        self.object_map = {}
        self.init_object_map()  # Setup object map

    def init_object_map(self):
        current_object_names = self.env.get_object_names()
        self.object_map = {unknown: self.prior_object_names.copy() for unknown in current_object_names if unknown != "taxi"}

        # If an object has the same name as in the previous env, then it is the same. Update the object map accordingly
        for key, value in self.object_map.items():
            if key in self.prior_object_names:
                for value2 in self.object_map.values():
                    if key in value2:
                        value2.remove(key)
                self.object_map[key] = [key]

        # If I pass in for known_objects every object, then the environment is solved from the start
        self.solved = sum(len(possibilities) for possibilities in self.object_map.values()) == len(self.prior_object_names)

    def add_experience(self, action: int, state: int, outcome: Outcome):
        """Records experience of state action transition"""

        # Convert the observation to an outcome, combine with the set of literals to get an example to add to memory
        literals, instance_name_map = self.env.get_literals(state)
        example = Example(action, literals, outcome)

        print("Experienced Example:")
        print(example)

        assignments = get_possible_object_assignments(example, self.previous_ruleset)
        self.possible_assignments.update(assignments)
        print("All assignments: ")
        print(self.possible_assignments)
        print()

        # Pare down the object map
        length_of_beliefs = sum(len(possibilities) for possibilities in self.object_map.values())
        new_object_map = determine_possible_object_maps(self.object_map, self.possible_assignments)
        self.object_map = new_object_map
        new_lengths_of_beliefs = sum(len(possibilities) for possibilities in self.object_map.values())

        if new_lengths_of_beliefs != length_of_beliefs:
            print("Learned something new")

        if not self.solved and new_lengths_of_beliefs == len(self.prior_object_names):
            self.solved = True

        print("New object map:")
        for key, value in self.object_map.items():
            print(f"{key}: {value}")

    def compute_possible_transitions(self, state: int, action: int, literals=None, instance_name_map=None) -> List[Transition]:
        """
        Returns the effects (transitions) of taking the action given the condition
        If unknown, return None
        """

        # This is computational expensive, so if we can pass them as params to the function, let's do so
        if literals is None or instance_name_map is None:
            literals, instance_name_map = self.env.get_literals(state)

        transitions = determine_transition_given_action(self.env, state, action, self.object_map, self.previous_ruleset, literals=literals)

        if transitions is None:
            return []

        all_effects = []

        # Maps unique object name in the tree, to the real instance index in the environment
        name_instance_map = {v: k for k, v in instance_name_map.items()}

        # Loop over all possible transitions (i.e., if we don't know what the object is, it could
        # transition to a state where a gem or key is picked up), and generate an effect for each
        for transition in transitions:
            atts = []
            outcomes = []

            effect = transition

            for reference, outcome in effect.value.items():
                edge_type = reference.edge_type
                to_ob = reference.to_ob
                att_name = reference.att_name

                unique_name = ""
                # We handle the taxi case separately, as it isn't attached to anything
                if edge_type is None:
                    unique_name = "taxi0"
                    known_class_name = "taxi"  # This is only used so the env can say which attribute is which index
                else:
                    # Use the fact that only one object can be at the end of each relation to figure out what
                    # the desired object is in this state
                    for edge in literals.base_object.edges:
                        if edge.type == edge_type:
                            unique_name = edge.to_node.full_name()
                            break

                    # This is only used so the env can say which attribute is which index
                    # `taxi-IN-key`, will extract `key`
                    # Probably this should be a helper function in the env
                    known_class_name = to_ob

                    # unique_name = class_name + str(test_id)
                assert unique_name != "", "We should have found a match"

                # Do some weird indices hacks to figure out which attribute index is being referred to
                # so the env can directly modify the imagined state.
                class_idx = self.env.OB_NAMES.index(known_class_name)

                att_idx = self.env.ATT_NAMES[class_idx].index(att_name)
                instance_id = name_instance_map[unique_name]

                att_range = self.env.instance_index_map[instance_id]
                att = att_range[0] + att_idx

                atts.append(att)
                outcomes.append(outcome)

            # Only create new effect if it isn't a JointNoEffect
            if len(atts) > 0:
                effect = Outcome(atts, outcomes)

            all_effects.append(effect)

        # If there is a chance of more than one transition (i.e., if we don't know what the object is, it could
        # transition to a state where a gem or key is picked up) (in the agent's mind, only one actually happens)
        probability = 1.0 / len(all_effects)
        all_transitions = [Transition(effect, probability) for effect in all_effects]

        return all_transitions

    def get_reward(self, state: int, next_state: int, action: int):
        """Assumes all rewards are known in advance"""
        return self.env.get_reward(state, next_state, action)

    def next_state(self, state: int, observation) -> int:
        return self.env.apply_effect(state, observation)

    def unreachable_state(self, from_state: int, to_state: int) -> bool:
        return self.env.unreachable_state(from_state, to_state)

    def end_of_episode(self, state: int) -> bool:
        return self.env.end_of_episode(state)

    def print_model(self):
        """Returns predictions in an easy to read format"""
        print(self.ruleset)

    def save(self, filepath):
        raise NotImplementedError("Save not implemented for object transfer model")
