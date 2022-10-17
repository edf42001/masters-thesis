from typing import List

from algorithm.transition_model import TransitionModel
from common.structures import Transition

from symbolic_stochastic_domains.symbolic_classes import Example, Outcome
from symbolic_stochastic_domains.object_transfer import get_possible_object_assignments,\
    determine_possible_object_maps, determine_transition_given_action


class ObjectTransferModel(TransitionModel):
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

        current_object_names = self.env.get_object_names()
        self.object_map = {unknown: self.prior_object_names.copy() for unknown in current_object_names if unknown != "taxi"}
        self.solved = False  # Whether the object_map is now one to one

        self.possible_assignments = set()  # Possible objects that could be other objects or not

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

    def compute_possible_transitions(self, state: int, action: int) -> List[Transition]:
        """
        Returns the effects (transitions) of taking the action given the condition
        If unknown, return None
        """

        literals, instance_name_map = self.env.get_literals(state)

        transitions = determine_transition_given_action(self.env, state, action, self.object_map, self.previous_ruleset)

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

            for ob_att_str, outcome in effect.value.items():
                # obb_att_str is formatted either `taxi.y` or `taxi-IN-key.state`. In general, `ob1-pred1-ob2-pred2...-obn`
                identifier_str, att_name_str = ob_att_str.split(".")

                unique_name = ""
                # We handle the taxi case separately, as it isn't attached to anything
                if identifier_str == "taxi":
                    unique_name = "taxi0"
                    known_class_name = "taxi"  # This is only used so the env can say which attribute is which index
                else:
                    splits = identifier_str.split("-")

                    # Use the fact that only one object can be at the end of each relation to figure out what
                    # the desired object is in this state
                    base_object_and_relation = splits[0] + "-" + splits[1]
                    for edge in literals.base_object.edges:
                        edge_string = "taxi-" + edge.type.name  # Remove number
                        if edge_string == base_object_and_relation:
                            unique_name = edge.to_node.full_name()

                    # This is only used so the env can say which attribute is which index
                    # `taxi-IN-key`, will extract `key`
                    # Probably this should be a helper function in the env
                    known_class_name = splits[-1]

                    # unique_name = class_name + str(test_id)
                assert unique_name != "", "We should have found a match"

                # Do some weird indices hacks to figure out which attribute index is being referred to
                # so the env can directly modify the imagined state.
                class_idx = self.env.OB_NAMES.index(known_class_name)

                att_idx = self.env.ATT_NAMES[class_idx].index(att_name_str)
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
