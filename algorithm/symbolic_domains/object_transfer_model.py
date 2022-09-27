from typing import List

from effects.effect import JointEffect
from algorithm.transition_model import TransitionModel
from common.structures import Transition

from symbolic_stochastic_domains.symbolic_classes import Example, Outcome
from test.object_transfer.test_object_transfer_learning_heist import get_possible_object_assignments, determine_possible_object_maps
from test.object_transfer.test_object_transfer_exploration import determine_transition_given_action


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

    def add_experience(self, action: int, state: int, obs: JointEffect):
        """Records experience of state action transition"""

        # Convert the observation to an outcome, combine with the set of literals to get an example to add to memory
        outcome = Outcome(obs)
        literals, instance_name_map = self.env.get_literals(state)
        example = Example(action, literals, outcome)

        print("Experienced Example:")
        print(example)

        assignments = get_possible_object_assignments(example, self.previous_ruleset)
        self.possible_assignments.add(assignments)
        print("All assignments: ")
        print(self.possible_assignments)
        print()

        # Pare down the object map
        new_object_map = determine_possible_object_maps(self.object_map, self.possible_assignments)

        self.object_map = new_object_map

        if not self.solved and sum(len(possibilities) for possibilities in self.object_map.values()) == len(self.prior_object_names):
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

        transition = determine_transition_given_action(self.env, state, action, self.object_map, self.previous_ruleset)

        if transition is None:
            return []

        # Sometimes we have more than one transition object. This causes bugs in my code. Pick the one that applies
        if len(transition) > 1:
            assert self.solved, "Can't invert unless solved"
            reverse_object_map = {value[0]: key for key, value in self.object_map.items()}
            reverse_object_map["taxi"] = "taxi"

            found_any_match = False
            for t in transition:
                for object_string in t.outcome.value.keys():
                    outcome_known_object_str = object_string.split(".")[0]
                    splits = outcome_known_object_str.split("-")
                    outcome_unknown_object_str = "-".join(splits[:2]) + "-" + reverse_object_map[splits[-1]]
                    print(outcome_unknown_object_str)

                    found_match = False
                    for edge in literals.base_object.edges:
                        edge_string = "taxi-" + str(edge)[:-1]  # Remove number
                        # print(f"Edge String: {edge_string}, u: {outcome_unknown_object_str}")
                        if edge_string == outcome_unknown_object_str:
                            found_match = True
                            transition = t
                            break

                    if found_match:
                        found_any_match = True

            assert found_any_match, "One of the rules better match"
        else:
            transition = list(transition)
            transition = transition[0]

        transitions = []

        effect = transition.outcome

        atts = []
        outcomes = []

        # Maps unique object name in the tree, to the real instance index in the environment
        name_instance_map = {v: k for k, v in instance_name_map.items()}

        # Important! Need to replace previous rule name from returned transition with it's current mapped
        # name in the env. Or maybe just say once we know everything just plan normally using the mappings?
        # Only do this when solved? Because we can flip the mapping?
        if self.solved:
            att_list = list(effect.value.keys())
            effect_list = list(effect.value.values())
            reverse_object_map = {value[0]: key for key, value in self.object_map.items()}
            reverse_object_map["taxi"] = "taxi"
            # Really complicated way of replacing the string "pass" with "idpyo" in the att name, for example
            att_list = [att.replace(att.split(".")[0].split("-")[-1], reverse_object_map[att.split(".")[0].split("-")[-1]]) for att in att_list]

            effect = JointEffect(att_list, effect_list)

        for ob_att_str, outcome in effect.value.items():
            # obb_att_str is formatted either `taxi.y` or `taxi-IN-key.state`. In general, `ob1-pred1-ob2-pred2...-obn`
            identifier_str, att_name_str = ob_att_str.split(".")

            # We handle the taxi case seperatley, as it isn't attached to anything
            if identifier_str == "taxi":
                unique_name = "taxi0"
            else:
                # We have to find the correct match. A dictionary would be good for this
                # Check for the match by lookng for all of that object, and finding the one with the matching connection
                class_name = identifier_str.split("-")[-1]  # `taxi-IN-key`, will extract `key`

                test_id = 0
                while True:
                    test_name = class_name + str(test_id)
                    node = literals.node_lookup[test_name]
                    edge = node.to_edges[0]
                    test_identifier_str = f"{edge.from_node.object_name[:-1]}-{str(edge)[:-1]}"  # Will be`taxi-IN-key`, maybe `taxi-ON-key`
                    if test_identifier_str == identifier_str:  # Check for a match
                        break
                    test_id += 1  # Try the next object of this type
                # This the object that was being referenced
                unique_name = class_name + str(test_id)

            # Extract the name of the object class, and the identifier number
            class_name, id_str = unique_name[:-1], unique_name[-1]

            # TODO: class_idx requires the original object name again so I guess look it up again? Convert back?
            # This seem like a silly amount of conversions
            if self.solved and class_name != "taxi":
                class_name = self.object_map[class_name]
                assert len(class_name) == 1, "Should only be one object left"
                class_name = class_name[0]

            class_idx = self.env.OB_NAMES.index(class_name)

            att_idx = self.env.ATT_NAMES[class_idx].index(att_name_str)
            instance_id = name_instance_map[unique_name]

            att_range = self.env.instance_index_map[instance_id]
            att = att_range[0] + att_idx

            atts.append(att)
            outcomes.append(outcome)

        # Only create new effect if it isn't a JointNoEffect
        if len(atts) > 0:
            effect = JointEffect(atts, outcomes)

        transitions.append(Transition(effect, 1.0))

        return transitions

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
