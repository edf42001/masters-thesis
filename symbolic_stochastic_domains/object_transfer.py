"""
Created on 10/14/22 by Ethan Frank

Utility functions for evaluating deterministic 1-to-1 object mapping
"""

from typing import List, Set
import itertools

import numpy as np

from symbolic_stochastic_domains.symbolic_classes import RuleSet, Outcome, Example, PredicateTree


class ObjectAssignment:
    """
    Keeps track of one possible assignment from unknown to known
    objects that would match the current observation
    """

    def __init__(self):
        self.positives = {}
        self.negatives = {}

    def add_positive(self, unknown, known):
        assert unknown not in self.positives, "One object can never be two different objects"
        self.positives[unknown] = known

    def add_negative(self, unknown, known):
        # Use lists here because it is possible that 1 unknown is not two different objects
        if unknown not in self.negatives:
            self.negatives[unknown] = [known]
        else:
            self.negatives[unknown].append(known)

    def dict_to_str(self, dictionary):
        return str({key: ", ".join(values) for key, values in dictionary.items()})

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if len(self.positives) > 0 and len(self.negatives) > 0:
            return str(self.positives) + " - ~" + self.dict_to_str(self.negatives)
        elif len(self.negatives) > 0:
            return "~" + self.dict_to_str(self.negatives)
        elif len(self.positives) > 0:
            return str(self.positives)
        else:
            return "{}"

    def __eq__(self, other):
        return self.positives == other.positives and self.negatives == other.negatives


class ObjectAssignmentList:
    """
    A collection of object assignments retrieved from an example.
    At least one, possibly more, of the assignments must be the true cause
    """

    def __init__(self, assignments):
        self.assignments = assignments
        self.hash = hash(self.__str__())

    def add_assignments(self, assignment_list):
        self.assignments.extend(assignment_list.assignments)
        self.hash = hash(self.__str__())

    def __str__(self):
        return "[" + " or ".join(str(a) for a in self.assignments) + "]"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return all(a == b for a, b in zip(self.assignments, other.assignments))


def determine_bindings_for_same_outcome(condition: PredicateTree, state: PredicateTree):
    """
    Say we have a tree representing the condition for a rule.
    The other tree is a state, with different object names.
    What mappings of objects names make it so the rule condition tree is applicable to the other?

    This is for the specific case when the rule says an outcome and the actual outcome matched.
    """

    # Make sure to remove the number from all the object names

    assignment = ObjectAssignment()

    # First, check all positive edges. All of these must match
    # However, note that if any properties of the positive edges don't match, this was not the real rule
    # That caused this to occur. So, we return None just like below, and hope that some other rule covered
    # this occurrence.
    for edge in condition.base_object.edges:
        # Find matching edge
        found = False
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                # This is the match, these objects are the same

                # Check if any of the properties do not match.
                for prop, value in edge.to_node.properties.items():
                    # If the property is not present, assume it is false.
                    # Then, I don't have to add false properties to every object, but the learner can't cheat
                    # by using which properties are present to distinguish objects
                    other_value = False if prop not in edge2.to_node.properties else edge2.to_node.properties[prop]

                    if value != other_value:
                        return None

                assignment.add_positive(edge2.to_node.object_name, edge.to_node.object_name)
                found = True
                break

        # If this happens, it is because this action has more than one rule that covers it,
        # and this condition was not the one. This is fine, the other condition will cover it.
        # If neither cover it, then we have a paradox in the code.
        if not found:
            return None

    # Now all negative edges, any matches here must be false
    for edge in condition.base_object.negative_edges:
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                assignment.add_negative(edge2.to_node.object_name, edge.to_node.object_name)
                break

    # For later down the line, it'll be easier if the list is completely empty, instead of having and empty object
    if len(assignment.negatives) == 0 and len(assignment.positives) == 0:
        return ObjectAssignmentList([])
    else:
        return ObjectAssignmentList([assignment])


def determine_bindings_for_no_outcome(condition: PredicateTree, state: PredicateTree) -> ObjectAssignmentList:
    """
    Say we have a tree representing the condition for a rule.
    The other tree is a state, with different object names.
    What mappings of objects names make it so the rule condition tree is applicable to the other?

    This is for the specific case when the rule says an outcome but instead nothing happens.
    """

    assignments = []

    # First, make sure all the positive edges match. If they don't we can't really give a reason,
    # because the missing positive edge could be why nothing happened

    # The same is true of properties. If any of the properties of positive edges do not match, that could be
    # why nothing happened. So we can't really give any more information.

    for edge in condition.base_object.edges:
        # Find matching edge
        found = False
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                # This is the match, these objects are the same

                # Check if any of the properties do not match.
                for prop, value in edge.to_node.properties.items():
                    # If the property is not present, assume it is false.
                    # Then, I don't have to add false properties to every object, but the learner can't cheat
                    # by using which properties are present to distinguish objects
                    other_value = False if prop not in edge2.to_node.properties else edge2.to_node.properties[prop]

                    if value != other_value:
                        return ObjectAssignmentList([])

                found = True
                break

        if not found:
            return ObjectAssignmentList([])

    # Now, let's look for negative edges that perhaps are causing the issue
    # Also some positive edges could be missing. Each of these are independent, add them to the array separately.
    for edge in condition.base_object.edges:
        for edge2 in state.base_object.edges:  # (Could probably combine this with the above, then throw out if bad?)
            if edge.type == edge2.type:
                assignment = ObjectAssignment()
                assignment.add_negative(edge2.to_node.object_name, edge.to_node.object_name)
                assignments.append(assignment)
                break

    for edge in condition.base_object.negative_edges:
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                assignment = ObjectAssignment()
                assignment.add_positive(edge2.to_node.object_name, edge.to_node.object_name)
                assignments.append(assignment)
                break

    return ObjectAssignmentList(assignments)


def information_gain_of_action(env, state: int, action: int, object_map, prev_ruleset: RuleSet) -> float:
    """
    Returns the expected information gain from taking an action, given the current knowledge of the world.
    measured based on net decrease in number of possibilities in object map (wrong, it's better to have one go to 0
    then 3 go down by 1), use entropy total decrease instead)
    """
    # List of known object names without taxi and with wall (wall is static so is not in the list normally)
    known_objects = set(env.OB_NAMES)
    known_objects.add("wall")
    known_objects.remove("taxi")

    # print()
    # print(f"State {state}")
    # print(f"Action {action}")
    # print(f"Current object map: {object_map}")

    literals, _ = env.get_literals(state)
    # print(f"Literals: {literals}")

    # Get the rules that apply to this situation
    applicable_rules = [rule for rule in prev_ruleset.rules if rule.action == action]
    # assert len(applicable_rules) == 1, "My code only works for one rule for now"
    # for rule in applicable_rules:
    #     print(f"Applicable rules: {rule}")
    #     print()

    # Description of how my brute force algorithm works.
    # Step 1: There are some objects in the state, and some we have the whole list of previously known object
    # Create every combination of mappings possible
    # context_objects = set([diectic_obj.split("-")[-1] for diectic_obj in rule.context.referenced_objects])
    state_objects = set([diectic_obj.split("-")[-1] for diectic_obj in literals.referenced_objects])
    # print(f"Context objects: {context_objects}")
    # print(f"State objects: {state_objects}")

    # Track total info gain and number of permutations so we can calculate an expected info gain
    total_info_gain = 0
    num_permutations = 0

    mappings_to_choose_from = (object_map[unknown_object] for unknown_object in state_objects)
    permutations = itertools.product(*mappings_to_choose_from)

    # permutations = itertools.permutations(known_objects, len(state_objects))
    # print("Permutations:")
    for permutation in permutations:
        # Remove ones where there is a duplicate assignment. Two objects can not be mapped to the same
        # Technically there should be no reason why not but it breaks literals.copy_replace_names
        if len(set(permutation)) != len(permutation):
            continue

        num_permutations += 1

        mapping = {state_object: permute_object for state_object, permute_object in zip(state_objects, permutation)}
        # print(mapping)
        mapping["taxi"] = "taxi"  # Taxi has to be there but always maps to itself

        new_literals = literals.copy_replace_names(mapping)
        # print(f"New tree: {new_literals}")

        # Check each rule. Because rules are constructed to be mutually exclusive, either one of them will
        # be applicable, or none of them will be applicable.
        # TODO: Could simply computations by noting there are some times where a rule will never apply.
        # i.e., if it is missing edges that are relevant

        # If one rule applies, find that rule, otherwise, the effect will be NoEffect
        outcome = Outcome([], [], no_effect=True)
        for rule in applicable_rules:
            # print(f"Rule: {rule}")
            assert len(rule.outcomes.outcomes) == 1, "Only deal with one possible outcome"

            applicable = new_literals.base_object.contains(rule.context.base_object)
            # print(f"Applicable: {applicable}")
            # print()

            if applicable:
                # TODO Really you could put a break statement in here but I'm leaving in this assertion just to check
                assert outcome.is_no_effect(), "A second rule was applicable which doesn't make sense"
                outcome = rule.outcomes.outcomes[0]

        # print(f"Predicted outcome: {outcome}")

        # Get object assignments from this example. Is this the part that could be shortcut?
        example = Example(action, literals, outcome)
        possible_assignment = get_possible_object_assignments(example, prev_ruleset)
        # print(f"Possible assignments: {possible_assignment}")

        # print("Previous object map:")
        # print(object_map)

        new_object_map = determine_possible_object_maps(object_map, possible_assignment)
        prev_num_options = sum(len(possibilities) for possibilities in object_map.values())
        new_num_options = sum(len(possibilities) for possibilities in new_object_map.values())

        # print(f"New object map: {new_object_map}")
        # print(f"Length update: {prev_num_options}->{new_num_options}")

        # Info gain is change in bits required to express number of object possibilities, which is log2 of length
        total_info_gain += np.log2(prev_num_options) - np.log2(new_num_options)

    # Step 2: Assuming that mapping is the real one, see what would happen.

    # Step 3: Given what would happen and what we know, would we gain any information from that action?

    # There is probably a less roundabout method of doing this. For example, maybe we only care about objects
    # That are referenced the way the objects in the rule is referenced?
    # See if the ones where those are the same have same/different effects
    # print(f"Total info {total_info_gain}, num permutations: {num_permutations}")
    return total_info_gain / num_permutations


def information_gain_of_state(env, state: int, object_map, prev_ruleset: RuleSet) -> float:
    """Returns the total info gain over all actions for a state"""
    return sum([information_gain_of_action(env, state, a, object_map, prev_ruleset) for a in range(env.get_num_actions())])


def determine_transition_given_action(env, state: int, action: int, object_map, prev_ruleset: RuleSet) -> Set[Outcome]:
    """
    Given a current state and current object map belief,
    what are the possible next states for a specific actions?
    If there is more than one possibility depending on object map bindings, return None.
    This is an opportunity to gain information (possibly?). Otherwise, return the transition? Or the state?
    """

    # All this is the exact same as information_gain_of_action. See there for more info.
    # In fact, all this code is the exact same, because in order to know what info we learn we
    # have to figure out what the transition was base on the previous rule.

    literals, _ = env.get_literals(state)

    # Get the rules that apply to this situation
    applicable_rules = [rule for rule in prev_ruleset.rules if rule.action == action]
    # assert len(applicable_rules) == 1, "My code only works for one rule for now"
    # rule = applicable_rules[0]

    # TODO: Exclude permutations that are not relavent to the rule, or not relevant to the current object map
    # context_objects = set([diectic_obj.split("-")[-1] for diectic_obj in rule.context.referenced_objects])

    # Objects that are currently in the state
    state_objects = set([diectic_obj.split("-")[-1] for diectic_obj in literals.referenced_objects])

    applicable_tracker = None  # Stores outcome as we process permutations so we can look for contradictions

    # Filter the object map by only objects in the state. Then, create all possible combinations of state objects
    # and what we believe they could be. Then, check if all the outcomes match.
    mappings_to_choose_from = (object_map[unknown_object] for unknown_object in state_objects)
    permutations = itertools.product(*mappings_to_choose_from)

    for permutation in permutations:
        # Remove ones where there is a duplicate assignment. Two objects can not be mapped to the same
        # Technically there should be no reason why not but it breaks literals.copy_replace_names
        if len(set(permutation)) != len(permutation):
            continue

        mapping = {state_object: permute_object for state_object, permute_object in zip(state_objects, permutation)}
        mapping["taxi"] = "taxi"  # Taxi has to be there but always maps to itself

        new_literals = literals.copy_replace_names(mapping)

        # Because rules are constructed to be mutually exclusive, either one will apply, or neither will apply
        # If one applies, take that as the outcome. If none apply, then nothing happens. It's like an OR.
        # TODO: What if the same action leads to different outcomes depending on the rule?
        any_applied = False
        for rule in applicable_rules:
            assert len(rule.outcomes.outcomes) == 1, "Only deal with one possible outcome"

            applicable = new_literals.base_object.contains(rule.context.base_object)

            if applicable:
                any_applied = True

        if applicable_tracker is None:
            applicable_tracker = any_applied
        elif applicable_tracker != any_applied:  # If we every get different answers, we don't know so return none
            return None

    # Returns the outcome if something will happen, no effect if nothing was applicable to the rule
    # TODO: This returns the object names from the rule. It should probably replace those with the current object names
    return set(rule.outcomes.outcomes[0] for rule in applicable_rules) if applicable_tracker else {Outcome([], [], no_effect=True)}


def get_possible_object_assignments(example: Example, prev_ruleset: RuleSet) -> List[ObjectAssignmentList]:
    """
    Return possible unknown to known object assignments for this example
    and the previously known ruleset
    """

    action = example.action
    literals = example.state
    outcome = example.outcome

    # Get the rules that apply to this situation
    applicable_rules = [rule for rule in prev_ruleset.rules if rule.action == action]

    # To determine possible causes, loop over all rules. Some of them might not even apply,
    # So I think we first need to check if they could apply. Otherwise, ignore that rule?
    # Because when there was only one rule we knew that was the one that should apply?
    # But sometimes it didn't? No I think it always did

    # Keeps track of if we found assignments. Only used to verify that when more than one rule is applicable,
    # at least one is matched if the outcomes matched.
    # See determine_bindings_for_same_outcome for more info (when it returns None)
    found_an_assignment = False

    all_all_assignments = []
    # print(f"{len(applicable_rules)} applicable rules")

    # Nothing happened. In this case, and all the results from determine_bindings_for_no_outcome,
    # Because none of them can be the case.
    if outcome.is_no_effect():
        for rule in applicable_rules:
            assignments = determine_bindings_for_no_outcome(rule.context, literals)
            all_all_assignments.append(assignments)
    else:
        all_assignments = ObjectAssignmentList([])
        for rule in applicable_rules:
            # Wait, could the same action have different outcomes depending on the situation?
            outcome_occured = outcome.is_no_effect() == rule.outcomes.outcomes[0].is_no_effect()

            # Use different reasoning based on if we had a positive or negative example
            if outcome_occured:
                assignments = determine_bindings_for_same_outcome(rule.context, literals)
            else:
                assignments = determine_bindings_for_no_outcome(rule.context, literals)

            # print(f"Assignments for\n{rule}:\n{assignments}")
            # print()
            if assignments is not None:
                found_an_assignment = True
                all_assignments.add_assignments(assignments)

        assert not (len(applicable_rules) > 1 and not found_an_assignment), "At least one rule must not return None"

        # Make sure all rules have the same outcome. What happens if they don't?
        # Update: I don't even know if this is correct?
        # No rules will have an outcome of no effect?
        assert all(rule.outcomes.outcomes[0].is_no_effect() ==
                   applicable_rules[0].outcomes.outcomes[0].is_no_effect()
                   for rule in applicable_rules)

        all_all_assignments = [all_assignments]

    # print(f"All possible assignments: {all_assignments}")
    # print(f"All All possible assignments: {all_all_assignments}")
    return all_all_assignments


def determine_possible_object_maps(object_map: dict, possible_assignments: List[ObjectAssignmentList]):
    """
    Tries to figure out which assignments are the true ones and which are not,
    in the process learning which object is which
    """

    # Create a copy so as to not modify the original object
    # Need to use deepcopy on object map because it is a dict of lists
    object_map = {key: value.copy() for key, value in object_map.items()}

    # Basically, one or more assignment in each assignment list must be true, find which
    for assignment_list in possible_assignments:
        # Create a list of whether or not each assignment is False. This helps us narrow down what must be true
        are_false = [False] * len(assignment_list.assignments)
        for i, assignment in enumerate(assignment_list.assignments):
            has_positives = len(assignment.positives) > 0
            has_negatives = len(assignment.negatives) > 0

            assert not (has_positives and has_negatives), "Should never have both, I think"

            # We can run different checks depending on whether or not this is a positive or negative assignment
            if has_positives:
                is_false = any(known not in object_map[unknown] for unknown, known in assignment.positives.items())
                are_false[i] = is_false
            else:
                is_false = any((known in object_map[unknown] and len(object_map[unknown]) == 1)
                               for unknown, knowns in assignment.negatives.items() for known in knowns)
                are_false[i] = is_false

        # Pick out only the ones that are definitely not false
        not_false_assignments = [a for (a, is_false) in zip(assignment_list.assignments, are_false) if not is_false]

        # If the length is one, then we know that one must be true, so we can apply it
        if len(not_false_assignments) == 1:
            assignment = not_false_assignments[0]

            # If we know it is one thing, set all of those to their correct value
            for unknown, known in assignment.positives.items():
                assert known in object_map[unknown], f"We say it must be true so it better be an option: {unknown}"
                object_map[unknown] = [known]

            # Remove everything from negatives (it may have been removed already)
            for unknown, knowns in assignment.negatives.items():
                for known in knowns:  # Negatives are list of things the object isn't
                    if known in object_map[unknown]:
                        object_map[unknown].remove(known)

        # If there are two possible assignments, for now, let's just apply all of them
        # If there are multiple positives, say it could be either. If negatives, just remove all of those
        elif len(not_false_assignments) > 1:
            # print("Dealing with more than one assignment")
            # print(assignment_list)

            # Generate list of positive possibilities for each assignment
            # Narrow down the object map using this list
            positive_possibilities = dict()
            has_negatives = False
            for assignment in not_false_assignments:
                for unknown, known in assignment.positives.items():
                    if unknown not in positive_possibilities:
                        positive_possibilities[unknown] = [known]
                    else:
                        positive_possibilities[unknown].append(known)

                # Note that negatives are lists.
                if len(assignment.negatives) > 0:
                    has_negatives = True

            # print(positive_possibilities)
            # Only update if we don't have any negatives. This is because some of them could be the reason
            # Technically we should ask if any of them are actually true, otherwise we can't do much
            # If we don't know which one is the true one.
            if not has_negatives:
                for unknown, knowns in positive_possibilities.items():
                    object_map[unknown] = [value for value in object_map[unknown] if value in knowns]

        # If length is 0 we don't have to do anything

    # Check if any objects have been brought to 1, if so, remove those from the others
    # If this causes another to go to one, continue looping
    changed = True
    while changed:
        changed = False
        for known, unknowns in object_map.items():
            if len(unknowns) == 1:
                unknown = unknowns[0]
                for known2, unknowns2 in object_map.items():
                    if known2 != known and unknown in unknowns2:
                        unknowns2.remove(unknown)
                        changed = True

    # Verify no lists went to 0, which indicates a conflict in rules somewhere
    for known, unknowns in object_map.items():
        assert len(unknowns) > 0, f"Should always have a belief about what objects it is: {known}"

    return object_map

# TODO: test multiple rules, properties effects on learning ruleset.
