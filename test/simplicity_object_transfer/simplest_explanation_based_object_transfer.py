"""
Created on 11/20/22 by Ethan Frank

To answer the question of what object this other object is most like, we try all possible combinations
and see which, when inserted as an extra example, creates the simplest ruleset as an outcome.
"""

import pickle
import random
import numpy as np
import itertools
from typing import List

from environment.symbolic_taxi import SymbolicTaxi
from environment.symbolic_heist import SymbolicHeist
from symbolic_stochastic_domains.symbolic_classes import Example, ExampleSet, Rule, RuleSet, DeicticReference, Outcome
from symbolic_stochastic_domains.learn_ruleset_outcomes import RulesetLearner


def remap_examples(examples: ExampleSet, mapping: dict, previous_examples: ExampleSet) -> List[Example]:
    new_example_list = []
    for example in examples.examples.keys():
        state = example.state
        action = example.action
        outcome = example.outcome

        # Replace objects referenced in outcome and literals with their new name
        new_literals = state.copy_replace_names(mapping)

        # Copy outcome but replace names. Has to be a cleaner way to do this.
        # Problem is we only calculate the hash in the constructor
        new_outcome = Outcome(
            [DeicticReference(key.from_ob, key.edge_type, mapping[key.to_ob] if key.to_ob != '' else '', key.att_name, key.att_num)
             for key in outcome.value.keys()],
            list(outcome.value.values()),
            outcome.no_effect)

        # Create new example
        new_example = Example(example.action, new_literals, new_outcome)

        # TODO: need to somehow use hashing or some more effecient way of figuring this out
        # TODO: What to do when there is a contradiction? Set to 0 probability?
        # Check if there is a contradiction: We have experienced this exact set before but had a different outcome
        for ex in previous_examples.examples:
            if action == ex.action and new_outcome != ex.outcome and (new_literals.base_object.string_no_numbers() == ex.state.base_object.string_no_numbers()):
                print("Whoops, that's a contradiction!", new_literals, ex.state, new_outcome, ex.outcome)
                print(mapping)
                return None

        # Also need to check new examples as we begin remapping!
        for ex in new_example_list:
            if action == ex.action and new_outcome != ex.outcome and (new_literals.base_object.string_no_numbers() == ex.state.base_object.string_no_numbers()):
                print("Whoops, that's a contradiction!", new_literals, ex.state, new_outcome, ex.outcome)
                print(mapping)
                return None

        new_example_list.append(new_example)

    return new_example_list


def get_object_permutation_rule_complexities(mappings_to_choose_from, old_ruleset, old_examples, new_examples, learner, objects):
    permutations = itertools.product(*mappings_to_choose_from)
    store_permutations = []
    complexities = []
    rulesets = []

    # Initial complexity of the ruleset
    initial_complexity = sum([len(rule.context.nodes) for rule in old_ruleset.rules])

    for permutation in permutations:
        mapping = {state_object: permute_object for state_object, permute_object in zip(objects, permutation)}
        mapping["taxi"] = "taxi"  # Taxi has to be there but always maps to itself

        remaped_examples = remap_examples(new_examples, mapping, old_examples)

        # Indicates a contradiction in the mapping. Indicate this with a huge complexity, and move on
        if remaped_examples is None:
            complexities.append(1000)
            store_permutations.append(permutation)
            rulesets.append(None)
            continue

        print(mapping)
        # Add the examples, learn the new ruleset, then remove the example for the next
        old_examples.add_examples(remaped_examples)
        new_ruleset = learner.learn_ruleset(old_examples)
        old_examples.remove_examples(remaped_examples)

        # TODO: Needs to take into account properties
        complexity = sum([len(rule.context.nodes) for rule in new_ruleset.rules])
        complexity = complexity - initial_complexity

        complexities.append(complexity)
        store_permutations.append(permutation)
        rulesets.append(new_ruleset)

    return complexities, store_permutations, rulesets


def main():
    random.seed(2)
    np.random.seed(2)

    # Load previous taxi examples and rules
    with open("runners/data/heist_rules.pkl", 'rb') as f:
        taxi_ruleset: RuleSet = pickle.load(f)

    with open("runners/data/heist_examples.pkl", 'rb') as f:
        taxi_examples: ExampleSet = pickle.load(f)

    # Copy so hashes are updated
    taxi_ruleset = taxi_ruleset.copy()
    taxi_examples = taxi_examples.copy()

    # New env with shuffled names. Get previous names that this env had
    env = SymbolicHeist(stochastic=False, shuffle_object_names=True)
    prior_object_names = env.OB_NAMES.copy()
    prior_object_names.append("wall")
    prior_object_names.remove("taxi")
    env.restart()
    print(env.object_name_map)

    ruleset_learner = RulesetLearner()

    # Stores examples experienced so far
    new_examples = ExampleSet()

    # Names of objects that have been seen so far, so we can do permutations
    current_object_names = set()

    # env.restart(init_state=[1, 1])

    # Create object map: This is our belief for what object is what other object
    # object_map = {unknown: prior_object_names.copy() for unknown in env.get_object_names() if unknown != "taxi"}
    # print(object_map_counts)

    actions = [0, 0, 1, 0, 3, 3, 4, 1, 1, 1, 2, 2, 4]
    for i in range(len(actions)):

        # Take a step
        curr_state = env.get_state()
        # action = random.randint(0, env.get_num_actions()-1)
        action = actions[i]
        literals, outcome, name_id_map = env.step(action)
        example = Example(action, literals, outcome)
        env.draw_world(curr_state, delay=1)

        # if i < 120:
        #     continue

        print()
        print(i, example)
        new_examples.add_example(example)

        # Sets are unordered so we need to sort this one and convert it to a list so our mappings are consistent
        # Wait, why do we need to sort it?
        state_objects = sorted(list(set([diectic_obj.split("-")[-1] for diectic_obj in literals.referenced_objects])))

        # Keep track of all objects seen so far
        current_object_names.update(state_objects)
        print(current_object_names)

        # mappings_to_choose_from = (prior_object_names for _ in state_objects)
        mappings_to_choose_from = (prior_object_names + [f"object{i}"] for i in range(len(current_object_names)))
        print(prior_object_names, current_object_names)
        # TODO: Notice this maps ones that aren't in the current state? (in current_objects_names)
        # Should we only care about checking things that are in the current state? or always the full thing?
        # mappings_to_choose_from = (prior_object_names for _ in current_object_names)

        complexities, permutations = get_object_permutation_rule_complexities(
            mappings_to_choose_from, taxi_ruleset, taxi_examples, new_examples, ruleset_learner, current_object_names
        )

        for complexity, permutation in zip(complexities, permutations):
            print(f"{complexity}: {state_objects}->{permutation}")

        min_complexity = min(complexities)
        print(f"Min complexity: {min_complexity}")
        if min_complexity == 0:
            print("Ruleset changed!")
        # assert min_complexity == 0, "Not yet deal with a changing ruleset"

        object_map_counts = {unknown: {name: 0 for name in prior_object_names} for unknown in env.get_object_names() if unknown != "taxi"}

        for complexity, permutation in zip(complexities, permutations):
            for unknown, known in zip(state_objects, permutation):
                if known not in object_map_counts[unknown]:
                    object_map_counts[unknown][known] = 0
                object_map_counts[unknown][known] += 1 if complexity == min_complexity else 0

        for unknown, knowns in object_map_counts.items():
            print(unknown, ", ".join([f"{c}" for c in knowns.values()]))


if __name__ == "__main__":
    main()
