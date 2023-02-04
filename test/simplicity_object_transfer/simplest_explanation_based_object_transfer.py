"""
Created on 11/20/22 by Ethan Frank

To answer the question of what object this other object is most like, we try all possible combinations
and see which, when inserted as an extra example, creates the simplest ruleset as an outcome.
"""

import pickle
import random
import numpy as np
import itertools

from environment.symbolic_taxi import SymbolicTaxi
from environment.symbolic_heist import SymbolicHeist
from symbolic_stochastic_domains.symbolic_classes import Example, ExampleSet, Rule, RuleSet, DeicticReference, Outcome
from symbolic_stochastic_domains.learn_ruleset_outcomes import RulesetLearner


def main():
    random.seed(2)
    np.random.seed(2)

    # Load previous taxi examples and rules
    with open("runners/symbolic_heist_rules.pkl", 'rb') as f:
        taxi_ruleset: RuleSet = pickle.load(f)

    with open("runners/heist_examples.pkl", 'rb') as f:
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

    ruleset_learner = RulesetLearner(env, use_prior_names=True)

    # env.restart(init_state=[1, 1])

    # Create object map: This is our belief for what object is what other object
    # object_map = {unknown: prior_object_names.copy() for unknown in env.get_object_names() if unknown != "taxi"}
    object_map_counts = {unknown: {name: 0 for name in prior_object_names} for unknown in env.get_object_names() if unknown != "taxi"}
    print(object_map_counts)

    for i in range(200):

        # Take a step
        curr_state = env.get_state()
        action = random.randint(0, env.get_num_actions()-1)
        literals, outcome, name_id_map = env.step(action)
        example = Example(action, literals, outcome)
        env.draw_world(curr_state, delay=0)

        # if i < 29:
        #     continue

        print()
        print(i, example)

        # Sets are unordered so we need to sort this one and convert it to a list so our mappings are consistent
        state_objects = sorted(list(set([diectic_obj.split("-")[-1] for diectic_obj in literals.referenced_objects])))

        mappings_to_choose_from = (prior_object_names for _ in state_objects)
        permutations = itertools.product(*mappings_to_choose_from)
        store_permutations = []
        complexities = []

        # Initial complexity of the ruleset
        initial_complexity = sum([len(rule.context.nodes) for rule in taxi_ruleset.rules])

        for permutation in permutations:
            # Remove ones where there is a duplicate assignment. Two objects can not be mapped to the same
            # Technically there should be no reason why not but it breaks literals.copy_replace_names
            if len(set(permutation)) != len(permutation):
                continue

            mapping = {state_object: permute_object for state_object, permute_object in zip(state_objects, permutation)}
            mapping["taxi"] = "taxi"  # Taxi has to be there but always maps to itself

            # print(mapping)

            # Replace objects referenced in outcome and literals with their new name
            new_literals = literals.copy_replace_names(mapping)

            # Copy outcome but replace names. Has to be a cleaner way to do this.
            # Problem is we only calculate the hash in the constructor
            new_outcome = Outcome(
                [DeicticReference(key.from_ob, key.edge_type, mapping[key.to_ob] if key.to_ob != '' else '', key.att_name, key.att_num)
                 for key in outcome.value.keys()],
                list(outcome.value.values()),
                outcome.no_effect)

            # Create new example
            new_example = Example(action, new_literals, new_outcome)

            # TODO: need to somehow use hashing or some more effecient way of figuring this out
            # TODO: What to do when there is a contradiction? Set to 0 probability?
            # Check if there is a contradiction: We have experienced this exact set before but had a different outcome
            found = False
            for ex in taxi_examples.examples:
                if action == ex.action and new_literals == ex.state and new_outcome != ex.outcome:
                    print("Whoops, that's a contradiction!", new_literals, ex.state, new_outcome, ex.outcome)
                    found = True
                    break

            if found:
                continue

            # Add the example, learn the new ruleset, then remove the example for the next
            taxi_examples.add_example(new_example)
            new_ruleset = ruleset_learner.learn_ruleset(taxi_examples)
            taxi_examples.remove_example(new_example)

            # TODO: Needs to take into account properties
            complexity = sum([len(rule.context.nodes) for rule in new_ruleset.rules])
            complexity = complexity - initial_complexity

            complexities.append(complexity)
            store_permutations.append(permutation)

            # print(complexity)
            # print(new_ruleset)
            # print()

            # Perhaps when the new rule doesn't mention the object we don't count it
            # because it has nothing to do with it.

        for complexity, permutation in zip(complexities, store_permutations):
            print(f"{complexity}: {state_objects}->{permutation}")

        # Update the counts
        for complexity, permutation in zip(complexities, store_permutations):
            for unknown, known in zip(state_objects, permutation):
                object_map_counts[unknown][known] += 1 if complexity > 0 else 0

        print(object_map_counts)

        # Compute probabilities from complexities:
        alpha = 1  # exponent base
        for unknown, knowns in object_map_counts.items():
            counts = -np.array(list(knowns.values()))
            probabilities = np.exp(counts / alpha)
            probabilities = probabilities / np.sum(probabilities)
            print(unknown, ", ".join(f"{a:.3f}" for a in probabilities))


if __name__ == "__main__":
    main()
