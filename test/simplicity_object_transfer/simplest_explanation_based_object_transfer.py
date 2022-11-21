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
from symbolic_stochastic_domains.symbolic_classes import Example, ExampleSet, Rule, RuleSet, DeicticReference
from symbolic_stochastic_domains.learn_ruleset_outcomes import RulesetLearner


def main():
    random.seed(1)
    np.random.seed(1)

    # Load previous taxi examples and rules
    with open("runners/symbolic_taxi_rules.pkl", 'rb') as f:
        taxi_ruleset: RuleSet = pickle.load(f)

    with open("runners/taxi_examples.pkl", 'rb') as f:
        taxi_examples: ExampleSet = pickle.load(f)

    # Copy so hashes are updated
    taxi_ruleset = taxi_ruleset.copy()
    taxi_examples = taxi_examples.copy()

    # New env with shuffled names. Get previous names that this env had
    env = SymbolicTaxi(stochastic=False, shuffle_object_names=True)
    prior_object_names = env.OB_NAMES.copy()
    prior_object_names.append("wall")
    prior_object_names.remove("taxi")
    env.restart()

    ruleset_learner = RulesetLearner(env, use_prior_names=True)

    for i in range(300):
        # Take a step
        action = random.randint(0, env.get_num_actions()-1)
        literals, outcome, name_id_map = env.step(action)
        example = Example(action, literals, outcome)

        print()
        print(i, example)

        # Sets are unordered so we need to sort this one and convert it to a list so our mappings are consistent
        state_objects = sorted(list(set([diectic_obj.split("-")[-1] for diectic_obj in literals.referenced_objects])))

        mappings_to_choose_from = (prior_object_names for _ in state_objects)
        permutations = itertools.product(*mappings_to_choose_from)
        store_permutations = []
        complexities = []

        for permutation in permutations:
            # Remove ones where there is a duplicate assignment. Two objects can not be mapped to the same
            # Technically there should be no reason why not but it breaks literals.copy_replace_names
            if len(set(permutation)) != len(permutation):
                continue

            mapping = {state_object: permute_object for state_object, permute_object in zip(state_objects, permutation)}
            mapping["taxi"] = "taxi"  # Taxi has to be there but always maps to itself

            new_literals = literals.copy_replace_names(mapping)
            new_example = Example(action, new_literals, outcome)

            # TODO: need to somehow use hashing or some other way of figuring this out
            found = False
            for ex in taxi_examples.examples:
                if action == ex.action and new_literals == ex.state and outcome != ex.outcome:
                    print("Whoops, that's a contradiction!", new_literals, ex.state, outcome, ex.outcome)
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

            complexities.append(complexity)
            store_permutations.append(permutation)

        for complexity, permutation in zip(complexities, store_permutations):
            print(f"{complexity}: {permutation}")


if __name__ == "__main__":
    main()
