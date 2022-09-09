"""
Created on 9/7/22 by Ethan Frank

When running my fixed heist runner (added get_object_names), learning ruleset freezes)
"""
import pickle
import time

from symbolic_stochastic_domains.learn_ruleset_outcomes import RulesetLearner
from environment.symbolic_heist import SymbolicHeist


if __name__ == "__main__":
    with open("/home/edf42001/Documents/College/Thesis/masters-thesis/runners/heist_runner_examples_freeze.pkl", 'rb') as f:
        examples = pickle.load(f)

    env = SymbolicHeist(stochastic=False)

    print(examples)
    print()
    start_time = time.perf_counter()
    learner = RulesetLearner(env)
    ruleset = learner.learn_ruleset(examples)
    end_time = time.perf_counter()
    print(f"Ruleset learning took {end_time - start_time:.3f} (# of examples {len(examples.examples)})")
    print(ruleset)
