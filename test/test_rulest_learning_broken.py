import pickle
import time

from symbolic_stochastic_domains.learn_ruleset_outcomes import learn_ruleset_outcomes

# Something's up, because it works here?
with open("../runners/example.pkl", 'rb') as f:
    examples = pickle.load(f)

examples = examples.copy()


start_time = time.perf_counter()
ruleset = learn_ruleset_outcomes(examples)
end_time = time.perf_counter()
print(f"Took {end_time-start_time}")
print("Resulting ruleset:")
print(ruleset)
