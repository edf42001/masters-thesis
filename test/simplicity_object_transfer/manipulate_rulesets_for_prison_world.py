"""
Created on 10/3/22 by Ethan Frank

We have a taxi ruleset and a heist ruleset, but they need to be updated so that a state of 0 indicates
being picked up, and the dropoff action needs to be remapped to 6.

Turns out, most of this wasn't necessary, because the issue was not that picking up wasn't 0,
but that I had to return more than one state from the transition function. But, the combining
part was necessary.
"""

import pickle
from symbolic_stochastic_domains.symbolic_classes import Rule, Outcome, DeicticReference, Example, ExampleSet
from symbolic_stochastic_domains.predicates_and_objects import PredicateType
from effects.effect import SetToNumber


if __name__ == "__main__":
    with open("runners/data/taxi_learned_data.pkl", 'rb') as f:
        taxi_ruleset, taxi_examples, taxi_experience = pickle.load(f)

    print(taxi_ruleset)

    dropoff_passenger_rule: Rule = taxi_ruleset.rules[5]
    dropoff_passenger_rule.action = 6  # More actions in prison world, need to increase action number (5 is unlock)
    # Prison world only has 2 destinations, update accordingly.
    dropoff_passenger_rule.outcomes.outcomes[0] = Outcome([DeicticReference("taxi", PredicateType.IN, "pass", "state")], [SetToNumber(2, 2)])

    print()
    print(taxi_ruleset)

    # Remap dropoff examples
    print(taxi_examples)
    for e in taxi_examples.examples:
        if e.action == 5:  # Same as above, need to update dropoff actions
            e.action = 6

            if not e.outcome.is_no_effect():
                e.outcome = Outcome([DeicticReference("taxi", PredicateType.IN, "pass", "state")], [SetToNumber(2, 2)])

    print()
    print(taxi_examples)

    # Remap experiences as well
    for experience in taxi_experience.experiences[0]:
        if 5 in taxi_experience.experiences[0][experience]:
            taxi_experience.experiences[0][experience][6] = taxi_experience.experiences[0][experience][5]
            del taxi_experience.experiences[0][experience][5]

    for experience in taxi_experience.experiences[1]:
        if 5 in taxi_experience.experiences[1][experience]:
            taxi_experience.experiences[1][experience][6] = taxi_experience.experiences[1][experience][5]
            del taxi_experience.experiences[1][experience][5]

    with open("runners/data/taxi_learned_data_remapped.pkl", 'wb') as f:
        pickle.dump((taxi_ruleset, taxi_examples, taxi_experience), f)
