"""
Created on 10/3/22 by Ethan Frank

We have a taxi ruleset and a heist ruleset, but they need to be updated so that a state of 0 indicates
being picked up, and the dropoff action needs to be remapped to 6.

Turns out, most of this wasn't necessary, because the issue was not that picking up wasn't 0,
but that I had to return more than one state from the transition function. But, the combining
part was necessary.
"""

import pickle
from symbolic_stochastic_domains.symbolic_classes import RuleSet, Rule, Outcome, DeicticReference
from symbolic_stochastic_domains.predicates_and_objects import PredicateType
from effects.effect import SetToNumber


if __name__ == "__main__":
    with open("runners/symbolic_taxi_rules.pkl", 'rb') as f:
        symbolic_taxi_ruleset = pickle.load(f)

    with open("runners/symbolic_heist_rules.pkl", 'rb') as f:
        symbolic_heist_ruleset = pickle.load(f)

    dropoff_passenger_rule: Rule = symbolic_taxi_ruleset.rules[5]
    dropoff_passenger_rule.action = 6  # More actions in prison world, need to increase action number

    prison_world_ruleset = RuleSet([])
    for rule in symbolic_heist_ruleset.rules:
        prison_world_ruleset.add_rule(rule)

    # Copy extra rules from taxi (only passenger stuff)
    prison_world_ruleset.add_rule(symbolic_taxi_ruleset.rules[4])
    prison_world_ruleset.add_rule(symbolic_taxi_ruleset.rules[5])

    # Prison world only has 2 destinations, update accordingly.
    dropoff_passenger_rule: Rule = prison_world_ruleset.rules[9]
    dropoff_passenger_rule.outcomes.outcomes[0] = Outcome([DeicticReference("taxi", PredicateType.IN, "pass", "state")], [SetToNumber(2, 2)])

    with open("runners/symbolic_prison_rules.pkl", 'wb') as f:
        pickle.dump(prison_world_ruleset, f)

    print(symbolic_taxi_ruleset)
    print()
    print(symbolic_heist_ruleset)
    print()
    print(prison_world_ruleset)
