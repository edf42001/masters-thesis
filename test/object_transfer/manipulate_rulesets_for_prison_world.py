"""
Created on 10/3/22 by Ethan Frank

We have a taxi ruleset and a heist ruleset, but they need to be updated so that a state of 0 indicates
being picked up, and the dropoff action needs to be remapped to 6.

Turns out, most of this wasn't necessary, because the issue was not that picking up wasn't 0,
but that I had to return more than one state from the transition function. But, the combining
part was necessary.
"""

import pickle
from symbolic_stochastic_domains.symbolic_classes import RuleSet, Rule
from effects.effect import JointEffect, SetToNumber


if __name__ == "__main__":
    with open("runners/symbolic_taxi_rules.pkl", 'rb') as f:
        symbolic_taxi_ruleset = pickle.load(f)

    with open("runners/symbolic_heist_rules.pkl", 'rb') as f:
        symbolic_heist_ruleset = pickle.load(f)


    pickup_passenger_rule: Rule = symbolic_taxi_ruleset.rules[4]
    pickup_passenger_rule.outcomes.outcomes[0].outcome = JointEffect(["taxi-ON2D-pass.state"], [SetToNumber(0, 0)])
    dropoff_passenger_rule: Rule = symbolic_taxi_ruleset.rules[5]
    dropoff_passenger_rule.action = 6

    pickup_key_rule: Rule = symbolic_heist_ruleset.rules[5]
    pickup_key_rule.outcomes.outcomes[0].outcome = JointEffect(["taxi-ON2D-key.state"], [SetToNumber(0, 0)])
    unlock_lock_rule: Rule = symbolic_heist_ruleset.rules[6]
    unlock_lock_rule.outcomes.outcomes[0].outcome = JointEffect(["taxi-IN-key.state", "taxi-TOUCH_DOWN2D-lock.state"], [SetToNumber(2, 2), SetToNumber(0, 0)])
    pickup_gem_rule: Rule = symbolic_heist_ruleset.rules[7]
    pickup_gem_rule.outcomes.outcomes[0].outcome = JointEffect(["taxi-ON2D-gem.state"], [SetToNumber(0, 0)])

    prison_world_ruleset = RuleSet([])
    for rule in symbolic_heist_ruleset.rules:
        prison_world_ruleset.add_rule(rule)

    # Copy extra rules from taxi (only passenger stuff)
    prison_world_ruleset.add_rule(symbolic_taxi_ruleset.rules[4])
    prison_world_ruleset.add_rule(symbolic_taxi_ruleset.rules[5])

    # Prison world only has 2 destinations, update accordingly.
    dropoff_passenger_rule: Rule = prison_world_ruleset.rules[9]
    dropoff_passenger_rule.outcomes.outcomes[0].outcome = JointEffect(["taxi-IN-pass.state"], [SetToNumber(2, 2)])

    # Watch out, this also saves the new action as 6.
    with open("runners/symbolic_prison_rules.pkl", 'wb') as f:
        pickle.dump(prison_world_ruleset, f)

    with open("runners/symbolic_taxi_rules_updated.pkl", 'wb') as f:
        pickle.dump(symbolic_taxi_ruleset, f)

    with open("runners/symbolic_heist_rules_updated.pkl", 'wb') as f:
        pickle.dump(symbolic_heist_ruleset, f)

    print(prison_world_ruleset)
