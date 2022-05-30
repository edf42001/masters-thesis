import pickle

from effects.effect import JointEffect, JointNoEffect, Increment
from symbolic_stochastic_domains.symbolic_classes import Rule, OutcomeSet, Outcome
from symbolic_stochastic_domains.learn_parameters import learn_params


if __name__ == "__main__":
    with open("stochastic_doorworld_exampleset.pkl", "rb") as f:
        examples = pickle.load(f)

    action = 0
    context = []
    outcomes = OutcomeSet()
    outcomes.add_outcome(Outcome(JointNoEffect()), 0.0)
    outcomes.add_outcome(Outcome(JointEffect(["taxi.x"], [Increment(1, 0)])), 0.0)

    rule1 = Rule(action, context, outcomes)
    params = learn_params(rule1, examples)

    action = 1
    context = []
    outcomes = OutcomeSet()
    outcomes.add_outcome(Outcome(JointNoEffect()), 0.0)
    outcomes.add_outcome(Outcome(JointEffect(["taxi.x"], [Increment(0, 1)])), 0.0)

    rule2 = Rule(action, context, outcomes)
    params = learn_params(rule2, examples)
