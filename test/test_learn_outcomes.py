import pickle

from symbolic_stochastic_domains.symbolic_classes import Rule, OutcomeSet
from symbolic_stochastic_domains.learn_outcomes import learn_outcomes
from environment.symbolic_door_world import TouchLeft, Taxi, Switch, Predicate


if __name__ == "__main__":
    with open("stochastic_doorworld_exampleset.pkl", "rb") as f:
        examples = pickle.load(f)

    action = 0
    context = [TouchLeft(Taxi("taxi", x=1), Switch("switch", x=3))]
    # context = []
    outcomes = OutcomeSet()

    rule1 = Rule(action, context, outcomes)
    learn_outcomes(rule1, examples)
    print("Final rule:")
    print(rule1)

    # action = 1
    # context = []
    # outcomes = OutcomeSet()
    # rule2 = Rule(action, context, outcomes)
    # learn_outcomes(rule2, examples)
