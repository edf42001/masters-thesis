from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Rule
from symbolic_stochastic_domains.symbolic_utils import context_matches, rule_score
from symbolic_stochastic_domains.learn_parameters import learn_params


def learn_outcomes(rule: Rule, examples: ExampleSet):
    """
    Given LearnParameters, an algorithm for learning a distribution over outcomes, we can
    now consider the problem of taking an incomplete rule r consisting of a context, an action,
    and perhaps a set of deictic references, and finding the optimal way to fill in the rest of
    the rule: that is, the set of outcomes and the probabilities P that maximize the likelihood/score

    Changes rule in-place
    """

    # print("Rule:")
    # print(rule)
    # print()
    # print("Examples:")
    # print(examples)
    # print()

    # In this case, because we consider each outcome to be unique, we just have to find all outcomes
    # this rule applies to (i.e., matching action and context)
    unique_outcomes = []
    for example in examples.examples.keys():
        if example.outcome in unique_outcomes:
            continue

        if rule.action == example.action and context_matches(rule.context, example.state_set):
            unique_outcomes.append(example.outcome)

    # print("Unique outcomes:")
    # for outcome in unique_outcomes:
    #     print(outcome)
    # print()

    # Update outcome
    rule.outcomes.outcomes = unique_outcomes
    rule.outcomes.probabilities = learn_params(rule, examples)

    # print(f"Final probabilities: {rule.outcomes.probabilities}")
    score = rule_score(rule, examples)
    # print(f"Rule score: {score:0.4}")
