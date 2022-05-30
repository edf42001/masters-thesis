from typing import List, Dict

from effects.effect import JointEffect
from environment.symbolic_door_world import Predicate


class Outcome:
    """An outcome is which attributes change and how"""
    def __init__(self, outcome: JointEffect):
        self.outcome = outcome

        # The "value" attribute is the dictionary of att/effect pairs
        self.num_affected_atts = len(self.outcome.value)

    def get_num_affected_atts(self):
        return self.num_affected_atts

    def __str__(self):
        return str(self.outcome)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.outcome == other.outcome

    def __hash__(self):
        # Inherit the unique hash from the joint effect
        return self.outcome.__hash__()


class OutcomeSet:
    """A set of outcomes, each associated with a probability"""
    def __init__(self):
        self.outcomes: List[Outcome] = []
        self.probabilities: List[float] = []

    def add_outcome(self, outcome: Outcome, p: float):
        """Adds an outcome and associated probability to the list"""
        self.outcomes.append(outcome)
        self.probabilities.append(p)

    def get_total_num_affected_atts(self):
        """Returns the total number of changed attributes over all outcomes in the set"""
        return sum([o.get_num_affected_atts() for o in self.outcomes])

    def __str__(self):
        ret = ""
        for p, outcome in zip(self.probabilities, self.outcomes):
            ret += f"{p:0.3}: {outcome}\n"

        # Remove trailing newline
        return ret[:-1]

    def __repr__(self):
        return self.__str__()


class Example:
    """
    An example consists of all the literals in the starting state, the action taken, and the outcomes
    """
    def __init__(self, action: int, state: List[Predicate], outcome: Outcome):
        self.action = action
        self.state = state
        self.outcome = outcome

    def __str__(self):
        ret = ""

        ret += f"Action {self.action}:\n"

        # Only show true literals
        ret += ", ".join([str(lit) for lit in self.state if lit.value])

        ret += "\n"
        ret += str(self.outcome)

        return ret

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.action == other.action and self.state == other.state and self.outcome == other.outcome

    def __hash__(self):
        return hash(self.__str__())


class ExampleSet:
    """A collection of examples"""
    def __init__(self):
        # Instead of storing each individual example, we can store them as keys in a dictionary with counts
        self.examples: Dict[Example, int] = dict()

    def add_example(self, example):
        # Don't store every example to save space. Use dict of counts. Only issue is we lose order
        # Perhaps could have a mapping of example to int so we can store ints instead of these?
        if example not in self.examples:
            self.examples[example] = 1
        else:
            self.examples[example] += 1

    def __str__(self):
        ret = ""

        for example, count in self.examples.items():
            ret += f"{example} ({count})\n"

        # Remove final \n
        return ret[:-1]

    def __repr__(self):
        return self.__str__()


class Rule:
    """A rule consists of a action, set of deictic references, context, and outcome set"""
    # Deictic references are a todo

    def __init__(self, action, context, outcomes):
        self.action: int = action
        self.context: List[Predicate] = context
        self.outcomes: OutcomeSet = outcomes

    def score(self, examples: ExampleSet) -> float:
        """
        Scores a rule on a set of examples. The score is the total likelihood - the penalty,
        where the penalty is the number of literals/effects in the outcomes and context of the rule
        This encourages simpler rules
        """
        alpha = 0.5  # Penalty multiplier. Notice num atts in outcomes and len(self.context) are treated equally
        penalty = alpha * (self.outcomes.get_total_num_affected_atts() + len(self.context))

        # Approximate noise probability, used for calculating likelihood
        p_min = 0.01

        # For example, if outcome1 predicts 2 examples with probability 0.25, and outcome2 predicts
        # six examples with probability 0.75, the likelihood is 0.25^2 * 0.75^6.
        # Log likelihood of this is 0.25 * 2 + 0.75 * 6

        log_likelihood = 0

        # # TODO:  Note, this is exactly the same computation done in learn_params. Should be someway to reuse that
        # # I would like to use examples_applicable_by_rule and num_examples_covered_by_outcome, but this
        # # creates cyclic imports. Perhaps this should be member functions instead of ones that take in args
        # applicable_examples = examples_applicable_by_rule(self, examples)
        #
        # for i, outcome in enumerate(self.outcomes.outcomes):
        #     num_covered = num_examples_covered_by_outcome(outcome, applicable_examples, examples)
        #
        #     log_likelihood += math.log10(self.outcomes.probabilities[i]) * num_covered

        return log_likelihood - penalty

    def __str__(self):
        ret = ""
        ret += f"Action {self.action}:\n"
        ret += f"{{{self.context}}}\n"
        ret += str(self.outcomes)
        return ret

    def __repr__(self):
        return self.__str__()


class RuleSet:
    """A collection of rules"""
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def score(self, examples: ExampleSet) -> float:
        """Returns the total score of this ruleset"""
        # Rules can be considered separately
        return sum(rule.score(examples) for rule in self.rules)

    def __str__(self):
        return "\n".join([str(rule) for rule in self.rules])

    def __repr__(self):
        return self.__str__()
