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

        # Keeps track of how many outcomes in the example set of the current session,
        # does the default rule apply to a noise or a no change outcome. Used for likelihood calculation
        self.default_rule_num_no_change = 0
        self.default_rule_num_noise = 0

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def __str__(self):
        return "\n".join([str(rule) for rule in self.rules])

    def __repr__(self):
        return self.__str__()
