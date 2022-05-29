from typing import List, Dict

from effects.effect import JointEffect
from environment.symbolic_door_world import Predicate


class Outcome:
    """An outcome is which attributes change and how"""
    def __init__(self, outcome):
        self.outcome: JointEffect = outcome

    def __str__(self):
        return str(self.outcome)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.outcome == other.outcome


class OutcomeSet:
    """A set of outcomes, each associated with a probability"""
    def __init__(self):
        self.outcomes: List[Outcome] = []
        self.probabilities: List[float] = []

    def __str__(self):
        ret = ""
        for p, outcome in zip(self.probabilities, self.outcomes):
            ret += f"{p:0.3}: {outcome}\n"
        return ret

    def __repr__(self):
        return self.__str__()


class Example:
    """
    An example consists of all the literals in the starting state, the action taken, and the outcomes
    """
    def __init__(self, action, state, outcome):
        self.action: int = action
        self.state: List[Predicate] = state
        self.outcome: Outcome = outcome

    def __str__(self):
        ret = ""

        ret += f"Action {self.action}:\n"

        # Only show true literals
        for lit in self.state:
            if lit.value:
                ret += str(lit) + ", "

        # Remove trailing comma
        ret = ret[:-2]

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

        return ret

    def __repr__(self):
        return self.__str__()