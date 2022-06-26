from typing import List, Dict, Tuple

from effects.effect import JointEffect
from symbolic_stochastic_domains.predicates_and_objects import Predicate
from symbolic_stochastic_domains.predicate_tree import PredicateTree


class Outcome:
    """An outcome is which attributes change and how"""
    def __init__(self, outcome: JointEffect):
        self.outcome = outcome

        # The "value" attribute is the dictionary of att/effect pairs
        self.num_affected_atts = len(self.outcome.value)

    def get_num_affected_atts(self):
        return self.num_affected_atts

    def copy(self):
        # Return a copy of this. TODO: Don't need to copy the join effect because we won't be editing it?
        return Outcome(self.outcome)

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

    def copy(self):
        ret = OutcomeSet()
        ret.outcomes = [outcome.copy() for outcome in self.outcomes]
        ret.probabilities = self.probabilities.copy()
        return ret

    def __str__(self):
        ret = ""
        for p, outcome in zip(self.probabilities, self.outcomes):
            ret += f"{p:0.3}: {outcome}\n"

        # Remove trailing newline
        return ret[:-1]

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        # TODO: Does this compare elements of the lists for equality, or just the lists themselves?
        # What about floating point comparisons for probabilities?
        return self.outcomes == other.outcomes and self.probabilities == other.probabilities


class Example:
    """
    An example consists of all the literals in the starting state, the action taken, and the outcomes
    """
    def __init__(self, action: int, state: PredicateTree, outcome: Outcome):
        self.action = action
        self.state = state
        self.outcome = outcome

        # Convert state to a set, so we can also have easy lookup with hashes
        # self.state_set = set(self.state)

        # Examples are used in a lookup dictionary so they need to be hashed, I believe each one has a unique string
        # so we can use that as the hash. This might not be the most efficient way, but make sure to calculate it
        # Only once here at the beginning.
        self.hash = hash(self.__str__())

    def copy(self):
        ret = Example(self.action, self.state.copy(), self.outcome.copy())
        # ret.hash = self.hash  # I don't think we need this because it is created upon init
        return ret

    def __str__(self):
        # Only show true literals in the example
        return f"Action {self.action}: {self.state} --> {self.outcome}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        # TODO: Need to rewrite equals
        return self.action == other.action and self.state == other.state and self.outcome == other.outcome

    def __hash__(self):
        return self.hash


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

    def copy(self):
        return {example.copy: count for example, count in self.examples.items()}

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

    def copy(self):
        return Rule(self.action, [literal.copy() for literal in self.context], self.outcomes.copy())

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
        self.default_rule_covered_examples: List[Example] = []

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def copy(self):
        # TODO? Do I need to copy the default rule stuff, I think not because it always gets filled in
        return RuleSet([rule.copy() for rule in self.rules])

    def __str__(self):
        return "\n".join([str(rule) for rule in self.rules])

    def __repr__(self):
        return self.__str__()
