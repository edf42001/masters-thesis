from typing import Tuple, List
from symbolic_stochastic_domains.state import State


def covers(outcome: dict, example: Tuple[State, State]) -> bool:
    """
    Returns true if the outcome covers the example
    Meaning, when you apply the outcome to the initial state of the example,
    does the end state equal the final state?
    """
    s1 = example[0]
    s2 = example[1]

    # Go through every state variable. If they differ, see if the outcome would fix that
    # Otherwise, this outcome does not cover this
    for term1, term2 in zip(s1, s2):
        if term1.get_negated() != term2.get_negated():  # If conflict
            name = term2.get_unique_id()
            if name not in outcome or outcome[name] != term2.true():  # Does an outcome not cover this?
                return False

    # Also, need to check the outcome doesn't undo anything
    # Some redundant checks in here
    for term, truth in outcome.items():
        if term in s2 and s2[term] != truth:
            return False

    return True


def remove(examples, outcomes) -> bool:
    """
    Searches for an outcome it can remove from the set
    returns true if it found one to remove

    Removes in place
    """

    removed = False
    for i in range(len(outcomes)-1, -1, -1):
        if redundant(examples, outcomes, outcomes[i]):
            del outcomes[i]
            removed = True

    return removed


def redundant(examples, outcomes, outcome) -> bool:
    """
    Checks if an outcome is redundant. i.e., every example it covers is covered by other outcomes
    """
    covered_examples = []

    # First, find all the examples this outcome covers
    for example in examples:
        if covers(outcome, example):
            covered_examples.append(example)

    # Then, see if for every example, there is another that covers it
    for example in covered_examples:
        one_that_covers = False
        for outcome2 in outcomes:
            if outcome2 != outcome and covers(outcome2, example):
                one_that_covers = True

        if not one_that_covers:
            return False

    return True


def contradictory(outcome1, outcome2):
    """Outcomes are contradictory if any term in their intersection predicts opposite effects"""
    for term in outcome1:
        if term in outcome2 and outcome1[term] != outcome2[term]:
            return True

    return False


def add(outcomes: List[dict]) -> bool:
    """
    Creates a new outcome by merging two non-contradictory outcomes
    Returns True if it found a pair to merge
    Modifies outcomes in-place
    """

    for outcome in outcomes:
        # Look for a non-contradictory outcome
        for outcome2 in outcomes:
            if outcome != outcome2 and not contradictory(outcome, outcome2):
                # Merge them together, add to list, return true
                addition = outcome.copy()

                # Combine all the terms. This overwrites some with the same value but that is fine
                # Could check for non already there to save memory writes
                for term, truth in outcome2.items():
                    addition[term] = truth

                outcomes.append(addition)

                return True

    return False


def learn_outcomes(examples, outcomes):
    """
    Learns a minimal set of outcomes by combining and dropping outcomes until
    everything is explained as small as possible
    """

    # Step one: Use add operator to "pick a pair of non-contradictory outcomes in the set and create
    # a new outcome that is their conjunction"

    # Step 2:
    # drops an outcome from the set. Outcomes can only be dropped if they were overlapping
    # with other outcomes on every example they cover, otherwise the outcome set would not remain proper
    print(examples)
    print(outcomes)

    # To cover an example means if you apply the outcome it explains the example
    # i.e. if you apply heads(c1), heads(c2) to HT, you get HH, which explains a result of HH
    # To test for this, every change from s1 -> s2 must be listed in the outcome

    # An outcome is removed if every state change it covers is already covered by other things
    # For the no change case, find every example it covers. Then, see if it is covered by another outcome
    # If this is true for all, then it can be removed

    # Add combines non contradictory outcomes into a new one. For example, heads(c1) and tails(c1) contradict
    removed = True
    added = True

    i = 0
    while removed or added:
        print("Before removing")
        print(outcomes)
        # Remove any redundant outcomes, modifying outcomes list in place
        removed = remove(examples, outcomes)
        print("After removing")
        print(outcomes)
        print()

        print("Before Adding")
        print(outcomes)
        # Merge two outcomes, modify outcomes in-place
        added = add(outcomes)
        print("After Adding")
        print(outcomes)
        print()

        print(removed, added)
        i += 1

        if i == 3:
            break

    return outcomes


def test_covers(examples, outcomes):
    for outcome in outcomes:
        for example in examples:
            cover = covers(outcome, example)
            if cover:
                print("{} covers {}".format(outcome, example))
            else:
                print("{} does not cover {}".format(outcome, example))


def test_redundant(examples, outcomes):
    for outcome in outcomes:
        re = redundant(examples, outcomes, outcome)
        if re:
            print("{} is redundant".format(outcome))
        else:
            print("{} is not redundant".format(outcome))
