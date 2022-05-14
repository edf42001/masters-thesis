from typing import List
from symbolic_stochastic_domains.utils import covers

from symbolic_stochastic_domains.learn_parameters import learn_params


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


def add(outcomes: List[dict]) -> List[List[dict]]:
    """
    Creates a new outcome by merging two non-contradictory outcomes
    Returns True if it found a pair to merge
    Modifies outcomes in-place
    """

    # Returns a list of list of outcomes. Basically, generates all new outcomes that can be made by merging
    new_outcomes = []

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

                # Make sure it isn't a duplicate. Such as adding H(c1) to H(c1), H(c2)
                # TODO: it still does duplicates like (H(c1), H(c2)) and (H(c2), H(c1))
                if addition != outcome and addition != outcome2:
                    new_set = outcomes.copy()
                    new_set.append(addition)

                    new_outcomes.append(new_set)

    return new_outcomes


def remove(examples, outcomes) -> List[List[dict]]:
    """
    Searches for an outcome it can remove from the set
    returns true if it found one to remove

    Removes in place
    """

    new_outcomes = []
    for i in range(len(outcomes)-1, -1, -1):
        if redundant(examples, outcomes, outcomes[i]):
            new_outcome = outcomes.copy()
            del new_outcome[i]
            new_outcomes.append(new_outcome)

    return new_outcomes


def outcomes_equal(outcomes1, outcomes2) -> bool:
    """Returns true if two sets of outcomes are the same"""

    # Check if every outcome pair in the two lists is the same
    # Assumes the same order
    for o1, o2 in zip(outcomes1, outcomes2):
        if o1 != o2:  # Dictionaries can be compared with just ==
            return False

    return True


def learn_outcomes(examples, outcomes: List[dict]):
    """
    Learns a minimal set of outcomes by combining and dropping outcomes until
    everything is explained as small as possible
    """

    # Step one: Use add operator to "pick a pair of non-contradictory outcomes in the set and create
    # a new outcome that is their conjunction"

    # Step 2:
    # drops an outcome from the set. Outcomes can only be dropped if they were overlapping
    # with other outcomes on every example they cover, otherwise the outcome set would not remain proper

    # To cover an example means if you apply the outcome it explains the example
    # i.e. if you apply heads(c1), heads(c2) to HT, you get HH, which explains a result of HH
    # To test for this, every change from s1 -> s2 must be listed in the outcome

    # An outcome is removed if every state change it covers is already covered by other things
    # For the no change case, find every example it covers. Then, see if it is covered by another outcome
    # If this is true for all, then it can be removed

    # Add combines non contradictory outcomes into a new one. For example, heads(c1) and tails(c1) contradict

    # Uses greedy search through search space

    print("Starting examples and outcomes")
    print(examples)
    print(outcomes)

    _, likelihood = learn_params(examples, outcomes)
    print("Starting likelihood {}".format(likelihood))
    print()

    best_likelihood = likelihood
    last_best_outcomes = None
    best_outcomes = outcomes

    # Loop until no change occurs
    while best_outcomes != last_best_outcomes:
        print("Starting with: {}".format(best_outcomes))
        last_best_outcomes = best_outcomes

        # Generate all possible next 1-distance instances and combine into list
        add_outcomes = add(best_outcomes)
        remove_outcomes = remove(examples, best_outcomes)

        new_outcome_list = add_outcomes
        new_outcome_list.extend(remove_outcomes)

        # Find which has the best likelihood
        for new_outcomes in new_outcome_list:
            _, likelihood = learn_params(examples, new_outcomes)

            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_outcomes = new_outcomes

            print("l: {:.4f}, {}".format(likelihood, new_outcomes))
        print()

    # At this point we will have greedy searched the best set of outcomes
    print("Result {}".format(best_outcomes))
    return best_outcomes


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
