from typing import List, Dict
import numpy as np

from symbolic_stochastic_domains.symbolic_classes import ExampleSet, RuleSet, Rule, OutcomeSet, Outcome, Example
from symbolic_stochastic_domains.symbolic_utils import context_matches, covers, applicable
from symbolic_stochastic_domains.learn_outcomes import learn_outcomes


def applicable_by_outcome(rule: Rule, example: Example, outcome: Outcome):
    return (
            rule.action == example.action and
            context_matches(rule.context, example.state_set) and
            covers(outcome, example)
    )


def print_examples_rule_covers(rule: Rule, examples: ExampleSet):
    for outcome in rule.outcomes.outcomes:
        applicable = [example for example in examples.examples.keys() if applicable_by_outcome(rule, example, outcome)]
        print(f"Outcome: {outcome}: {applicable}")


def find_greedy_rule_by_removing_lits(examples: ExampleSet, relevant_examples: List[Example], irrelevant_examples: List[Example]):
    valid_outcomes = OutcomeSet()
    valid_outcomes.add_outcome(relevant_examples[0].outcome, 1.0)

    # print("Finding rule by removing lits")

    new_rules = []
    scores = []

    # In the case where only one action causes an effect, we can just get it from the relavant examples
    action = relevant_examples[0].action

    # FInd the list of objects referred to in the outcomes that we must have deictic references for
    # In order for an object ot be in outcomes, MUST be in either action or conditions
    objects_in_outcome = set([key.split(".")[0] for key in relevant_examples[0].outcome.outcome.value.keys()])
    # print(f"Objects referenced in outcomes: {objects_in_outcome}")

    # Try to explain each example
    for example in relevant_examples:
        rule = Rule(action=action, context=[], outcomes=OutcomeSet())
        rule.context = [lit.copy() for lit in example.state]  # Start by fully describing the example
        # print(example)

        learn_outcomes(rule, examples)
        # print("Rule:")
        # print(rule)

        # Greedily try removing all literals until score doesn't improve
        i = 0
        best_score = 0
        while i < len(rule.context):
            literal = rule.context.pop(i)
            # print(f"Trying to remove {literal}")

            # Update outcomes for the new rule:
            learn_outcomes(rule, examples)

            # Ensure it only covers these outcomes, and not any in irrelevant examples
            # The second object is relavant, first is just taxi. Will need to find some other way to refer to taxi
            objects_in_context = set([pred.object2 for pred in rule.context])
            # print(f"Objects in context: {objects_in_context}")

            # Make sure we have a reference to each object, but ignore taxi for now, that is referenced in the action
            # technically, like MoveLeft(taxi1)
            have_correct_deictic_references = all([(obj == "taxi" or obj in objects_in_context) for obj in objects_in_outcome])

            # TODO: this is inefficient, let's give up on the first false positive, instead of calling learn_outcomes
            if (
                have_correct_deictic_references and
                rule.outcomes == valid_outcomes and not
                any([applicable(rule, example) for example in irrelevant_examples])
            ):
                i = 0
                # Score is number of examples in relevant set this applies to.
                score = sum([examples.examples[example] for example in relevant_examples if applicable(rule, example)])
                # print(f"Current score: {score}")
                if score >= best_score:
                    # TODO: Is this the most effecient way of calculating?
                    i = 0  # We actually want to keep going in this case, the ones that were bad are at the front
                    best_score = score
                else:
                    # print("Not better score")
                    i += 1
                    rule.context.insert(0, literal)
            else:
                i += 1
                rule.context.insert(0, literal)
                # print("Not applicable")

        learn_outcomes(rule, examples)
        # print("Final rule:")
        # print(rule)
        new_rules.append(rule)
        scores.append(best_score)

    # print("Final new rules:")
    # for rule, score in zip(new_rules, scores):
    #     print(rule, score)

    best_index = np.argmax(scores)
    return new_rules[best_index]


def learn_minimal_ruleset_for_outcome(examples: ExampleSet, outcome: Outcome) -> List[Rule]:
    """
    The goal is to explain every example in examples that has outcome of outcome,
    without overlapping any other examples
    """
    # print(f"Learning rules for outcome {outcome}")
    rules = []

    relevant_examples = [ex for ex in examples.examples.keys() if ex.outcome == outcome]
    irrelevant_examples = []

    while len(relevant_examples) > 0:
        # print("Relevant examples:")
        # for ex in relevant_examples:
        #     print(ex)
        # print()

        # Lets try greedily starting with empty context, then adding literals until we cover the most examples
        # Then repeat until we cover all the other examples
        # best_rule = find_optimal_greedy_rule(examples, relevant_examples)
        # rules.append(best_rule)

        best_rule = find_greedy_rule_by_removing_lits(examples, relevant_examples, irrelevant_examples)
        rules.append(best_rule)

        # Update relevant examples
        irrelevant_examples = [example for example in relevant_examples if applicable(best_rule, example)]
        relevant_examples = [example for example in relevant_examples if not applicable(best_rule, example)]
        # print("Remaining relevant examples")
        # for ex in relevant_examples:
        #     print(ex)
        # print()

        # Somehow we need to learn that the thing that explains this example is
        # TouchRight(taxi, door), Open(door, door), and
        # not any of the other things.

    # print("Final ruleset:")
    # for rule in rules:
    #     print(rule)

    return rules


def learn_ruleset_outcomes(examples: ExampleSet) -> RuleSet:
    """
    Given a set of training examples, learns the optimal ruleset to explain them
    For each outcome, finds the minimal set of rules that explains that and only that (for deterministic world)
    """

    # RUleset for an example form a if (A ^ B) v (~C) v (D) structure where the rules are or'd.

    # First, get a list of every unique outcome
    unique_outcomes = []

    for example in examples.examples.keys():
        if example.outcome in unique_outcomes:
            continue

        unique_outcomes.append(example.outcome)

    # print("Unique outcomes:")
    # print(unique_outcomes)

    # Learn the rules for each outcome
    rules = []
    for outcome in unique_outcomes:
        new_rules = learn_minimal_ruleset_for_outcome(examples, outcome)
        rules.extend(new_rules)

    # print()
    # print("Final final ruleset")
    # for rule in rules:
    #     print(rule)

    return RuleSet(rules)
