from typing import List, Dict
import numpy as np

from symbolic_stochastic_domains.symbolic_classes import ExampleSet, RuleSet, Rule, OutcomeSet, Outcome, Example
from symbolic_stochastic_domains.symbolic_utils import context_matches, covers, applicable
from symbolic_stochastic_domains.learn_outcomes import learn_outcomes
from symbolic_stochastic_domains.predicate_tree import PredicateTree, Edge, Node
from symbolic_stochastic_domains.predicates_and_objects import TouchLeft2D, TouchRight2D, TouchUp2D, TouchDown2D, Open, On2D, In, PredicateType

from effects.effect import JointNoEffect

# Mapping from level to possible contexts, so we don't have to regenerate them each time
# What is the memory usage of this?
NEW_CONTEXTS = dict()  # Disable it for now


def create_new_contexts_from_context(context: PredicateTree) -> List[PredicateTree]:
    new_contexts = []

    p_types = [PredicateType.TOUCH_LEFT2D, PredicateType.TOUCH_RIGHT2D, PredicateType.TOUCH_DOWN2D,
               PredicateType.TOUCH_UP2D, PredicateType.ON2D, PredicateType.IN]
    object_names = ["key", "lock", "gem", "wall"]
    for p_type in p_types:
        for object_name in object_names:
            if not context.base_object.has_edge_with(p_type, object_name):  # Don't make duplicate edge (Hash table from CSDS 233)?
                copy1 = context.copy()
                copy2 = context.copy()

                # Need to test both positive and negative version of the literal
                copy1.base_object.add_edge(Edge(p_type, Node(object_name)))  # Feel like these should just be nodes
                copy2.base_object.add_negative_edge(Edge(p_type, Node(object_name)))

                new_contexts.append(copy1)
                new_contexts.append(copy2)

    return new_contexts


def only_applies_to_outcome(rule: Rule, examples: ExampleSet):
    """
    Checks that a rule only covers examples with the one outcome in the rule's outcome set
    For example, if we're trying to find a rule that says when the taxi moves down, we don't want to also
    cover examples where we don't move
    """
    outcome = rule.outcomes.outcomes[0]  # Only one outcome allowed
    for example in examples.examples.keys():
        # Is the outcome different? And if so, is this a rule we cover? If so, that is bad, return False
        if rule.action == example.action and example.outcome != outcome and context_matches(rule.context, example.state):
            return False

    return True


def applicable_by_outcome(rule: Rule, example: Example, outcome: Outcome):
    return (
            rule.action == example.action and
            context_matches(rule.context, example.state) and
            covers(outcome, example)
    )


def print_examples_rule_covers(rule: Rule, examples: ExampleSet):
    for outcome in rule.outcomes.outcomes:
        applicable = [example for example in examples.examples.keys() if applicable_by_outcome(rule, example, outcome)]
        print(f"Outcome: {outcome}: {applicable}")


# Perhaps I need to remake the rules but in this manner
# What is the best way to figure out the minimal set that covers the maximum examples?
# Structure learning, multiple instance learning?
def find_greedy_rule_by_adding_lits(examples: ExampleSet, relevant_examples: List[Example], irrelevant_examples: List[Example]):
    # In the case where only one action causes an effect, we can just get it from the relevant examples
    action = relevant_examples[0].action

    # FInd the list of objects referred to in the outcomes that we must have deictic references for
    # In order for an object ot be in outcomes, MUST be in either action or conditions
    objects_in_outcome = set([key.split(".")[0] for key in relevant_examples[0].outcome.outcome.value.keys()])

    # List of test contexts to try
    test_contexts = [PredicateTree()]

    level = 0  # Current level of # literals in context

    # For now, only try up to two literals in the context
    while level < 3:
        best_score = 1  # Start at 1 to fix a bug where it sometimes chose rules that covered 0 examples
        best_rule = None
        for context in test_contexts:  # Iterate over each context
            # Make a rule with that context
            outcomes = OutcomeSet()
            outcomes.add_outcome(relevant_examples[0].outcome, 1.0)
            rule = Rule(action=action, context=context, outcomes=outcomes)

            # Make sure we have a reference to each object, but ignore taxi for now, that is referenced in the action
            have_correct_deictic_references = all(
                (obj == "taxi" or obj in context.base_object.referenced_objects) for obj in objects_in_outcome
            )
            if (
                have_correct_deictic_references and
                only_applies_to_outcome(rule, examples) and not
                # Simply be removing the [] around the list comprehension, we turn it to a generator
                # That means instead of creating the whole list and then iterating through it to search for trues,
                # any() will now stop upon the first true it finds.
                any(applicable(rule, example) for example in irrelevant_examples)
            ):
                # Ok, looks like there are some issues with my method.
                # It's not just ~TouchDown(taxi, wall), because there's also the issue of when the taxi runs into
                # an open door. So ~touchDown covers most, but not all scenarios.
                # Yup, if you look at the ratios, ~TouchDown2D(taxi, wall) as 99% increment, 0.0099 no effect
                # If the rule is good, record its score. Score is number of examples in relevant set this applies to.
                score = sum([examples.examples[example] for example in relevant_examples if applicable(rule, example)])
                if score >= best_score:
                    best_score = score
                    best_rule = rule

        # If we found a valid rule, this will be the best, adding any more context will make it cover less
        if best_rule is not None:
            return best_rule

        # Otherwise, try adding every literal to every literal in the context and trying again
        # This process could produce duplicates. I.e, if you have 1 then add 2, or 2 then add 1.
        # The hashes will be different because the order was different
        level += 1

        # If we've already encountered this level before, just reread the saved contexts instead of recalculating them
        if level not in NEW_CONTEXTS:
            new_contexts = []
            for context in test_contexts:
                new_contexts.extend(create_new_contexts_from_context(context))
            NEW_CONTEXTS[level] = new_contexts
        else:
            new_contexts = NEW_CONTEXTS[level]

        test_contexts = new_contexts

    # If we get down here, we weren't able to do it with only two literals
    # TODO: for the case of Action 5: {[In(taxi, key), TouchDown2D(taxi, lock)]}, we should see if it is faster
    # to do additive or subtractive
    print("No rule found :(")
    # TODO: could try using subtraction method in this instance only
    return None


def find_greedy_rule_by_removing_lits(examples: ExampleSet, relevant_examples: List[Example], irrelevant_examples: List[Example]):
    # print("Finding rule by removing lits")

    new_rules = []  # Could probably remove these by keeping only the best rule, but what if there is a tie?
    scores = []

    # In the case where only one action causes an effect, we can just get it from the relevant examples
    action = relevant_examples[0].action

    # FInd the list of objects referred to in the outcomes that we must have deictic references for
    # In order for an object ot be in outcomes, MUST be in either action or conditions
    objects_in_outcome = set([key.split(".")[0] for key in relevant_examples[0].outcome.outcome.value.keys()])
    # print(f"Objects referenced in outcomes: {objects_in_outcome}")

    # Try to explain each example
    for example in relevant_examples:
        outcomes = OutcomeSet()
        outcomes.add_outcome(relevant_examples[0].outcome, 1.0)
        rule = Rule(action=action, context=[], outcomes=outcomes)
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

            # Ensure it only covers these outcomes, and not any in irrelevant examples
            # The second object is relavant, first is just taxi. Will need to find some other way to refer to taxi
            # TODO and think about: Does the predicate have to be true? i.e, the agent is "interacting" with the object?
            # Yup, this definetly seems to lead to better rules. For Action 4, it replaced:
            # [~TouchDown2D(taxi, wall), TouchUp2D(taxi, wall), ~In(taxi, key)] with [On2D(taxi, key)]
            objects_in_context = set([lit.object2 for lit in rule.context if lit.value])
            # print(f"Objects in context: {objects_in_context}")

            # Make sure we have a reference to each object, but ignore taxi for now, that is referenced in the action
            # technically, like MoveLeft(taxi1)
            have_correct_deictic_references = all([(obj == "taxi" or obj in objects_in_context) for obj in objects_in_outcome])

            # TODO: this is inefficient, let's give up on the first false positive, instead of calling learn_outcomes
            if (
                have_correct_deictic_references and
                only_applies_to_outcome(rule, examples) and not
                any([applicable(rule, example) for example in irrelevant_examples])
            ):
                i = 0
                # Score is number of examples in relevant set this applies to.
                score = sum([examples.examples[example] for example in relevant_examples if applicable(rule, example)])
                # print(f"Current score: {score}, best_score: {best_score}")
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

        # best_rule = find_greedy_rule_by_removing_lits(examples, relevant_examples, irrelevant_examples)
        best_rule = find_greedy_rule_by_adding_lits(examples, relevant_examples, irrelevant_examples)
        rules.append(best_rule)

        # Update relevant examples by removing ones the new rule didn't apply to, add those ones to irrelevant examples
        irrelevant_examples.extend([example for example in relevant_examples if applicable(best_rule, example)])
        relevant_examples = [example for example in relevant_examples if not applicable(best_rule, example)]
        # print("Remaining relevant examples")
        # for ex in relevant_examples:
        #     print(ex)
        # print()
        # print("irrelevant examples")
        # for ex in irrelevant_examples:
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

    # Ruleset for an example form a if (A ^ B) v (~C) v (D) structure where the rules are or'd.

    # First, get a list of every unique outcome
    unique_outcomes = []

    # Find all the unique outcomes that have been experienced. Exclude JointNoEffect, we will assume
    # anything not covered by the other rules leads to no effect
    for example in examples.examples.keys():
        if example.outcome in unique_outcomes or type(example.outcome.outcome) is JointNoEffect:
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
