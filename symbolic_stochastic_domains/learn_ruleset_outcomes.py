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
                # TODO: Should the tree handle the node ids? Or can they all be 0? No, because there is on key and in key
                # Thus, the tree should handle the node ids. FOr now lets try making them all 0?
                # Weird hack to get the correct unique id to work. Probably could store this somewhere
                identifier = 0
                while (object_name + str(identifier)) in context.node_lookup:
                    identifier += 1
                copy1.add_node(object_name + str(identifier))
                copy1.add_edge("taxi0", object_name + str(identifier), p_type)
                copy2.add_node(object_name + str(identifier))
                copy2.add_edge("taxi0", object_name + str(identifier), p_type, negative=True)

                new_contexts.append(copy1)
                new_contexts.append(copy2)

    # Manually handle the addition of variables for locks being open. In the future, this should be done automatically
    len_contexts = len(new_contexts)  # Store length since we will be appending
    # for i in range(len_contexts):
    # Check if the context has a lock being mentioned
    for e, edge in enumerate(context.base_object.edges):
        if edge.to_node.object_name[:-1] == "lock":
            copy1 = context.copy()
            copy2 = context.copy()

            copy1.add_property(edge.to_node.object_name, PredicateType.OPEN, True)
            copy2.add_property(edge.to_node.object_name, PredicateType.OPEN, False)

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


def find_rule_by_first_order_inductive_logic(examples: ExampleSet, relevant_examples: List[Example], irrelevant_examples: List[Example]):
    # See https://www.geeksforgeeks.org/first-order-inductive-learner-foil-algorithm/
    # relevant_examples is our positive examples
    # irrelevant_examples is our negative examples
    # Both of these lists are constrained to examples that match the action

    # This information is static throughout the learning process
    action = relevant_examples[0].action
    outcomes = OutcomeSet()
    outcomes.add_outcome(relevant_examples[0].outcome, 1.0)

    # Find the list of objects referred to in the outcomes that we must have deictic references for
    # In order for an object ot be in outcomes, MUST be in either action or conditions
    # Due to how we now implement outcomes with the digit appended to them, we need to remove that
    # The referenced object is the one at the end of the chain
    objects_in_outcome = set([key.split(".")[0] for key in relevant_examples[0].outcome.outcome.value.keys()])

    # print("Relevent")
    # for example in relevant_examples:
    #     print(example)
    # print("Irrelevent")
    # for example in irrelevant_examples:
    #     print(example)
    # print()

    # Initial rule with empty context. Initialize how many negative examples it covers
    tree = PredicateTree()
    tree.add_node("taxi0")
    contexts = [tree]
    rule = Rule(action=action, context=contexts[0], outcomes=outcomes)
    new_rule_negatives = [example for example in irrelevant_examples if context_matches(rule.context, example.state)]

    # Wait, could I replace rule.context with just context?

    # Our goal is learn the best rule that covers the most of the positive examples, and none of the negative examples
    lit_counter = 1  # How many lits have been addded, used for checking when deictic references should be used
    # End when deictic references are met and the rule covers no negatives
    while len(new_rule_negatives) > 0 or not \
          all((obj == "taxi" or obj in rule.context.referenced_objects) for obj in objects_in_outcome):
        # Generate all candidate literals to add to the new rule
        new_contexts = create_new_contexts_from_context(rule.context)

        # Pick the best one based on the FOIL gain metric
        # L is the candidate literal to add to rule R. p0 = number of positive bindings of R
        # n0 = number of negative bindings of R. p1 = number of positive binding of R + L
        # n1 = number of negative bindings of R + L. t  = number of positive bindings of R also covered by R + L
        p0 = sum([examples.examples[ex] for ex in relevant_examples if context_matches(rule.context, ex.state)])
        n0 = sum([examples.examples[ex] for ex in irrelevant_examples if context_matches(rule.context, ex.state)])

        # TODO: Store best in a list, or store top N in a heap, instead of storing all then sorting

        scores = []
        for context in new_contexts:
            # Verify we have all the correct deictic references in this context.
            # We assume there is no way to get a deictic reference later? Taxi is referenced in the action
            # TODO: Needs to be the `exact` object referenced in the outcome, i.e. if there is a lock being unlocked
            # below us, on lock isn't the same lock. Whoops, this fails when there are two objects mentioned, because
            # we can't reference both of them at once. Perhaps only activate this when the number of literals is
            # equal to the number of references? I don't know if this is a hack or valid.
            # Could say, the rule needs to reference at least one of them? That would also speed it up
            have_correct_deictic_references = all(
                (obj == "taxi" or obj in context.referenced_objects) for obj in objects_in_outcome
            )

            if lit_counter >= len(objects_in_outcome) and not have_correct_deictic_references:
                # print(f"Did not have references: {objects_in_outcome}, {context.referenced_objects}")
                scores.append(-10)  # Is negative 10 small enough?
                continue

            p1 = sum([examples.examples[ex] for ex in relevant_examples if context_matches(context, ex.state)])
            n1 = sum([examples.examples[ex] for ex in irrelevant_examples if context_matches(context, ex.state)])

            # t = number of positive bindings of R also covered by R + L. TODO: how to reduce duplicate calculations
            t = sum([examples.examples[ex] for ex in relevant_examples if context_matches(context, ex.state) and context_matches(rule.context, ex.state)])

            if p1 != 0:  # If the new rule covers no examples that leads to invalid value in log2
                gain = t * (np.log2(p1 / (p1 + n1)) - np.log2(p0 / (p0 + n0)))
            else:
                gain = -10
            # print(f"{context}, {p0}, {n0}, {p1}, {n1}, {t} {gain:.4f}")

            scores.append(gain)

        idxs = np.argsort(scores)
        scores = [scores[i] for i in idxs]
        new_contexts = [new_contexts[i] for i in idxs]
        # print()
        # for score, context in zip(scores[-5:], new_contexts[-5:]):
        #     print(f"{context}: {score:.4f}")

        best_context = new_contexts[-1]
        # print("Chose best context:")
        # print(best_context)

        # Create new rule from this best context
        rule = Rule(action=action, context=best_context, outcomes=outcomes)

        # Update the negative examples this still covers
        new_rule_negatives = [example for example in new_rule_negatives if context_matches(rule.context, example.state)]

        # We've added another literal
        lit_counter += 1

        # print("New rule negatives:")
        # for ex in new_rule_negatives:
        #     print(ex)

    # Now we found the best rule for this subset. In the outer loop, remove all positive examples and keep going
    # print("Done, new rule:")
    # print(rule)

    return rule


def learn_minimal_ruleset_for_outcome(examples: ExampleSet, outcome: Outcome) -> List[Rule]:
    """
    The goal is to explain every example in examples that has outcome of outcome,
    without overlapping any other examples
    """
    # print(f"Learning rules for outcome {outcome}")
    rules = []

    relevant_examples = [ex for ex in examples.examples.keys() if ex.outcome == outcome]
    action = relevant_examples[0].action

    # All instances with the same action, but a different outcome
    irrelevant_examples = [ex for ex in examples.examples.keys() if ex.action == action and ex.outcome != outcome]

    while len(relevant_examples) > 0:
        # print("Relevant examples:")
        # for ex in relevant_examples:
        #     print(ex)
        # print()

        # Lets try greedily starting with empty context, then adding literals until we cover the most examples
        # Then repeat until we cover all the other examples
        # best_rule = find_optimal_greedy_rule(examples, relevant_examples)
        # rules.append(best_rule)

        # For rules, can have decitic reference checks combining tree and outcome.
        best_rule = find_rule_by_first_order_inductive_logic(examples, relevant_examples, irrelevant_examples)
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
    # unique_outcomes = [unique_outcomes[4]]  # test learning issues
    # unique_outcomes = [unique_outcomes[1]]  # test learning issues
    # unique_outcomes = [unique_outcomes[0]]  # test learning issues
    # unique_outcomes = [unique_outcomes[5]]  # test learning issues

    # del unique_outcomes[1]

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