from typing import List

import numpy as np

from symbolic_stochastic_domains.predicate_tree import PredicateTree
from symbolic_stochastic_domains.predicates_and_objects import PredicateType
from symbolic_stochastic_domains.symbolic_classes import ExampleSet, RuleSet, Rule, OutcomeSet, Outcome, Example
from symbolic_stochastic_domains.symbolic_utils import context_matches, applicable

# Mapping from level to possible contexts, so we don't have to regenerate them each time
# What is the memory usage of this?
NEW_CONTEXTS = dict()  # Disable it for now


class RulesetLearner:
    def __init__(self, env):
        self.env = env

    def create_new_contexts_from_context(self, context: PredicateTree) -> List[PredicateTree]:
        new_contexts = []

        p_types = [PredicateType.TOUCH_LEFT, PredicateType.TOUCH_RIGHT, PredicateType.TOUCH_DOWN,
                   PredicateType.TOUCH_UP, PredicateType.ON, PredicateType.IN]

        # Get the list of objects in the environment from the env. Modify it: no taxi, add wall
        # This is because we need a list of all objects the taxi can interact with
        object_names = self.env.get_object_names()
        object_names.remove("taxi")

        for p_type in p_types:
            for object_name in object_names:
                if not context.base_object.has_edge_with(p_type, object_name):
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

                    if object_name == "lock":
                        copy3 = copy1.copy()
                        copy4 = copy1.copy()

                        copy3.add_property(object_name + str(identifier), PredicateType.OPEN, True)
                        copy4.add_property(object_name + str(identifier), PredicateType.OPEN, False)

                        new_contexts.append(copy3)
                        new_contexts.append(copy4)

        return new_contexts

    def initialize_deictic_rules(self, outcome: Outcome) -> List[PredicateTree]:
        """
        Because we know that any object mentioned in the outcome must have a rule in the tree,
        we can initialize the rule as such to save processing.
        """

        tree = PredicateTree()
        tree.add_node("taxi0")
        for reference in outcome.value.keys():
            if reference.edge_type is not None:  # Ones where only the taxi changed

                # Make sure we get a unique name for the object
                identifier = 0
                while (reference.to_ob + str(identifier)) in tree.node_lookup:
                    identifier += 1
                tree.add_node(reference.to_ob + str(identifier))
                tree.add_edge("taxi0", reference.to_ob + str(identifier), reference.edge_type)

        contexts = [tree]

        # Check for any locks to add properties to. We have to do this here, otherwise
        # it will not ever check both property rules for locks
        for edge in tree.base_object.edges:
            object_name = edge.to_node.object_name
            if object_name == "lock":
                copy1 = tree.copy()
                copy2 = tree.copy()

                copy1.add_property(edge.to_node.full_name(), PredicateType.OPEN, True)
                copy2.add_property(edge.to_node.full_name(), PredicateType.OPEN, False)

                contexts.append(copy1)
                contexts.append(copy2)

        return contexts

    def find_rule_by_first_order_inductive_logic(self, examples: ExampleSet, relevant_examples: List[Example], irrelevant_examples: List[Example]):
        # See https://www.geeksforgeeks.org/first-order-inductive-learner-foil-algorithm/
        # relevant_examples is our positive examples
        # irrelevant_examples is our negative examples
        # Both of these lists are constrained to examples that match the action

        # This information is static throughout the learning process
        action = relevant_examples[0].action
        outcomes = OutcomeSet()
        outcomes.add_outcome(relevant_examples[0].outcome, 1.0)


        # print("Relevent")
        # for example in relevant_examples:
        #     print(example)
        # print("Irrelevent")
        # for example in irrelevant_examples:
        #     print(example)
        # print()

        # In order for an object ot be in outcomes, MUST be in either action or conditions
        # Start with contexts that specifically mentions diectic references. Modify as needed
        new_contexts = self.initialize_deictic_rules(outcomes.outcomes[0])
        rule = Rule(action=action, context=new_contexts[0], outcomes=outcomes)
        new_rule_negatives = irrelevant_examples

        # Goal is learn the best rule that covers the most of the positive examples, and none of the negative examples
        # End when the rule covers no negatives
        while len(new_rule_negatives) > 0:

            # Pick the best one based on the FOIL gain metric
            # L is the candidate literal to add to rule R. p0 = number of positive bindings of R
            # n0 = number of negative bindings of R. p1 = number of positive binding of R + L
            # n1 = number of negative bindings of R + L. t  = number of positive bindings of R also covered by R + L
            p0 = sum([examples.examples[ex] for ex in relevant_examples if context_matches(rule.context, ex.state)])
            n0 = sum([examples.examples[ex] for ex in irrelevant_examples if context_matches(rule.context, ex.state)])

            # TODO: Store best in a list, or store top N in a heap, instead of storing all then sorting

            scores = []
            for context in new_contexts:
                # Find p1 and n1
                p1 = sum([examples.examples[ex] for ex in relevant_examples if context_matches(context, ex.state)])
                n1 = sum([examples.examples[ex] for ex in irrelevant_examples if context_matches(context, ex.state)])

                # t = number of positive bindings of R also covered by R + L.
                # t = sum([examples.examples[ex] for ex in relevant_examples if (context_matches(context, ex.state) and context_matches(rule.context, ex.state))])
                # I think this is always true, rule.context will match ex.state, that's why ex is in relevant examples
                # But I'm not sure about this, the examples are only there because the outcome matches
                # assert t == p1, f"{p1}, {n1}, {t}"  # A check to make sure
                t = p1

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

            # Extract the best score to look for ties
            best_score = scores[-1]
            best_contexts = []
            i = len(scores) - 1
            while scores[i] == best_score and i >= 0:
                best_contexts.append(new_contexts[i])
                i -= 1

            # In case of ties, find one that matches diectic references. Start with end of the list just in case
            best_context = new_contexts[-1]

            # print("Chose best context:")
            # print(best_context)

            # Create new rule from this best context
            rule = Rule(action=action, context=best_context, outcomes=outcomes)

            # Update the negative examples this still covers
            new_rule_negatives = [example for example in new_rule_negatives if context_matches(rule.context, example.state)]

            # If not done, add more candidate literals to the rule
            if len(new_rule_negatives) > 0:
                new_contexts = self.create_new_contexts_from_context(rule.context)

            # print("New rule negatives:")
            # for ex in new_rule_negatives:
            #     print(ex)

        # Now we found the best rule for this subset. In the outer loop, remove all positive examples and keep going
        # print("Done, new rule:")
        # print(rule)

        return rule

    def learn_minimal_ruleset_for_outcome(self, examples: ExampleSet, outcome: Outcome) -> List[Rule]:
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
            best_rule = self.find_rule_by_first_order_inductive_logic(examples, relevant_examples, irrelevant_examples)
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

    def learn_ruleset(self, examples: ExampleSet) -> RuleSet:
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
            if example.outcome in unique_outcomes or example.outcome.is_no_effect():
                continue

            unique_outcomes.append(example.outcome)

        # print("Unique outcomes:")
        # print(unique_outcomes)
        # unique_outcomes = [unique_outcomes[4]]  # test learning issues

        # Learn the rules for each outcome
        rules = []
        for outcome in unique_outcomes:
            new_rules = self.learn_minimal_ruleset_for_outcome(examples, outcome)
            rules.extend(new_rules)

        # print()
        # print("Final final ruleset")
        # for rule in rules:
        #     print(rule)

        return RuleSet(rules)
