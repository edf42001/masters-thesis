import copy
from typing import List, Dict
import numpy as np

from symbolic_stochastic_domains.symbolic_classes import ExampleSet, RuleSet, Rule, OutcomeSet, Outcome, Example
from effects.effect import JointNoEffect, NoiseEffect
from symbolic_stochastic_domains.symbolic_utils import applicable, examples_applicable_by_rule, ruleset_score
from symbolic_stochastic_domains.learn_outcomes import learn_outcomes


def remove_redundant_rules(ruleset: RuleSet, examples: ExampleSet, new_rule: Rule):
    """Removes rules from a ruleset that cover any example the new_rule covers"""
    print("Removing redundant rules")
    for i in range(len(ruleset.rules)-1, 0, -1):  # Start at the end, go down to i = 1 to ignore the default rule
        test_rule = ruleset.rules[i]

        # Search for any example this rule applies to and so does the other
        for example in examples.examples:
            if applicable(test_rule, example) and applicable(new_rule, example):
                # Remove this rule, and move to the next one
                print(f"Found redundant rule #{i}")
                ruleset.rules.pop(i)
                continue


def calculate_default_rule(ruleset: RuleSet, examples: ExampleSet):
    """Given the other rules in a ruleset, calculates the parameters of the default rule in-place"""
    # The default rule has no context, and two outcomes, no change or noise (anything goes)
    # The default rule has to cover everything that isn't covered by any other rule
    # The default rule is indicated by an action of -1.

    covered_examples: Dict[Example, bool] = dict()

    for rule in ruleset.rules:
        # Ignore default rule
        if rule.action == -1:
            continue

        # Use the dictionary to store which examples are covered and which are not
        applicable_examples = examples_applicable_by_rule(rule, examples)
        for example in applicable_examples:
            covered_examples[example] = True

    # Go through uncovered examples, see if they are noise or no change, and create the default rule
    no_change = 0
    noise = 0
    total = 0

    # Also, we update the list of covered examples in here
    ruleset.default_rule_covered_examples = []
    for example in examples.examples.keys():
        if example not in covered_examples:
            # Keep track of what the default rule covers
            ruleset.default_rule_covered_examples.append(example)

            count = examples.examples[example]  # Use dictionary to get total count of this example
            total += count

            if type(example.outcome.outcome) is JointNoEffect:
                no_change += count
            else:
                noise += count

    # Construct the outcome set of the new default rule
    outcomes = OutcomeSet()
    if noise != 0:
        outcomes.add_outcome(Outcome(NoiseEffect()), noise / total)
    if no_change != 0:
        outcomes.add_outcome(Outcome(JointNoEffect()), no_change / total)

    # Update these values in the ruleset for future use of calculating likelihood of default rule
    ruleset.default_rule_num_no_change = no_change
    ruleset.default_rule_num_noise = noise

    # Update the default rule in the ruleset
    default_rule = Rule(-1, [], outcomes)
    ruleset.rules[0] = default_rule


class ExplainExamples:
    """
    Quote, 'ExplainExamples takes as input a training set E and a rule set R and creates new,
    alternative rule sets that contain additional rules modeling the training examples
    that were covered by the default rule in R.'

    Creates rules for examples that aren't covered by other examples.
    """
    @staticmethod
    def execute(ruleset: RuleSet, examples: ExampleSet) -> List[RuleSet]:
        # ExplainExamples(R, E)
        # Inputs:
        # A rule set R
        # A training set E
        # Computation:

        # For each example (s, a, s  ) ∈ E covered by the default rule in R

        new_rulesets = []

        # Note: for now, assume that we only run this at the start so this is all of the examples
        for example in ruleset.default_rule_covered_examples:
            print(f"Explaining example:\n{example}")
            print()

            # Step 1: Create a new rule r
            new_rule = Rule(-1, [], OutcomeSet())

            # Step 1.1: Create an action and context for r
            # Create new variables to represent the arguments of a
            # Use them to create a new action substitution σ
            # Set r’s action to be σ −1 (a)
            # Note: Our actions don't take parameters, so we don't need to do the above
            # No need to make a copy here because this is an int
            new_rule.action = example.action

            # Set r’s context to be the conjunction of boolean and equality literals that can
            # be formed using the variables and the available functions and predicates
            # (primitive and derived) and that are entailed by s
            # Note: I don't know what this means so, we are going to set it to the context (state) of the example
            # Make sure to make a copy, or any literals removed will be reflected in the example
            new_rule.context = example.state.copy()

            # Step 1.2: Create deictic references for r
            # Collect the set of constants C whose properties changed from s to s  , but
            # which are not in σ
            # For each c ∈ C
            # Create a new variable v and extend σ to map v to c
            # Create ρ, the conjunction of literals containing v that can be formed using
            # the available variables, functions, and predicates, and that are entailed by s
            # Create deictic reference d with variable v and restriction σ −1 (ρ)
            # If d uniquely refers to c in s, add it to r
            # Note: Completely ignore this whole thing above, no deictic references for now
            pass

            # Step 1.3: Complete the rule
            # Call InduceOutcomes to create the rule’s outcomes.
            # Note: I don't know how this applies, because we are using outcomes instead of booleans. Just set this
            # as the only outcome for now. Will this change for stochastic? Because this rule is applicable to more than
            # one thing?

            # Edit the rule in place with the outcomes
            learn_outcomes(new_rule, examples)

            # Step 2: Trim literals from r
            # Create a rule set R  containing r and the default rule
            # Greedily trim literals from r, ensuring that r still covers (s, a, s  ) and filling in the
            # outcomes using InduceOutcomes until R  ’s score stops improving
            # Note: In place trim the rule
            # Note: Create default ruleset, calculate parameters of the default rule
            default_ruleset = RuleSet([Rule(action=-1, context=[], outcomes=OutcomeSet()), new_rule])
            calculate_default_rule(default_ruleset, examples)

            # print("Untrimmed new rule:")
            # print(new_rule)
            # print()
            ExplainExamples.trim_rule(new_rule, example, default_ruleset, examples)
            print()
            print("Trimmed ruleset:")
            print(default_ruleset)
            print()

            # Step 3: Create a new rule set containing r
            # Create a new rule set R  = R
            # Add r to R  and remove any rules in R  that cover any examples r covers
            # Recompute the set of examples that the default rule in R  covers and the parameters
            # of this default rule
            # Add R  to the return rule sets R O
            new_ruleset = copy.deepcopy(ruleset)
            print("To be inserted into ruleset:")
            print(new_ruleset)

            # Remove any rules in R that cover an example that r covers
            remove_redundant_rules(ruleset, examples, new_rule)

            # Insert the new rule
            new_ruleset.add_rule(new_rule)

            # Recompute the default examples
            calculate_default_rule(new_ruleset, examples)

            # Add to list
            new_rulesets.append(new_ruleset)

        # Output:
        # A set of rule sets, R O
        return new_rulesets

    @staticmethod
    def trim_rule(rule: Rule, example: Example, ruleset: RuleSet, examples: ExampleSet):
        """
        Greedily removes literals from the rule's context while it is applicable to the example and the score improves
        """

        print(f"Trimming rule")
        best_score = ruleset_score(ruleset, examples)
        # print(f"Current score: {best_score:0.3}")

        # Greedily try removing all literals until score doesn't improve
        i = 0
        while i < len(rule.context):
            literal = rule.context.pop(i)
            # print(f"Trying to remove {literal}")

            # Update outcomes for the new rule:
            learn_outcomes(rule, examples)

            # Because we modified the ruleset, need to update the default rule:
            calculate_default_rule(ruleset, examples)
            # print("Current ruleset")
            # print(ruleset)

            if applicable(rule, example):
                score = ruleset_score(ruleset, examples)
                # print(f"Current score: {score:0.3}")
                if score > best_score:
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

        # Need to recalculate the things in case we had reinserted at the end
        learn_outcomes(rule, examples)
        calculate_default_rule(ruleset, examples)
        print("Final rule")
        print(rule)


class DropRules:
    """
    DropRules cycles through all the rules in the current rule set, and removes each one
    in turn from the set. It returns a set of rule sets, each one missing a different rule.
    """
    @staticmethod
    def execute(ruleset: RuleSet, examples: ExampleSet) -> List[RuleSet]:
        new_rulesets = []

        # Remove every rule, but start at one to not remove the default rule
        for i in range(1, len(ruleset.rules)):
            new_ruleset = copy.deepcopy(ruleset)
            new_ruleset.rules.pop(i)

            # Now that we have modified the rule, update the default rule's params
            calculate_default_rule(new_ruleset, examples)

            new_rulesets.append(new_ruleset)

        return new_rulesets


class DropLits:
    """
    DropLits selects every rule r ∈ R n times, where n is the number of literals in the
    context of r; It then creates a new rule r  by removing that literal from r’s context;
    """
    @staticmethod
    def execute(ruleset: RuleSet, examples: ExampleSet) -> List[RuleSet]:
        new_rulesets = []

        # For every rule and every literal in that rule, try to drop it, then integrate the new rule
        for i, rule in enumerate(ruleset.rules):
            for j, literal in enumerate(rule.context):
                # Create copies of the new ruleset and the new rule
                new_ruleset = copy.deepcopy(ruleset)
                new_rule = copy.deepcopy(rule)

                # Remove the literal
                new_rule.context.pop(j)

                # Update the outcomes for this rule
                learn_outcomes(new_rule, examples)

                # We know it is more general so we can automatically remove the old rule,
                # before trying to remove other redundant rules
                new_ruleset.rules.pop(i)
                remove_redundant_rules(new_ruleset, examples, new_rule)
                # Now that done removing redundant rules, add the new rule (don't forget this step)
                new_ruleset.add_rule(new_rule)

                # Update the parameters of the default rule
                calculate_default_rule(new_ruleset, examples)

                # Add the rule to the list
                new_rulesets.append(new_ruleset)

        return new_rulesets


def learn_ruleset(examples: ExampleSet) -> RuleSet:
    """Given a set of training examples, learns the optimal ruleset to explain them"""

    # An action of -1 indicates default rule. Perhaps should have a specific subclass to represent it
    default_rule = Rule(action=-1, context=[], outcomes=OutcomeSet())

    print("Example set:")
    print(examples)
    print()

    ruleset = RuleSet([default_rule])

    # Update the starting statistics for this ruleset. We need to know that the default rule covers everything
    calculate_default_rule(ruleset, examples)

    # best_score = ruleset_score(ruleset, examples)  # todo, make this work? why doesn't it work
    best_score = -1E10

    # Break the while loop when no improvement has been made
    while True:
        # Generate all next rulesets
        new_rulesets = []
        print("Executing ExplainExamples")
        new_rulesets.extend(ExplainExamples.execute(ruleset, examples))
        print(f"Created {len(new_rulesets)} rules")
        print("Executing DropRules")
        new_rulesets.extend(DropRules.execute(ruleset, examples))
        print(f"Created {len(new_rulesets)} rules")
        print("Executing DropLits")
        new_rulesets.extend(DropLits.execute(ruleset, examples))
        print(f"Created {len(new_rulesets)} rules")

        scores = [ruleset_score(rules, examples) for rules in new_rulesets]

        # Be careful. If you use the same variable for loop iteration as to store the best ruleset, it overwrites it
        print("New rulesets:")
        for r in new_rulesets:
            print(r)
            print("-----")

        print(f"Ruleset scores {scores}")

        best = np.argmax(scores)
        new_score = scores[best]

        # We are done if the score does not improve
        if new_score <= best_score:
            print("New best score not bigger than best, done")
            break

        # Update best score
        best_score = new_score

        # Choose best ruleset greedily
        ruleset = new_rulesets[best]

        print("Best ruleset")
        print(ruleset)
        print("-----------------------------------------------")

    print(f"Final ruleset: score = {best_score:0.4}")
    print(ruleset)


