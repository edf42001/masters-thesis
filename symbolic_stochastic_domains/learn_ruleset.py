import copy

from symbolic_stochastic_domains.symbolic_classes import ExampleSet, RuleSet, Rule, OutcomeSet, Outcome, Example
from effects.effect import JointNoEffect
from symbolic_stochastic_domains.symbolic_utils import applicable


class ExplainExamples:
    """
    Quote, 'ExplainExamples takes as input a training set E and a rule set R and creates new,
    alternative rule sets that contain additional rules modeling the training examples
    that were covered by the default rule in R.'

    Creates rules for examples that aren't covered by other examples.
    """
    @staticmethod
    def execute(ruleset: RuleSet, examples: ExampleSet):
        # ExplainExamples(R, E)
        # Inputs:
        # A rule set R
        # A training set E
        # Computation:

        # For each example (s, a, s  ) ∈ E covered by the default rule in R

        # Note: for now, assume that we only run this at the start so this is all of the examples
        for example in examples.examples:
            # Step 1: Create a new rule r
            rule = Rule(-1, [], OutcomeSet())

            # Step 1.1: Create an action and context for r
            # Create new variables to represent the arguments of a
            # Use them to create a new action substitution σ
            # Set r’s action to be σ −1 (a)
            # Note: Our actions don't take parameters, so we don't need to do the above
            rule.action = example.action

            # Set r’s context to be the conjunction of boolean and equality literals that can
            # be formed using the variables and the available functions and predicates
            # (primitive and derived) and that are entailed by s
            # Note: I don't know what this means so, we are going to set it to the context (state) of the example
            rule.context = example.state

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
            rule.outcomes.add_outcome(example.outcome, 1.0)

            # Step 2: Trim literals from r
            # Create a rule set R  containing r and the default rule
            # new_ruleset = ruleset.add_rule(rule)
            # Greedily trim literals from r, ensuring that r still covers (s, a, s  ) and filling in the
            # outcomes using InduceOutcomes until R  ’s score stops improving
            # Note: In place trim the rule
            new_ruleset = copy.deepcopy(ruleset)
            new_ruleset.add_rule(rule)
            print(ruleset)
            print(new_ruleset)
            ExplainExamples.trim_rule(rule, example, ruleset, examples)

            # Step 3: Create a new rule set containing r
            # Create a new rule set R  = R
            # Add r to R  and remove any rules in R  that cover any examples r covers
            # Recompute the set of examples that the default rule in R  covers and the parameters
            # of this default rule
            # Add R  to the return rule sets R O
            # Output:
            # A set of rule sets, R O

            ruleset.add_rule(rule)

        return ruleset

    @staticmethod
    def trim_rule(rule: Rule, example: Example, rules: RuleSet, examples: ExampleSet):
        """
        Greedily removes literals from the rule's context while it is applicable to the example and the score improves
        """

        print(f"Trimming rule\n{rule}")

        best_score = rule.score(example)
        print(f"Current score: {best_score:0.3}")

        i = 0
        while i < len(rule.context):
            literal = rule.context[i]
            print(f"Trying to remove {literal}")
            del rule.context[i]

            if applicable(rule, example):
                score = rule.score(example)
                print(f"Current score: {score:0.3}")
                if score > best_score:
                    i = 0
                    best_score = score
                else:
                    print("Not better score")
                    i += 1
                    rule.context.insert(0, literal)
            else:
                i += 1
                rule.context.insert(0, literal)
                print("Not applicable")

        print("Final rule")
        print(rule)


def learn_ruleset(examples: ExampleSet) -> RuleSet:
    """Given a set of training examples, learns the optimal ruleset to explain them"""
    outcomes = OutcomeSet()
    outcomes.add_outcome(Outcome(JointNoEffect()), 1.0)
    # An action of -1 indicates default rule. Perhaps should have a specific subclass to represent it
    default_rule = Rule(action=-1, context=[], outcomes=outcomes)

    ruleset = RuleSet([default_rule])
    new_ruleset = ExplainExamples.execute(ruleset, examples)

    print("New ruleset")
    print(new_ruleset)
