import random

from environment.symbolic_door_world import SymbolicDoorWorld, TouchLeft, Taxi, Switch
from symbolic_stochastic_domains.symbolic_classes import Example, ExampleSet, Rule, OutcomeSet, Outcome
from symbolic_stochastic_domains.learn_ruleset import learn_ruleset

if __name__ == "__main__":
    random.seed(1)

    env = SymbolicDoorWorld(stochastic=False)
    example_set = ExampleSet()

    # Generate training examples
    for i in range(3):
        literals = env.get_literals(env.curr_state)
        action = random.randint(0, env.get_num_actions()-1)
        obs = env.step(action)
        outcome = Outcome(obs)
        example = Example(action, literals, outcome)
        example_set.add_example(example)

        # if env.end_of_episode(env.get_state()):
        #     env.restart()

    ruleset = learn_ruleset(example_set)
