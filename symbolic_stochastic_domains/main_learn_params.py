from symbolic_stochastic_domains.term import Term
from symbolic_stochastic_domains.state import State
from symbolic_stochastic_domains.learn_parameters import learn_params

def world_state_to_term_state(objects, term_names, state):
    s = State()
    terms = [Term(name=name, negated=not state[i], x=o) for name in term_names for i, o in enumerate(objects)]
    s.add_terms(terms)
    return s


def booleans_to_examples_outcomes(experiences):
    # Convert these boolean lists to lists of States, before and after
    examples = []
    for experience in experiences:
        s1 = experience[0]
        s2 = experience[1]

        s1 = world_state_to_term_state(objects, term_names, s1)
        s2 = world_state_to_term_state(objects, term_names, s2)

        examples.append([s1, s2])

    # Next, generate an outcome set
    outcomes = []

    for example in examples:
        outcomes.append(State.get_changed_terms(example[0], example[1]))

    return examples, outcomes


if __name__ == "__main__":
    objects = ["coin1", "coin2"]
    term_names = ["heads"]

    # Setup an example experience to learn outcomes on
    experiences = [
        [[False, True], [True, True]],
        [[True, False], [True, True]],
        [[True, True], [False, False]],
        [[True, True], [True, True]],
    ]

    # What is likelihood of default set?
    examples, outcomes = booleans_to_examples_outcomes(experiences)

    params = learn_params(examples, outcomes)
    print("Params from initial outcome set: " + str(params))
    print()

    # Again, but with the optimal (and correct) set of outcomes
    outcomes = [{'heads(coin1)': True, 'heads(coin2)': True}, {'heads(coin1)': False, 'heads(coin2)': False}]

    params = learn_params(examples, outcomes)
    print("Params from optimal set: " + str(params))
    print()