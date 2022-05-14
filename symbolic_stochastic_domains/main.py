from symbolic_stochastic_domains.rule import Rule
from symbolic_stochastic_domains.term import Term
from symbolic_stochastic_domains.coin_world import CoinWorld
from symbolic_stochastic_domains.state import State
from symbolic_stochastic_domains.learn_outcomes import learn_outcomes, test_covers, test_redundant


def world_state_to_term_state(objects, term_names, state):
    s = State()
    terms = [Term(name=name, negated=not state[i], x=o) for name in term_names for i, o in enumerate(objects)]
    s.add_terms(terms)
    return s


if __name__ == "__main__":
    objects = ["coin1", "coin2"]
    term_names = ["heads"]

    terms = [Term(name=name, x=o) for name in term_names for o in objects]

    # # Evaluate some iterations of coin world
    # world = CoinWorld()
    # for i in range(5):
    #     world.reset()
    #     state = world.state()
    #     world.step()
    #     next_state = world.state()
    #
    #     s1 = world_state_to_term_state(objects, term_names, state)
    #     s2 = world_state_to_term_state(objects, term_names, next_state)
    #     print(state, next_state)
    #     print(s1, s2)
    #
    #     diff = State.get_changed_terms(s1, s2)
    #     print(diff)
    # print()

    # Setup an example experience to learn outcomes on
    experiences = [
        [[False, True], [True, True]],
        [[True, False], [True, True]],
        [[True, True], [False, False]],
        [[True, True], [True, True]],
    ]

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

    result = learn_outcomes(examples, outcomes)
    print(result)
    print()

    # print("Covers test:")
    # test_covers(examples, outcomes)
    # print()
    #
    # print("Redundant test:")
    # test_redundant(examples, outcomes)
    # print()
