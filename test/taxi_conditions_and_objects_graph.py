import random
import graphviz
import time

from environment.taxi_world import TaxiWorld


def true_interactions(interaction):
    return [k for k, v in interaction.items() if v]


def visualize_interaction_graph(interactions: dict):
    graph = graphviz.Graph()

    # Taxi node
    graph.node("taxi", "taxi")

    # Object nodes
    for object, conditions in interactions.items():
        interaction = true_interactions(conditions)
        for i, condition in enumerate(interaction):
            graph.node(object + str(i), object)
            graph.edge("taxi", object + str(i), label=condition)

    graph.view()


if __name__ == "__main__":
    # For testing
    random.seed(1)

    env = TaxiWorld(stochastic=False, shuffle_actions=False)

    for i in range(4):
        state = env.get_state()
        interactions = env.get_interacting_objects(state)

        env.visualize()
        for object, conditions in interactions.items():
            print(f"{object}: {conditions}")
        print()
        print()

        visualize_interaction_graph(interactions)
        time.sleep(1.5)

        env.step(2)
        if i == 2:
            env.step(4)





