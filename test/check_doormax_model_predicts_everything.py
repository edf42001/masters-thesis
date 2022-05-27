import random
import pickle

from environment.taxi_world import TaxiWorld
from algorithm.doormax.utils import boolean_arr_to_string


if __name__ == "__main__":
    stochastic = False

    # For testing
    random.seed(2)  # TODO: Set to 2 and watch it crash

    # Learning
    env = TaxiWorld(stochastic=stochastic, shuffle_actions=True)

    # Assume transition model is already known
    with open("../runners/taxi-doormax-model.pkl", 'rb') as f:
        doormax_model = pickle.load(f)

    conditions = []
    for state in range(env.get_num_states()):
        for action in range(env.get_num_actions()):
            pred = doormax_model.compute_possible_transitions(state, action)

            if pred is None:
                factored_state = env.get_factored_state(state)
                condition = boolean_arr_to_string(env.get_condition(state))
                conditions.append(condition)
                print(f"None prediction: {factored_state}, {condition}, {action}")

    conditions = list(set(conditions))
    print(conditions)