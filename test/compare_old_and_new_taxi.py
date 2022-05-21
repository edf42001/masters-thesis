import random

from envs.taxi_world.doormax_taxi import DoormaxTaxi
from envs.taxi_world.taxi_world_env import TaxiWorldEnv, ACTION
from envs.taxi_world.doormax_helpers import boolean_arr_to_string

from environment.taxi_world import TaxiWorld
from algorithm.doormax.doormax_ruleset import DoormaxRuleset
from algorithm.doormax.doormax_simulator import DoormaxSimulator
from policy.value_iteration import ValueIteration


if __name__ == "__main__":
    # Old taxi:
    # Create the env
    old_env = TaxiWorldEnv()
    old_doormax = DoormaxTaxi(old_env)

    # New taxi
    params = {
        'discount_factor': 0.987  # Cannot be 1 for Rmax (why?)
    }

    new_env = TaxiWorld(stochastic=False)
    model = DoormaxRuleset(new_env)
    planner = ValueIteration(new_env.get_num_states(), new_env.get_num_actions(),
                             params['discount_factor'], new_env.get_rmax(), model)
    new_learner = DoormaxSimulator(new_env, model, planner, visualize=False)

    # How many iterations to iterate for, and iteration counter
    NUM_ITERATIONS = 40

    # Just in case random seed
    random.seed(0)

    # Main learning loop
    for i in range(NUM_ITERATIONS):
        state = old_env.get_state()
        state2 = new_env.get_factored_state(new_env.get_state())

        print("Choosing actions")
        print(f"Condition: {boolean_arr_to_string(new_env.get_condition(state2))}")
        action = old_doormax.select_action(state, discount_rate=0.987)
        action2 = planner.choose_action(new_env.get_state(), is_learning=True)

        if action != ACTION(action2):
            print("actions did not match " + str(i))
            break

        # Step env
        reward, done = old_env.step(action)
        obs = new_env.step(action.value)

        # reward, done = old_env.step(ACTION(action2))
        # obs = new_env.step(action2)

        # Step 3: observe new state s'
        new_state = old_env.get_state()
        new_state2 = new_env.get_factored_state(new_env.get_state())
        reward2 = new_env.get_last_reward()
        done2 = new_env.end_of_episode(new_env.get_state())

        print(action, reward, done, state, new_state)
        print(ACTION(action2), reward2, done2, state2, new_state2)
        print(f"Observations {obs}")

        # Step 4: Update model with addExperience(s, a, s', k)
        old_doormax.add_experience(state, action, new_state, k=1)
        old_doormax.rewards[old_doormax.env.state_hash(state), action.value] = reward

        model.add_experience(action.value, new_env.get_flat_state(state2), obs)

        # print("Predictions1")
        # old_doormax.print_predictions(old_doormax.predictions)
        # print("Predictions2")
        # model.print_model()

        if done2:
            print("Iteration " + str(i))
            new_env.restart()

        # new_env.visualize()

        print()
        print()

    print("RESULTS:")
    old_doormax.print_predictions(old_doormax.predictions)
    print("RESULTS2:")
    model.print_model()
