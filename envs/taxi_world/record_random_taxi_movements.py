from envs.taxi_world.taxi_world_env import TaxiWorldEnv, ACTION
from envs.taxi_world.doormax_taxi import DoormaxTaxi

import numpy as np

if __name__ == "__main__":
    env = TaxiWorldEnv(stochastic=True)
    doormax = DoormaxTaxi(env)

    action_map = list(ACTION)

    NUM_STEPS = 10000
    iterations = 0

    movement_data = np.empty((NUM_STEPS, 11))

    while iterations < NUM_STEPS:

        # env.draw_taxi()
        state = env.get_state()
        action = ACTION(np.random.randint(0, 4))
        reward, done = env.step(action)
        next_state = env.get_state()

        # Determine the effect
        dx = next_state[0] - state[0]
        dy = next_state[1] - state[1]

        effect = 0
        if dy == 1:
            effect = 0
        elif dx == 1:
            effect = 1
        elif dy == -1:
            effect = 2
        elif dx == -1:
            effect = 3
        else:
            effect = 4

        # Save conditions, action, and effect
        movement_data[iterations, 0] = action.value
        movement_data[iterations, 1:8] = [1 if letter == "1" else 0 for letter in doormax.conditions(state)]

        # Record dx, dy, and a 0-4 indexed direction effect for fun
        movement_data[iterations, 8] = dx
        movement_data[iterations, 9] = dy
        movement_data[iterations, 10] = effect

        print("Reward: {}".format(reward))
        print("pickup: {}, dropoff: {}".format(env.current_pickup, env.current_dropoff))
        print("x: {}, y: {}".format(env.taxi.x, env.taxi.y))
        print("passenger_in_taxi: {}".format(env.passenger_in_taxi()))

        iterations += 1

    # Save the data to csv
    print(movement_data)
    np.savetxt("movement_data.csv", movement_data, delimiter=",", fmt="%d")
