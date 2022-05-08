import logging
import random
import numpy as np

from environment.door_world import DoorWorld

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="log.log", filemode='w',
                        format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")

    world = DoorWorld()
    logging.info(world.visualize())

    num_steps = 100

    # Store state, conditions, action, and next state
    step_data = np.empty((num_steps, 1 + 9 + 1 + 1), dtype=int)

    for i in range(num_steps):
        action = random.randint(0, 1)

        state = world.get_state()
        conditions = world.get_condition(state)

        observation = world.step(action=action)

        done = world.end_of_episode()
        reward = world.get_last_reward()
        next_state = world.get_state()

        logging.info("Conditions {}".format("".join(["1" if c else "0" for c in conditions])))
        logging.info("Taking action {}. Done {} obs {} reward {}".format(action, done, observation, reward))
        logging.info(world.visualize())

        step_data[i, 0] = state
        step_data[i, 1:10] = conditions
        step_data[i, 10] = action
        step_data[i, 11] = next_state

        if done:
            pass

        if (i+1) % 50 == 0:
            logging.warning("Restarting world")
            world.restart()
            # break

    print(step_data)
    np.savetxt("step_data.csv", step_data, delimiter=",", fmt="%d")


def test_flat_state():
    """Verify conversions to and from flat state"""
    w = DoorWorld()

    for x1 in range(8):
        for open in range(2):
            factored_state = [x1, open]
            flat_state = w.get_flat_state(factored_state)
            assert flat_state == (2 * x1 + 1 * open)
            reverse_factored_state = w.get_factored_state(flat_state)
            assert reverse_factored_state == factored_state


test_flat_state()
