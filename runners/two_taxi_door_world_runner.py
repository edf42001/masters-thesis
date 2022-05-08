import logging
import random

from environment.door_world import DoorWorld

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="log.log", filemode='w')

    world = DoorWorld()
    world.visualize()

    for i in range(100):
        action = random.randint(0, 1)
        target = random.randint(0, 1)

        conditions = world.get_condition(world.get_state())

        observation = world.step(action=action, target=target)

        done = world.end_of_episode()
        reward = world.get_last_reward()
        print("Conditions {}".format("".join(["1" if c else "0" for c in conditions])))
        print("Taking action {} target {}. Done {} obs {} reward {}".format(action, target, done, observation, reward))
        world.visualize()

        if done:
            break


# def test_flat_state():
#     """Verify conversions to and from flat state"""
#     w = DoorWorld()
#
#     for x1 in range(11):
#         for x2 in range(11):
#             for open in range(2):
#                 factored_state = [x1, x2, open]
#                 flat_state = w.get_flat_state(factored_state)
#                 assert flat_state == (22 * x1 + 2 * x2 + 1 * open)
#                 reverse_factored_state = w.get_factored_state(flat_state)
#                 assert reverse_factored_state == factored_state
#
#
# test_flat_state()
