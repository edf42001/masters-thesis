import logging

from environment.door_world import DoorWorld

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="log.log", filemode='w')

    world = DoorWorld()
    world.visualize()
    world.step(0)
    world.step(1)
    world.step(3)

