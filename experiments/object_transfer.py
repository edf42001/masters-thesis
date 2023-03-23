"""
Created on 2/27/23 by Ethan Frank

Tests the basic first-order object-oriented MDP learner
"""

import datetime

import runners.taxi_object_transfer_runner
import runners.heist_object_transfer_runner
import runners.prison_object_transfer_runner


def main():

    taxi_known_objects = [None, ["wall"], ["pass"], ["dest"]]
    for known_objects in taxi_known_objects:
        print("Starting taxi transfer runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
        print(f"With known objects {known_objects}")
        runners.taxi_object_transfer_runner.main(known_objects=known_objects)

    # ["None", "Wall", "Key", "Gem", "Lock"]
    heist_known_objects = [None, ["wall"], ["key"], ["gem"], ["lock"]]
    for known_objects in heist_known_objects:
        print("Starting heist transfer runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
        print(f"With known objects {known_objects}")
        runners.heist_object_transfer_runner.main(known_objects=known_objects)

    # ["None", "Wall", "Key", "Gem", "Lock", "Passenger", "Destination"]
    prison_known_objects = [None, ["wall"], ["key"], ["gem"], ["lock"], ["pass"], ["dest"]]
    for known_objects in prison_known_objects:
        print("Starting heist transfer runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
        print(f"With known objects {known_objects}")
        runners.prison_object_transfer_runner.main(known_objects=known_objects)


if __name__ == "__main__":
    main()
