"""
Created on 2/27/23 by Ethan Frank

Tests the logic-based object discovery agent with some objects not anonymize
"""

import datetime

import runners.object_transfer.taxi_object_transfer_runner
import runners.object_transfer.heist_object_transfer_runner
import runners.object_transfer.prison_object_transfer_runner


def main():

    taxi_known_objects = [None, ["wall"], ["pass"], ["dest"], ["wall", "pass", "dest"]]
    for known_objects in taxi_known_objects:
        print("Starting taxi transfer runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
        print(f"With known objects {known_objects}")
        runners.object_transfer.taxi_object_transfer_runner.main(known_objects=known_objects)

    heist_known_objects = [None, ["wall"], ["key"], ["gem"], ["lock"], ["wall", "key", "gem", "lock"]]
    for known_objects in heist_known_objects:
        print("Starting heist transfer runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
        print(f"With known objects {known_objects}")
        runners.object_transfer.heist_object_transfer_runner.main(known_objects=known_objects)

    prison_known_objects = [None, ["wall"], ["key"], ["gem"], ["lock"], ["pass"], ["dest"], ["wall", "key", "gem", "lock", "pass", "dest"]]
    for known_objects in prison_known_objects:
        print("Starting prison transfer runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
        print(f"With known objects {known_objects}")
        runners.object_transfer.prison_object_transfer_runner.main(known_objects=known_objects)


if __name__ == "__main__":
    main()
