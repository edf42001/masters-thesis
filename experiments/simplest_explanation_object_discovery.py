"""
Created on 3/23/23 by Ethan Frank

Tests the simplest explanation object discovery agent
"""

import datetime

import runners.simplest_explanation.taxi_simplest_explanation_runner
import runners.simplest_explanation.heist_simplest_explanation_runner
import runners.object_transfer.prison_object_transfer_runner


def main():
    print("Starting taxi runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
    runners.simplest_explanation.taxi_simplest_explanation_runner.main()
    print("Starting heist runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
    runners.simplest_explanation.heist_simplest_explanation_runner.main()
    # print("Starting prison runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
    # runners.prison_runner.main()
    # print("Finished prison runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))

    # TODO: Create a version of prison runner that is just prison
    # Then verify that it works with both heist and taxi


if __name__ == "__main__":
    main()
