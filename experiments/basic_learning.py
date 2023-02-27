"""
Created on 2/27/23 by Ethan Frank

Tests the basic first-order object-oriented MDP learner
"""

import datetime

import runners.taxi_runner
import runners.heist_runner
import runners.prison_runner


def main():
    print("Starting taxi runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
    runners.taxi_runner.main()
    print("Starting heist runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
    runners.heist_runner.main()
    print("Starting prison runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))
    runners.prison_runner.main()
    print("Finished prison runner at " + datetime.datetime.now().strftime("%a %b %d, %H:%M:%S"))


if __name__ == "__main__":
    main()
