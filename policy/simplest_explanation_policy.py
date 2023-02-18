"""
Created on 2/5/23 by Ethan Frank

Implements simplest explanation rule learning object transfer
"""

import random
from collections import deque

from policy.policy import Policy
from symbolic_stochastic_domains.object_transfer import information_gain_of_action
from algorithm.symbolic_domains.simplest_explanation_model import SimplestExplanationModel
from symbolic_stochastic_domains.experience_helper import ExperienceHelper


class SimplestExplanationPolicy(Policy):
    def __init__(self, actions: int, model: SimplestExplanationModel):
        self.num_actions = actions

        self.model = model

        # Used to transfer information from the closest path to a new experience from the breadth first function
        self.path = []

        self.last_object_map = {}

        self.in_failure_speedup_mode_hack = False

        # Alternate paths if no reward can be found
        self.path_to_information_gain = []
        self.path_to_experience_1 = []
        self.path_to_experience_2 = []

    def choose_action(self, curr_state: int, is_learning: bool = True) -> int:
        # If an object map change occurred, we need to replan

        # if we have a path, execute it, otherwise, plan a new one (this is just a testing hack, needs to take
        # into account if the model changes)
        if len(self.path) > 0:
            action = self.path[-1]
            print(f"Continuing with action {action}, {self.path}")
            del self.path[-1]
            return action

        # Get new path (to reward). First, reset all other paths
        self.path_to_information_gain = []
        self.path_to_experience_1 = []
        self.path_to_experience_2 = []
        reward_path = self.combined_search(curr_state)

        print(reward_path)
        print(self.path_to_information_gain)
        print(self.path_to_experience_1)
        print(self.path_to_experience_2)

        if len(reward_path) > 0:
            self.path = reward_path
            action = self.path[-1]
            print(f"Found path to reward, Taking action {action}, {self.path}")
            del self.path[-1]
            return action

        if len(self.path_to_information_gain) > 0:
            self.path = self.path_to_information_gain
            action = self.path[-1]
            print(f"Found path to info gain, Taking action {action}, {self.path}")
            del self.path[-1]
            return action

        if len(self.path_to_experience_1) > 0:
            self.path = self.path_to_experience_1
            action = self.path[-1]
            print(f"Found path to experience 1, Taking action {action}, {self.path}")
            del self.path[-1]
            return action

        if len(self.path_to_experience_1) > 0:
            self.path = self.path_to_experience_2
            action = self.path[-1]
            print(f"Found path to experience 2, Taking action {action}, {self.path}")
            del self.path[-1]
            return action

        action = random.randint(0, self.num_actions-1)
        print(f"Couldn't find path, taking random action {action}")

        return action

    def combined_search(self, state: int):
        """Simultaneously searches for a goal, info gain, and new experience"""

        # List of visited states
        visited = set()

        # Queue to store in progress states
        q = deque()

        # Dictionary of state: [it's parent, the action that led to it]
        parents = {}

        # Initialize search
        q.append(state)
        parents[state] = [-1, -1]

        # print("Beginning breadth first search")
        while len(q) != 0:
            curr_state = q.popleft()
            visited.add(curr_state)
            literals, instance_name_map = self.model.env.get_literals(curr_state)

            # print(f"Popped state {curr_state}: {self.model.env.get_factored_state(curr_state)}")

            # Generate all next states
            # I choose to do multiple small for loops rather than one big one, because I like how it organizes
            # It doesn't interrupt immediately now if it finds an action though
            next_states = [-1] * self.num_actions

            # An issue in my code is the agent can't represent two states where it picks up an object,
            # and that object could be a gem or a key, it only is able to generate the real mapping
            # So it cheats by looking at the reward function. This hides it from it.
            too_many_transitions = [False] * self.num_actions
            for action in range(self.num_actions):
                # Compute the next state, or don't do anything if we don't know. Pass literals for efficiency
                transitions = self.model.compute_possible_transitions(
                    curr_state, action, literals=literals, instance_name_map=instance_name_map
                )

                if len(transitions) == 0:
                    # We don't know what will happen, skip, set next_state to -1
                    next_states[action] = -1
                    continue

                # TODO: need to figure this out
                # assert len(transitions) < 2, "Only 1 transition allowed per state (for now)"
                if len(transitions) > 1:
                    too_many_transitions[action] = True

                effect = transitions[0].effect  # Assume only one effect, extract it from the transition
                next_state = self.model.next_state(curr_state, effect)

                next_states[action] = next_state

            for action in range(self.num_actions):
                next_state = next_states[action]

                # Add next states to the queue and update parents
                if next_state == -1 or next_state in visited or next_state in q:
                    continue

                # Update parents for this state, and add to queue
                parents[next_state] = [curr_state, action]
                q.append(next_state)

            # Search for reward state, return if found
            for action in range(self.num_actions):
                next_state = next_states[action]
                if next_state != -1 and not too_many_transitions[action] and self.model.get_reward(curr_state, next_state, action) > 0:
                    path = self.get_path(next_state, parents)
                    print("Found path to reward")
                    return path

            # Search for info gain if path not already found
            if len(self.path_to_information_gain) == 0:
                for action in range(self.num_actions):
                    if information_gain_of_action(self.model.env, curr_state, action, self.model.object_map,
                                                  self.model.previous_ruleset, remove_duplicates=False) > 0:
                        path = self.get_path(curr_state, parents)
                        path.insert(0, action)
                        print("Found path to information gain")
                        self.path_to_information_gain = path
                        break

            # Search for new experiences. TODO Do not return if found, simply store in a path for later use
            if len(self.path_to_experience_1) == 0:
                for action in range(self.num_actions):
                    # Check if we haven't tried this action yet with the current combination of objects
                    experiences = self.model.new_experience_helper.experiences[0]  # Level 1 experience
                    for experience in ExperienceHelper.extract_experiences(literals, n=1):
                        if experience not in experiences or action not in experiences[experience]:
                            print(f"Found an experience: {experience}, {action}")
                            # TODO: currently it is taking the first it sees in the state
                            path = self.get_path(curr_state, parents)
                            path.insert(0, action)
                            self.path_to_experience_1 = path
                            break

            # Only if we couldn't find an experience 1
            if len(self.path_to_experience_1) == 0 and len(self.path_to_experience_2) == 0:
                for action in range(self.num_actions):
                    # Also record level 2 experiences. These will be tried if no state has a level 1 experience
                    experiences = self.model.new_experience_helper.experiences[1]  # Level 2 experience
                    for experience in ExperienceHelper.extract_experiences(literals, n=2):
                        if experience not in experiences or action not in experiences[experience]:
                            print(f"Found an experience 2: {experience}, {action}")
                            path = self.get_path(curr_state, parents)
                            path.insert(0, action)
                            self.path_to_experience_2 = path
                            break

        # If we get down here, there is no path, so return empty
        return []

    def get_path(self, parent, parents):
        path = []
        while parents[parent][0] != -1:
            path.append(parents[parent][1])
            parent = parents[parent][0]

        return path
