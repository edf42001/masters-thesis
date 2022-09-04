import random
from collections import deque

from algorithm.transition_model import TransitionModel
from policy.policy import Policy
from test.object_transfer.test_object_transfer_exploration import information_gain_of_action


class ObjectTransferPolicy(Policy):
    def __init__(self, actions: int, model: TransitionModel):
        self.num_actions = actions

        self.model = model

        # Used to transfer information from the closest path to a new experience from the breadth first function
        self.path_to_experience = None

    def choose_action(self, curr_state: int, is_learning: bool = True) -> int:
        # For now, return random actions until we can figure out which object is which
        # return random.randint(0, self.num_actions-1)

        # Step 1: Do breadth first search to a state where we have positive information gain
        # But if we know what each object is, then do normal breadth first search to find reward states.

        # # Ok, here's the current plan. Use breadth first search to try and make it to the goal state
        # # (state with the max reward). If we can't make it, take a random action.
        #
        # # Search for a way to the goal. If one exists, execute the first action (the path is reversed)
        if not self.model.solved:
            print("Not solved, looking for information")
            path = self.breadth_first_search_to_information_gain(curr_state)
            if len(path) > 0:
                print(f"Found path to info gain taking action {path[-1]}, {path}")
                return path[-1]
            print("Couldn't find path to info gain")
        else:
            print("Solved, looking for goal")
            path = self.breadth_first_search_to_goal(curr_state)
            if len(path) > 0:
                print(f"Found path to goal taking action {path[-1]}, {path}")
                return path[-1]
            print("Couldn't find path to goal")

        # If we couldn't find a path, choose a random action
        return random.randint(0, self.num_actions-1)

    def breadth_first_search_to_information_gain(self, state: int):
        """
        Uses breadth first search to find the sequence of actions to take
        to a state where it can gain information about which objects are which
        """
        # List of visited states
        visited = []

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
            visited.append(curr_state)

            # print(f"Popped state: {self.model.env.get_factored_state(curr_state)}")

            # Generate all next states
            for action in range(self.num_actions):
                # If we find an action with a positive info gain, then return the path to take that action
                # Otherwise, generate next states
                if information_gain_of_action(self.model.env, curr_state, action, self.model.object_map, self.model.previous_ruleset) > 0:
                    path = self.get_path(curr_state, parents)
                    path.insert(0, action)
                    return path

                # Compute the next state, or don't do anything if we don't know
                transitions = self.model.compute_possible_transitions(curr_state, action)  # TODO: only need to compute literals once, not for each action

                if len(transitions) == 0:
                    # We don't know what will happen, skip
                    continue

                effect = transitions[0].effect  # Assume only one effect, extract it from the transition
                next_state = self.model.next_state(curr_state, effect)

                # Already saw this state
                if next_state in visited:
                    continue

                # Update parents for this state. Need to do before otherwise it won't be in the list when we try
                parents[next_state] = [curr_state, action]

                # Found a goal state if the info gain from this state is positive
                if self.model.get_reward(curr_state, next_state, action) > 0:
                    return self.get_path(next_state, parents)  # Path to that state that gives us a reward

                # Otherwise, add to queue
                q.append(next_state)

        # If we get down here, there is no path, so return empty
        return []

    def breadth_first_search_to_goal(self, state: int):
        """Uses breadth first search to find the goal using what it currently knows about transitions"""

        # List of visited states
        visited = []

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
            visited.append(curr_state)

            # print(f"Popped state: {self.model.env.get_factored_state(curr_state)}")

            # Generate all next states
            for action in range(self.num_actions):
                # Compute the next state, or don't do anything if we don't know
                transitions = self.model.compute_possible_transitions(curr_state, action)  # TODO: only need to compute literals once, not for each action

                if len(transitions) == 0:
                    # We don't know, skip
                    continue

                effect = transitions[0].effect  # Assume only one effect, extract it from the transition
                next_state = self.model.next_state(curr_state, effect)

                # Already saw this state
                if next_state in visited:
                    continue

                # Update parents for this state. Need to do before otherwise it won't be in the list when we try
                parents[next_state] = [curr_state, action]

                # Found a goal state if reward is bigger than 0
                if self.model.get_reward(curr_state, next_state, action) > 0:
                    return self.get_path(next_state, parents)  # Path to that state that gives us a reward

                # Otherwise, add to queue
                q.append(next_state)

        # If we get down here, there is no path, so return empty
        return []

    def get_path(self, parent, parents):
        path = []
        while parents[parent][0] != -1:
            path.append(parents[parent][1])
            parent = parents[parent][0]

        return path
