import random
from collections import deque

from policy.policy import Policy

from symbolic_stochastic_domains.object_transfer import information_gain_of_action
from algorithm.symbolic_domains.object_transfer_model import ObjectTransferModel


class ObjectTransferPolicy(Policy):
    def __init__(self, actions: int, model: ObjectTransferModel):
        self.num_actions = actions

        self.model = model

        # Used to transfer information from the closest path to a new experience from the breadth first function
        self.path = []

        self.last_object_map = {}

    def choose_action(self, curr_state: int, is_learning: bool = True) -> int:
        # For now, return random actions until we can figure out which object is which
        # return random.randint(0, self.num_actions-1)

        # Step 1: Do breadth first search to a state where we have positive information gain
        # But if we know what each object is, then do normal breadth first search to find reward states.

        # Ok, here's the current plan. Use breadth first search to try and make it to the goal state
        # (state with the max reward). If we can't make it, take a random action.

        # If an object map change occured, we need to replan
        replan = False
        if str(self.last_object_map) != str(self.model.object_map):
            print("Model changed, replanning")
            self.last_object_map = self.model.object_map
            self.path = self.breadth_first_search_to_goal(curr_state)
            replan = True

        if not self.model.solved:
            print("Not solved, looking for information")
            if replan or len(self.path) == 0:
                self.path = self.breadth_first_search_to_information_gain(curr_state)

            if len(self.path) > 0:
                action = self.path[-1]
                print(f"Found path to info gain taking action {action}, {self.path}")
                del self.path[-1]
                return action

            print("Couldn't find path to info gain")
        else:
            print("Solved, looking for goal")
            if replan or len(self.path) == 0:
                self.path = self.breadth_first_search_to_goal(curr_state)

            if len(self.path) > 0:
                action = self.path[-1]
                print(f"Found path to goal taking action {action}, {self.path}")
                del self.path[-1]
                return action

            print("Couldn't find path to goal")

        # If we couldn't find a path, choose a random action
        return random.randint(0, self.num_actions-1)

    def breadth_first_search_to_information_gain(self, state: int):
        """
        Uses breadth first search to find the sequence of actions to take
        to a state where it can gain information about which objects are which
        """
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
            for action in range(self.num_actions):
                # If we find an action with a positive info gain, then return the path to take that action
                # Otherwise, generate next states
                if information_gain_of_action(self.model.env, curr_state, action, self.model.object_map, self.model.previous_ruleset) > 0:
                    path = self.get_path(curr_state, parents)
                    path.insert(0, action)
                    return path

                # Compute the next state, or don't do anything if we don't know. Pass literals for efficiency
                transitions = self.model.compute_possible_transitions(
                    curr_state, action, literals=literals, instance_name_map=instance_name_map
                )

                if len(transitions) == 0:
                    # We don't know what will happen, skip
                    continue

                # TODO figure this out assert len(transitions) < 2, "Only 1 transition per state"

                effect = transitions[0].effect  # Assume only one effect, extract it from the transition
                next_state = self.model.next_state(curr_state, effect)

                # Don't add if we already saw this state or it is already in the queue
                if next_state in visited or next_state in q:
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
            for action in range(self.num_actions):
                # Compute the next state, or don't do anything if we don't know
                transitions = self.model.compute_possible_transitions(
                    curr_state, action, literals=literals, instance_name_map=instance_name_map
                )

                if len(transitions) == 0:
                    # We don't know, skip
                    continue

                effect = transitions[0].effect  # Assume only one effect, extract it from the transition
                next_state = self.model.next_state(curr_state, effect)

                # Don't add if we already saw this state or it is already in the queue
                if next_state in visited or next_state in q:
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
