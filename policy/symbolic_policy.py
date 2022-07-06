import random
from collections import deque

from algorithm.transition_model import TransitionModel
from policy.policy import Policy


class SymbolicPolicy(Policy):
    def __init__(self, actions: int, model: TransitionModel):
        self.num_actions = actions

        self.model = model

    def choose_action(self, curr_state: int, is_learning: bool = True) -> int:
        # Ok, here's the current plan. Use breadth first search to try and make it to the goal state
        # (state with the max reward). If we can't make it, take a random action.

        # Search for a way to the goal. If one exists, execute the first action (the path is reversed)
        path = self.breadth_first_search_to_goal(curr_state)
        if len(path) > 0:
            print(f"Found path to reward taking action {path[-1]}, {path}")
            return path[-1]

        print("Couldn't find path to goal")

        path = self.breadth_first_search_to_untried_interaction(curr_state)
        if len(path) > 0:
            print(f"Found path to new experience taking action {path[-1]}, {path}")
            return path[-1]

        print("Couldn't find path to new experience")

    # If we couldn't find a path, choose a random action
        return random.randint(0, self.num_actions-1)

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
                    path = []
                    parent = next_state
                    while parents[parent][0] != -1:
                        path.append(parents[parent][1])
                        parent = parents[parent][0]

                    return path

                # Otherwise, add to queue
                q.append(next_state)

        # If we get down here, there is no path, so return empty
        return []

    def breadth_first_search_to_untried_interaction(self, state: int):
        # TODO: could integrate this into the other breadth first so it learns this while it's trying that,
        # so we don't have to do the whole breadth first twice
        # List of visited states
        visited = []

        # Queue to store in progress states
        q = deque()

        # Dictionary of state: [it's parent, the action that led to it]
        parents = {}

        # Initialize search
        q.append(state)
        parents[state] = [-1, -1]

        # The "on the key" state is 28686

        # print("Beginning breadth first search")
        while len(q) != 0:
            curr_state = q.popleft()
            visited.append(curr_state)

            # print(f"Popped state: {self.model.env.get_factored_state(curr_state)}")

            literals, _ = self.model.env.get_literals(curr_state)

            # if curr_state == 28686:
            #     print(f"Came from state: {state}")
            #     print(self.model.experience["key"])
            #     print(literals)

            # Generate all next states
            for action in range(self.num_actions):
                # Check if we haven't tried this action yet with the current combination of objects

                for edge in literals.base_object.edges:
                    to_object = edge.to_node.object_name[:-1]

                    # if curr_state == 28686 and state == 44238:
                    #     print(edge, action)
                    # Check if we haven't tried it
                    if to_object not in self.model.experience or edge.type not in self.model.experience[to_object] or action not in self.model.experience[to_object][edge.type]:
                        print(f"Found an experience: {to_object}, {str(edge.type)} {action}")
                        # print(f"{literals}")
                        # print(self.model.experience)
                        path = []
                        parent = curr_state
                        while parents[parent][0] != -1:
                            path.append(parents[parent][1])
                            parent = parents[parent][0]

                        path.insert(0, action)
                        return path

                # Compute the next state, or don't do anything if we don't know
                transitions = self.model.compute_possible_transitions(curr_state, action)

                if len(transitions) == 0:
                    continue

                effect = transitions[0].effect  # Assume only one effect, extract it from the transition
                next_state = self.model.next_state(curr_state, effect)

                # Already saw this state
                if next_state in visited:
                    continue

                # Update parents for this state. Need to do before otherwise it won't be in the list when we try
                parents[next_state] = [curr_state, action]

                # Otherwise, add to queue
                q.append(next_state)

        # If we get down here, there is no path, so return empty
        return []
