import random
from collections import deque

from algorithm.transition_model import TransitionModel
from policy.policy import Policy
from symbolic_stochastic_domains.predicates_and_objects import PredicateType


class ObjectTransferPolicy(Policy):
    def __init__(self, actions: int, model: TransitionModel):
        self.num_actions = actions

        self.model = model

        # Used to transfer information from the closest path to a new experience from the breadth first function
        self.path_to_experience = None

    def choose_action(self, curr_state: int, is_learning: bool = True) -> int:
        # Ok, here's the current plan. Use breadth first search to try and make it to the goal state
        # (state with the max reward). If we can't make it, take a random action.

        # Search for a way to the goal. If one exists, execute the first action (the path is reversed)
        path = self.breadth_first_search_to_goal(curr_state)
        if len(path) > 0:
            print(f"Found path to reward taking action {path[-1]}, {path}")
            return path[-1]

        print("Couldn't find path to goal")

        # Secondary check: Were we able to find a path to a new experience?
        if self.path_to_experience is not None:
            path = self.path_to_experience
            self.path_to_experience = None
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
                # Check if we haven't tried this action yet with the current combination of objects
                # We do this at the same time as searching for a goal to save processing power

                if self.path_to_experience is None:
                    literals, _ = self.model.env.get_literals(curr_state)  # TODO: we already get this in compute possible transitions, is a waste
                    for edge in literals.base_object.edges:
                        to_object = edge.to_node.object_name[:-1]

                        # A hacky hack to include properties in experience so the agent can try going throw a lock that opens
                        edge_type = str(edge.type)[14:] + ("_OPEN_" + str(edge.to_node.properties[PredicateType.OPEN]) if len(edge.to_node.properties) > 0 else "")

                        # Check if we haven't tried it
                        if to_object not in self.model.experience or edge_type not in self.model.experience[to_object] or action not in self.model.experience[to_object][edge_type]:
                            print(f"Found an experience: {str(edge.type)[14:]}-{to_object}, {action}")
                            # Path to the current state, plus add the new action on
                            path = self.get_path(curr_state, parents)
                            path.insert(0, action)
                            self.path_to_experience = path
                            break  # Just break out of the for edge for loop since we already found one

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
