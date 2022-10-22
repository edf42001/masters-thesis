import random
from collections import deque

from algorithm.transition_model import TransitionModel
from policy.policy import Policy
from symbolic_stochastic_domains.predicates_and_objects import PredicateType


class SymbolicPolicy(Policy):
    def __init__(self, actions: int, model: TransitionModel):
        self.num_actions = actions

        self.model = model

        # Current path we are executing as long as our knowledge doesn't change
        # This presumes that if an action does not happen as we expect, it will automatically cause a rule change
        self.path = []

        # Used to transfer information from the closest path to a new experience from the breadth first function
        self.path_to_experience = []

        # Store previous ruleset so we can research whenever it changes
        self.last_ruleset = None

        # Unable to find a path to experience either
        self.unable_to_find = False

    def choose_action(self, curr_state: int, is_learning: bool = True) -> int:
        # Ok, here's the current plan. Use breadth first search to try and make it to the goal state
        # (state with the max reward). If we can't make it, take a random action.

        # If a rule change occured, we need to replan, also update unable_to_find
        if str(self.last_ruleset) != str(self.model.ruleset):
            print("Model changed, replanning")
            self.last_ruleset = self.model.ruleset
            self.path = self.breadth_first_search_to_goal(curr_state)
            self.unable_to_find = False

        # Check for valid result, if so, return the first action (path is in reverse order)
        # and remove that action so we take the next action next time
        if len(self.path) > 0:
            action = self.path[-1]
            print(f"Found path to reward taking action {action}, {self.path}")
            del self.path[-1]
            return action

        # Otherwise, check the path to experience
        print("Couldn't find path to goal")

        # Only need to re-search if we don't currently know where we are going
        if len(self.path_to_experience) == 0 and not self.unable_to_find:
            self.path = self.breadth_first_search_to_goal(curr_state)  # TODO: Need to research for new experience?

            # If it is still 0, mark that we weren't able to find it, and we should just keep taking
            # random actions until the ruleset changes. This may break down in cases of stochasticity.
            if len(self.path_to_experience) == 0:
                self.unable_to_find = True

        # Secondary check: Were we able to find a path to a new experience?
        if len(self.path_to_experience) > 0:
            action = self.path_to_experience[-1]
            print(f"Found path to experience taking action {action}, {self.path_to_experience}")
            del self.path_to_experience[-1]
            return action

        print("Couldn't find path to new experience")

        # If we couldn't find a path, choose a random action
        return random.randint(0, self.num_actions-1)

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

            # Generate all next states. Check if each is a new experience. Pick a new experience at random
            new_experiences = [False] * self.num_actions
            for action in range(self.num_actions):
                # Check if we haven't tried this action yet with the current combination of objects
                # We do this at the same time as searching for a goal to save processing power

                if len(self.path_to_experience) == 0:
                    for edge in literals.base_object.edges:
                        to_object = edge.to_node.object_name

                        # A hacky hack to include properties in experience so the agent can try going throw a lock that opens
                        edge_type = str(edge.type)[14:] + ("_OPEN_" + str(edge.to_node.properties[PredicateType.OPEN]) if len(edge.to_node.properties) > 0 else "")

                        # Check if we haven't tried it
                        if to_object not in self.model.experience or edge_type not in self.model.experience[to_object] or action not in self.model.experience[to_object][edge_type]:
                            print(f"Found an experience: {edge.type.name}-{to_object}, {action}")
                            new_experiences[action] = True

                            # This action is a new experience, we don't need to process the others.
                            # Eventually, if one action gets more new experiences than others, take that first.
                            break

                # Compute the next state, or don't do anything if we don't know. Pass literals for efficiency
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

            # After trying all actions, if none lead to a goal, pick a new experience at random
            if any(new_experiences):
                new_experience_actions = [i for i, v in enumerate(new_experiences) if v]
                path = self.get_path(curr_state, parents)
                path.insert(0, random.choice(new_experience_actions))
                self.path_to_experience = path

        # If we get down here, there is no path, so return empty
        return []

    def get_path(self, parent, parents):
        path = []
        while parents[parent][0] != -1:
            path.append(parents[parent][1])
            parent = parents[parent][0]

        return path
