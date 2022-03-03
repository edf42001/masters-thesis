from algorithm.simulator import Simulator
from environment.environment import Environment
from policy.hierarchical_policy import HierarchicalPolicy
from policy.samplers.sampler import Sampler


class MaxQ(Simulator):
    def __init__(self, env: Environment, policy: HierarchicalPolicy, sampler: Sampler, visualize: bool = False, all_goals: bool = False):
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.visualize = visualize
        self.all_goals = all_goals

        self.hierarchy = env.get_hierarchy()
        self.action_stack = []

        self.curr_state = self.env.get_state()
        self.last_episode_steps = -1
        self.last_episode_reward = -1
        self.max_steps = -1
        self.is_learning = True

        # Non primitive actions are those with subtasks, i.e., the children list is not empty
        self.non_primitives = [subtask for subtask, children in self.hierarchy.children.items() if children]
        print("Non primitive actions: {}".format(self.non_primitives))
