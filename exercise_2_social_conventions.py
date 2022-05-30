import argparse
import random
import numpy as np
from typing import List, Tuple

from aasma import Agent
from aasma.utils import compare_results
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from lab1_solutions.exercise_1_single_random_agent import RandomAgent
from lab1_solutions.exercise_2_single_random_vs_greedy import GreedyAgent
from lab1_solutions.exercise_3_multi_agent import run_multi_agent

ACTIONS = 4
GO_NORTH, GO_SOUTH, GO_WEST, GO_EAST = range(ACTIONS)

MOVEMENTS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(MOVEMENTS)


class ConventionAgent(Agent):
    
    def __init__(self, agent_id: int, n_agents: int, social_conventions: List):
        super(ConventionAgent, self).__init__(f"Convention Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.conventions = social_conventions
        self.n_actions = ACTIONS
        
    def action(self) -> int:
        agent_order = self.conventions[0]
        action_order = self.conventions[1]
        prey_pos = self.observation[self.n_agents * 2:]
        agent_pos = (self.observation[self.agent_id * 2], self.observation[self.agent_id * 2 + 1])

        # TODO: Use the self.conventions attribute to pick how the agent should
        # corner the prey and then move the agent in the corresponding direction.
        # To implement the movement of the predator towards the prey you should
        # re-use other method(s) of this class.
        raise ValueError("Not implemented.")

    def advance_to_pos(self, agent_pos: Tuple, prey_pos: Tuple, agent_dest: int) -> int:
        """
        Choose movement action to advance agent towards the destination around prey
        
        :param agent_pos: current agent position
        :param prey_pos: prey position
        :param agent_dest: agent destination in relation to prey (0 for NORTH, 1 for SOUTH,
                            2 for WEST, and 3 for EAST)

        :return: movement index
        """
    
        def _get_prey_adj_locs(loc: Tuple) -> List[Tuple]:
            prey_x = loc[0]
            prey_y = loc[1]
            return [(prey_x, prey_y - 1), (prey_x, prey_y + 1), (prey_x - 1, prey_y), (prey_x + 1, prey_y)]
        
        def _move_vertically(distances) -> int:
            if distances[1] > 0:
                return DOWN
            elif distances[1] < 0:
                return UP
            else:
                return STAY
            
        def _move_horizontally(distances) -> int:
            if distances[0] > 0:
                return RIGHT
            elif distances[0] < 0:
                return LEFT
            else:
                return STAY
            
        prey_adj_locs = _get_prey_adj_locs(prey_pos)
        distance_dest = np.array(prey_adj_locs[agent_dest]) - np.array(agent_pos)
        abs_distances = np.absolute(distance_dest)
        if abs_distances[0] > abs_distances[1]:
            return _move_horizontally(distance_dest)
        elif abs_distances[0] < abs_distances[1]:
            return _move_vertically(distance_dest)
        else:
            roll = np.random.uniform(0, 1)
            return _move_horizontally(distance_dest) if roll > 0.5 else _move_vertically(distance_dest)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = SimplifiedPredatorPrey(grid_shape=(15, 15), n_agents=4, n_preys=1, max_steps=100, required_captors=4)

    # Set seeds.
    random.seed(3)
    np.random.seed(3)
    environment.seed(3)

    # 2 - Setup the teams
    conventions = [[0, 1, 2, 3], [GO_NORTH, GO_SOUTH, GO_WEST, GO_EAST]]
    teams = {
        
        "Random Team": [
                RandomAgent(environment.action_space[0].n),
                RandomAgent(environment.action_space[1].n),
                RandomAgent(environment.action_space[2].n),
                RandomAgent(environment.action_space[3].n),
        ],
    
        "Greedy Team": [
                GreedyAgent(agent_id=0, n_agents=4),
                GreedyAgent(agent_id=1, n_agents=4),
                GreedyAgent(agent_id=2, n_agents=4),
                GreedyAgent(agent_id=3, n_agents=4)
        ],

        "Greedy Team \nw/ Social Convention": [
                ConventionAgent(agent_id=0, n_agents=4, social_conventions=conventions),
                ConventionAgent(agent_id=1, n_agents=4, social_conventions=conventions),
                ConventionAgent(agent_id=2, n_agents=4, social_conventions=conventions),
                ConventionAgent(agent_id=3, n_agents=4, social_conventions=conventions)
        ]
    }

    # 3 - Evaluate teams
    results = {}
    for team, agents in teams.items():
        print(f'Running {team}.')
        result = run_multi_agent(environment, agents, opt.episodes)
        results[team] = result

    # 4 - Compare results
    compare_results(
        results,
        title="Teams Comparison on 'Predator Prey' Environment",
        colors=["orange", "green", "blue"]
    )
