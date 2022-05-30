import math
import random
import argparse
import numpy as np
from scipy.spatial.distance import cityblock

from aasma import Agent
from aasma.utils import compare_results
from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from lab1_solutions.exercise_1_single_random_agent import run_single_agent, RandomAgent

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)


class GreedyAgent(Agent):

    """
    A baseline agent for the SimplifiedPredatorPrey environment.
    The greedy agent finds the nearest prey and moves towards it.
    """

    def __init__(self, agent_id, n_agents):
        super(GreedyAgent, self).__init__(f"Greedy Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS

    def action(self) -> int:
        agents_positions = self.observation[:self.n_agents * 2]
        # TODO - Expect 5-8 lines of code
        #  You may use the two auxiliary methods down below
        #  Warning: Your code should work for an arbitrary number of preys
        #  (even though the examples considers a single one)
        prey_positions = self.observation[self.n_agents * 2:]
        agent_position = agents_positions[self.agent_id * 2], agents_positions[(self.agent_id * 2) + 1]
        closest_prey = self.closest_prey(agent_position, prey_positions)
        prey_found = closest_prey is not None
        return self.direction_to_go(agent_position, closest_prey) if prey_found else random.randrange(N_ACTIONS)

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def direction_to_go(self, agent_position, prey_position):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """
        distances = np.array(prey_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)
        if abs_distances[0] > abs_distances[1]:
            return self._close_horizontally(distances)
        elif abs_distances[0] < abs_distances[1]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    def closest_prey(self, agent_position, prey_positions):
        """
        Given the positions of an agent and a sequence of positions of all prey,
        returns the positions of the closest prey
        """
        min = math.inf
        closest_prey_position = None
        n_preys = int(len(prey_positions) / 2)
        for p in range(n_preys):
            prey_position = prey_positions[p * 2], prey_positions[(p * 2) + 1]
            distance = cityblock(agent_position, prey_position)
            if distance < min:
                min = distance
                closest_prey_position = prey_position
        return closest_prey_position

    # ############### #
    # Private Methods #
    # ############### #

    def _close_horizontally(self, distances):
        if distances[0] > 0:
            return RIGHT
        elif distances[0] < 0:
            return LEFT
        else:
            return STAY

    def _close_vertically(self, distances):
        if distances[1] > 0:
            return DOWN
        elif distances[1] < 0:
            return UP
        else:
            return STAY

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=30)
    opt = parser.parse_args()

    # 1 - Setup environment
    environment = SimplifiedPredatorPrey(
        grid_shape=(7, 7),
        n_agents=1, n_preys=1,
        max_steps=100, required_captors=1
    )
    environment = SingleAgentWrapper(environment, agent_id=0)

    # 2 - Setup agents
    agents = [
        RandomAgent(environment.action_space.n),
        GreedyAgent(agent_id=0, n_agents=1)
    ]

    # 3 - Evaluate agents
    results = {}
    for agent in agents:
        result = run_single_agent(environment, agent, opt.episodes)
        results[agent.name] = result

    # 4 - Compare results
    compare_results(results, title="Agents on 'Predator Prey' Environment", colors=["orange", "green"])

