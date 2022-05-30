import argparse
import random
import numpy as np

from aasma.utils import compare_results
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from lab1_solutions.exercise_3_multi_agent import run_multi_agent
from lab1_solutions.exercise_1_single_random_agent import RandomAgent
from lab1_solutions.exercise_2_single_random_vs_greedy import GreedyAgent

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    opt = parser.parse_args()

    # 1 - Setup the environment
    # TODO: Instantiate the SimplifiedPredatorPrey environment with the
    # desired number of predators required to capture the prey. The
    # environment must be instantiated with the following arguments:
    # grid_shape=(15, 15), n_agents=4, n_preys=1, max_steps=100, and
    # the number of required_captors.
    raise ValueError("Not implemented.")

    # Set seeds.
    random.seed(3)
    np.random.seed(3)
    environment.seed(3)

    # 2 - Setup the teams
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
        colors=["orange", "green"]
    )

