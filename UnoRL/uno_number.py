import argparse

import rlcard

from rlcard.utils import set_seed

from agents.same_number_agent import SameNumberAgent

def run(args):
    # Make environment
    env = rlcard.make(args.env, config={'seed': 42})
    num_episodes = 2

    # Seed numpy, torch, random
    set_seed(42)

    # Set agents
    agent = SameNumberAgent(num_actions=env.num_actions)
    env.set_agents([agent for _ in range(env.num_players)])

    for episode in range(num_episodes):

        # Generate data from the environment
        trajectories, player_wins = env.run(is_training=False)
        # Print out the trajectories
        print('\nEpisode {}'.format(episode))
        print(player_wins)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Same Number Agent Uno Example")
    parser.add_argument('--env', type=str, default='uno')

    args = parser.parse_args()

    run(args)