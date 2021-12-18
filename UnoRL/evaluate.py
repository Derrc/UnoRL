import os
import argparse

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        if model_path == 'experiments/uno_dqn_result/model.pth':
            agent.set_device(device)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    elif model_path == 'sameNumberRule':
        from agents.same_number_agent import SameNumberAgent
        agent = SameNumberAgent(num_actions=env.num_actions)
    elif model_path == 'bestRule':
        from agents.best_rule_agent import BestRuleAgent
        agent = BestRuleAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    
    return agent

def evaluate(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards = tournament(env, args.num_games)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Uno Agent Evaluation in RLCard")
    parser.add_argument('--env', type=str, default='uno')
    parser.add_argument('--models', nargs='*', default=['random', 'experiments/uno_qn_result/model.pth', 'experiments/uno_qn_result/model.pth', 'experiments/uno_dqn_result/model.pth', 'sameNumberRule'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_games', type=int, default=200)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)