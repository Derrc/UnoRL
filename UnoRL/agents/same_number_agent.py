import numpy as np
from rlcard.models.model import Model
import rlcard

class SameNumberAgent(object):
    ''' An agent that will select the same number that is on the target first.
    '''
    def __init__(self, num_actions):
        self.use_raw = False
        self.num_actions = num_actions

    def feed(self,ts):
        (state, action, reward, next_state, done) = tuple(ts)

    def step(self, state):
        ''' Predict the action given the current state in generating training data.
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted by the agent
        '''
        legalActions = list(state['legal_actions'].keys())
       
        target = state['raw_obs']['target']
        # print("target: " + target)
        targetNumber = target[2:]
        # print("targetNumber: " + targetNumber)

        def equalsTargetNumber(action):
            if action == targetNumber:
                return True 

            elif str(targetNumber).isnumeric() and str(action).isnumeric():
                if abs(int(action) - int(targetNumber)) % 15== 0:
                    return True
                return False
            elif isinstance(targetNumber, str):
                if 'wild' in targetNumber:
                    if targetNumber == (13 or 28 or 42 or 58):
                        return True
                if 'reverse' in targetNumber:
                     if targetNumber == (11 or 26 or 41 or 56):
                        return True
                if 'draw_2' in targetNumber:
                     if targetNumber == (12 or 27 or 42 or 57):
                        return True
                if 'wild_draw_4' in targetNumber:
                     if targetNumber == (14 or 29 or 43 or 59):
                        return True
            else:
                return False

        filteredLegalActions = list(filter(equalsTargetNumber,legalActions))
        if filteredLegalActions:
            return np.random.choice(filteredLegalActions)
        else:
            return np.random.choice(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info

# class UNORuleModelSameNumber(Model):
#     ''' UNO Rule Model version 1
#     '''

#     def __init__(self):
#         ''' Load pretrained model
#         '''
#         env = rlcard.make('uno')

#         rule_agent = SameNumberAgent()
#         self.rule_agents = [rule_agent for _ in range(env.num_players)]

#     @property
#     def agents(self):
#         ''' Get a list of agents for each position in a the game
#         Returns:
#             agents (list): A list of agents
#         Note: Each agent should be just like RL agent with step and eval_step
#               functioning well.
#         '''
#         return self.rule_agents

#     @property
#     def use_raw(self):
#         ''' Indicate whether use raw state and action
#         Returns:
#             use_raw (boolean): True if using raw state and action
#         '''
#         return True
