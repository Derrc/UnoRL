import collections
import numpy as np
import random

class qLearningAgent(object):
    def __init__(self, num_actions, epsilon = 0.05):
        ''' Initialize the q learning agent
        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions
        self.epsilon = float(epsilon)
        self.QValues = collections.defaultdict(float)

    def getQValue(self, state, action):

        #dict is initialized to be all 0.0, if a state is unseen before, it will return 0.0
        return self.QValues[(state, action)]

    def computeActionFromQValues(self, state):
        
        actions = list(state['legal_actions'].keys())

        if len(actions) == 0:
            return None

        Q = float('-inf')
        A = None

        for action in actions:
            QValue = self.getQValue(state, action)

            if QValue == Q:
                A = random.choice((A, action)) #break ties randomly for better behavior

            if QValue > Q:
                Q = QValue
                A = action

        return A

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
    def getValue(self, state):
        return self.computeValueFromQValues(state)
    
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = list(state['legal_actions'].keys())

        if len(actions) == 0:
          return 0.0

        max_action = self.getPolicy(state)
        QValue = self.getQValue(state, max_action)
        return QValue

    @staticmethod
    def step(self, state):
        ''' Predict the action given the curent state in generating training data.
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action chosen by the qlearning agent
        '''
        legalActions = list(state['legal_actions'].keys())
        action = np.random.choice(legalActions)
        def flipCoin(p):
            r = random.random()
            return r < p

        if len(legalActions) == 0:
            return action
        else:
            #self.epsilon
            if flipCoin(self.epsilon):
                action = np.random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        return action

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted chosen  by the agent
            probs (list): The list of action probabilities
        '''

        actions = list(state['legal_actions'].keys())
        for action in actions:
            maxQValue = self.getValue(nextState)
            QValue = self.getQValue(state, action)

            key = (state,action)
            val = (1 - self.alpha) * QValue + self.alpha * (reward + self.discount * maxQValue)
            self.QValues[key] = val

        probs = [0 for _ in range(self.num_actions)]
        # for i in state['legal_actions']:
        #     probs[i] = 1/len(state['legal_actions'])

        info = {}
        # info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info