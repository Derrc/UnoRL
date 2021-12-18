import collections
import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done'])

class QNAgent(object):
    def __init__(self, num_actions, epsilon = 0.05, discount=0.99,alpha = 0.00005):
        ''' Initialize the q learning agent
        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions
        self.epsilon = float(epsilon)
        self.discount = discount
        self.alpha = alpha
        self.QValues = {}

    def getQValue(self, state, action):

        #dict is initialized to be all 0.0, if a state is unseen before, it will return 0.0
        key = QKey(state['obs'],action)
        if not key in self.QValues.keys():
            self.QValues[key] = 0.0
        return self.QValues[key]

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

    def feed(self, ts):
        ''' Take in a transition state to train the agent. 
        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        maxQValue = self.getValue(next_state)
        QValue = self.getQValue(state, action)

        key = QKey(state['obs'],action)
        val = (1 - self.alpha) * QValue + self.alpha * (reward + self.discount * maxQValue)
        self.QValues[key] = val
        
        
    
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

    def step(self, state):
        ''' Predict the action given the curent state in generating training data.
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action chosen by the qlearning agent
        '''
        actions = list(state['legal_actions'].keys())
        maxQValue = float('-inf')
        bestAction = None
        for action in actions:
            key = QKey(state['obs'],action)
            if not key in self.QValues.keys():
                self.QValues[key] = 0.0
            currQValue = self.QValues[key]
            if (currQValue > maxQValue):
                maxQValue = currQValue
                bestAction = action

        def flipCoin(p):
            r = random.random()
            return r < p
       
        if flipCoin(self.epsilon):
            return np.random.choice(actions)
        else:
            return bestAction

            

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted chosen by the agent
            probs (list): The list of action probabilities
        '''
        actions = list(state['legal_actions'].keys())
        maxQValue = float('-inf')
        bestAction = None
        for action in actions:
            key = QKey(state['obs'],action)
            if not key in self.QValues.keys():
                self.QValues[key] = 0.0
            currQValue = self.QValues[key]
            if (currQValue > maxQValue):
                maxQValue = currQValue
                bestAction = action

        info = []
        qValueSum = 0
        for qvalue in self.QValues.values():
            qValueSum += qvalue
        for qvalue in self.QValues.values():
            info.append(float(qvalue / (qValueSum + 1) ))

        
        return bestAction, info


class QKey(object):
    def __init__(self,state,action):
        self.state = state
        self.action = action