import numpy as np
from rlcard.models.model import Model
import rlcard

class BestRuleAgent(object):
    ''' An agent that follows the following rules:
    1. If enemy card number is 3 or lower, play draw 2, wild 4
    2. If enemy last action was draw, keep same color, dont play wilds, same number diff color
    3. Play numbers that appeared more often first if same color
    4. If enemy card number is 1, do whatever to prevent them playing, i.e. skip,reverse
    5. Only play same number card to change color if size(newColorCards) > size(currColorCards)
    '''
    def __init__(self, num_actions):
        self.use_raw = False
        self.num_actions = num_actions


    def last_played_card_by(self,action_record,player_id):
        ''' reverse (as if iterating through backwards) '''
        action_record.reverse()
        for id, action in action_record:
            if id == player_id:
                return action

        return 'None'

    def action_to_string(self,actionID):
        actionStrings = ['r-0','r-1','r-2','r-3','r-4','r-5','r-6','r-7','r-8','r-9','r-skip','r-reverse','r-draw_2','r-wild','r-wild_draw_4',
                        'g-0','g-1','g-2','g-3','g-4','g-5','g-6','g-7','g-8','g-9','g-skip','g-reverse','g-draw_2','g-wild','g-wild_draw_4',
                        'b-0','b-1','b-2','b-3','b-4','b-5','b-6','b-7','b-8','b-9','b-skip','b-reverse','b-draw_2','b-wild','b-wild_draw_4',
                        'y-0','y-1','y-2','y-3','y-4','y-5','y-6','y-7','y-8','y-9','y-skip','y-reverse','y-draw_2','y-wild','y-wild_draw_4','draw']

        return actionStrings[actionID]

    def num_played(self,action_record,card):
        count = 0
        for id,action in action_record:
            if action == card:
                count += 1

        return count

    def findSameColor(self,legalActions,target):
        targetNum = target.split('-')[1]
        if targetNum == ('draw_2' or 'reverse' or 'skip' or 'wild' or 'wild_draw_4'):
            return None

        if targetNum in ['0','1','2','3','4','5','6','7','8','9']:
            targetNum = int(targetNum)
        for action in legalActions:
            if action % 15 == target and (action % 15 >= 0 and action % 15 <= 9):
                return action

    def numColorInHand(self,hand,color):
        count = 0
        for card in hand:
            if card.split('-')[0] == color:
                count += 1

        return count

    def findWild4(self,legalActions):
        for action in legalActions:
            if action == (14 or 29 or 44 or 59):
                return action
        return None

    def findWild(self,legalActions):
        for action in legalActions:
            if action == (13 or 28 or 43 or 58):
                return action
        return None

    def findDraw2(self,legalActions):
        for action in legalActions:
            if action == (12 or 27 or 42 or 57):
                return action
        return None
    
    def findSkip(self,legalActions):
        for action in legalActions:
            if action == (10 or 25 or 40 or 55):
                return action

        return None

    def findReverse(self, legalActions):
        for action in legalActions:
            if action == (11 or 26 or 41 or 56):
                return action

        return None



    def step(self, state):
        ''' Predict the action given the current state in generating training data.
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted by the agent
        '''

        ''' get all needed information from state'''
        legalActions = list(state['legal_actions'].keys())
        target = state['raw_obs']['target']
        hand = state['raw_obs']['hand']

        current_player_id = state['raw_obs']['current_player']
        enemy_player_id = (current_player_id + 1) % 2
        card_number = state['raw_obs']['num_cards'][current_player_id]
        enemy_card_number = state['raw_obs']['num_cards'][enemy_player_id]

        past_actions = state['action_record']
        last_played_enemy_card = self.last_played_card_by(past_actions,enemy_player_id)


        ''' string version of legal actions (not actionID) '''
        legalActions_string = []
        for action in legalActions:
            legalActions_string.append(self.action_to_string(action))

        ''' count number of cards played for each card in legal actions '''
        legalAction_card_counts = []
        for action in legalActions_string:
            legalAction_card_counts.append(self.num_played(past_actions,action))


        ''' if length of legalActions is 1, have to draw '''
        if len(legalActions) == 1:
            return np.random.choice(legalActions)



        ''' apply rules '''
        if enemy_card_number <= 3:
            if self.findDraw2(legalActions) != None:
                newList = list(filter(lambda x: x == self.findDraw2(legalActions),legalActions))
                return np.random.choice(newList)
            elif self.findWild4(legalActions) != None:
                newList = list(filter(lambda x: x == self.findWild4(legalActions),legalActions))
                return np.random.choice(newList)
            elif self.findSkip(legalActions) != None:
                newList = list(filter(lambda x: x == self.findSkip(legalActions),legalActions))
                return np.random.choice(newList)
            elif self.findReverse(legalActions) != None:
                newList = list(filter(lambda x: x == self.findReverse(legalActions),legalActions))
                return np.random.choice(newList)


        ''' enemy has more than 3 cards or less than 3 cards and we have no draw2, wild4, skip, or reverse'''
        if last_played_enemy_card == 'draw':
            ''' play same color cards that have been played the most '''
            maxCardNum = float('-inf')
            bestCard = None
            for card in range(len(legalActions)):
                if legalActions_string[card].split('-')[1] in ['0','1','2','3','4','5','6','7','8','9']:
                    if legalAction_card_counts[card] > maxCardNum:
                        maxCardNum = legalAction_card_counts[card]
                        bestCard = legalActions[card]
            if bestCard != None:
                newList = list(filter(lambda x: x == bestCard,legalActions))
                return np.random.choice(newList)

        ''' enemy's last card played wasn't draw, could have more than 3 cards or less than 3 cards'''
        ''' if enemy card number is 1 NEED to change color '''
        if enemy_card_number == 1:
            if self.findWild(legalActions) != None:
                newList = list(filter(lambda x: x == self.findWild(legalActions),legalActions))
                return np.random.choice(newList)

        ''' if same number to change color is available, check if size(newColor) > size(currColor) '''
        if self.findSameColor(legalActions,target) != None:
            sameColorAction = self.findSameColor(legalActions,target)
            currColor = target.split('-')[0]
            newColor = self.action_to_string(sameColorAction).split('-')[0]
            if self.numColorInHand(hand,newColor) > self.numColorInHand(hand,currColor):
                newList = list(filter(lambda x: x == sameColorAction,legalActions))
                return np.random.choice(newList)
        
        ''' else choose color cards that have been played the most '''
        maxCardNum = float('-inf')
        bestCard = None
        for card in range(len(legalActions_string)):
            currentCardNumber = legalActions_string[card].split('-')[1]
            if currentCardNumber in ['0','1','2','3','4','5','6','7','8','9']:
                if legalAction_card_counts[card] > maxCardNum:
                    maxCardNum = legalAction_card_counts[card]
                    bestCard = legalActions[card]
        if bestCard != None:
            newList = list(filter(lambda x: x == bestCard,legalActions))
            return np.random.choice(newList)

        ''' else no cards are available -> draw '''


        

        return np.random.choice(legalActions)


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