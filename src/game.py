import numpy as np
from abc import ABC, abstractmethod


class Game(ABC):
    "In this version, the structure only admits games with 2 players and 2 stages and the transition function decribes only the passage from stage 1 to stage 2"
    def __init__(self):
        self.N = 2                      # number of players
        self.H = 2                      # horizon (number of stages)
        self.actions = None             # list of actions (in this version assumed to be the same for both players)
        self.rewards = {1: {}, 2: {}}   # dictionary with rewards np.array for each state in each stage
        self.s_map = {1: {}, 2: {}}     # dictionary with indices for each state in each stage

        self._build()

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def transition(self, a1, a2):
        """ Determines the state of stage 2 based on the action taken in stage 1 """
        pass


class TreasureGame(Game):

    def _build(self):
        self.N = 2      # players number
        self.H = 2      # horizon

        # actions (same for each player)
        self.actions = [0,1]

        # unnormalised rewards tables for each state
        self.rewards = {
            1: {'s1': np.array([[(1.0, 1.0), (0.0,0.0)],
                                [(0.0, 0.0), (0.0,0.0)]]) },
            2: {'A': np.array([[(0.5,0.5), (0.0,0.0)],
                           [(0.0,0.0), (0.0,0.0)]]),

            'B': np.array([[(1.0,1.0), (0.0,0.0)],
                           [(0.0,0.0), (-2.0,2.0)]]),

            'O': np.array([[(1.0,1.0), (0.0,0.0)],
                           [(0.0,0.0), (0.0,0.0)]]) }
        }

        # indices of states per stage
        self.s_map = {
            1: {'s1': 0},
            2: {'A': 0, 'B': 1, 'O': 2}
        }

    def transition(self, a1, a2):
        if a1 == 0 and a2 == 0:
            return 'A'
        if a1 == 1 and a2 == 1:
            return 'B'
        return 'O'


class StagHuntGame(Game):

    def _build(self):
        self.N = 2      # players number
        self.H = 2      # horizon

        # actions (same for each player)
        self.actions = [0,1]

        # unnormalised rewards tables for each state
        self.rewards = {
            1: {'s1': np.array([[(0, 0), (0, 2)], 
                                [(2, 0), (1, 1)]]) },

            2: {'A': np.array([[(3.75, 3.75), (0, 2)],
                            [(2, 0), (1, 1)]]),

                'B': np.array([[(0, 0), (0, 2)],
                           [(2, 0), (1, 1)]]) }
        }

        # indices of states per stage
        self.s_map = {
            1: {'s1': 0},
            2: {'A': 0, 'B': 1}
        }
        

    def transition(self, a1, a2):
        """ Determines the state of stage 2 based on the action taken in stage 1 """
        if a1 == 0 and a2 == 0:
            return 'A'
        return 'B'


# Lookup table
game_dictionary = {
    "treasure": TreasureGame,
    "staghunt": StagHuntGame,
}