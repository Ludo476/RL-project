import numpy as np
from abc import ABC, abstractmethod


class LearningRule(ABC):
    """
    Base class for a learning rule, determines the framework that actual learning rules have to follow.
    """
    @abstractmethod
    def update_vars(self,current_action, current_hidden, num_players, actions, q_vals):
        """
        Determines the new actions for the agents.

        Args:
            current_action (list): current joint action [a1, a2].
            current_hidden (list): current joint hidden vars [xi_1, xi_2].
            num_players (int): number of agents.
            actions (list): possible actions for the agents to play (we assume they are the same for all agents).
            q_vals (list): Q-values for the possible actions of the agents.

        Returns:
            list, list: new action and new auxiliary variable.
        """
        pass


class LogLinearRule(LearningRule):
    """
    Implements the Log-Linear learning rule, which correspond to a softmax dependent from the parameter epsilon.
    Hidden variables are not necessary, the function returns the initialised value.
    Parameter: epsilon in (0,1)
    """
    def __init__(self, epsilon):
        if (epsilon <= 0) or (epsilon >=1):
            raise ValueError("The parameter epsilon has to be in (0,1).")
        self.epsilon = epsilon
        self.norm_rewards = False

    def update_vars(self, current_action, current_hidden, num_players, actions, q_vals):
        player_to_update = np.random.randint(num_players)

        if (player_to_update == 0):
            q_values_for_player = np.array(q_vals[player_to_update,:,current_action[1]])
        else:
            q_values_for_player = np.array(q_vals[player_to_update,current_action[0],:])

        unnorm_probs = pow(self.epsilon, -q_values_for_player)
        new_action_probs = unnorm_probs / (unnorm_probs.sum())
        
        new_action_for_player = np.random.choice(actions, p=new_action_probs)
        
        new_joint_action = list(current_action)
        new_joint_action[player_to_update] = new_action_for_player
        
        return new_joint_action, current_hidden
    
    
class MardenMoodRule(LearningRule):
    """
    Implements the Marden Mood learning rule.
    The hidden variables represent the internal mood of each agent: C (content) or D (discontent).
    Parameters: epsilon in (0,1), c >= num_players.
    """

    def __init__(self, epsilon, c, reward_prec:int=2):
        # self.beta = beta
        if (epsilon <= 0) or (epsilon >= 1):
            raise ValueError("The parameter epsilon has to be in (0,1).")
        self.epsilon = epsilon
        self.c = c
        self.norm_rewards = True
        self.reward_prec = reward_prec


    def update_vars(self,current_action, current_hidden, num_players, actions, q_vals):
        
        #Action update
        new_action = current_action[:]

        for i in range(num_players):
            if current_hidden[i]  == 'D':                       # discontent -> chooses randomly
                new_action[i] = np.random.choice(actions)

            else:                                               # content -> explores with a small probability
                prob_explore = pow(self.epsilon,self.c)

                if np.random.rand() < prob_explore:
                    other_actions = [a for a in actions if a != current_action[i]]
                    new_action[i] = np.random.choice(other_actions) if other_actions else current_action[i]     # choose a different action than the current if there are others
                else:
                    new_action[i] = current_action[i]
        
        #Mood update
        new_hidden = np.copy(current_hidden)

        for i in range(num_players):

            if (current_hidden[i] == 'C') and (current_action == new_action):   # content and action didn't change -> content
                new_hidden[i] = 'C'
            else:                                                               # else -> content with a higher prob. the higher is Q
                prob_content = pow(self.epsilon, 1 - q_vals[i][new_action[0]][new_action[1]])
                # print(f"Player {i}, prob to become C:", prob_content)

                if np.random.rand() < prob_content:
                    new_hidden[i] = 'C'
                else:
                    new_hidden[i] = 'D'

        return new_action, new_hidden
    


# Lookup table
learning_rule_dictionary = {
    "loglinear": LogLinearRule,
    "mardenmood": MardenMoodRule,
}