import numpy as np
import matplotlib.pyplot as plt
import copy

from src.game import Game 
from src.learning_rule import MardenMoodRule
import numbers
from tqdm import tqdm

# Learning Algorithm

class UnifiedLearning:
    """
    Implements the algorithm Unified Learning Framework for a multi-agent game with finite horizon and two players.
    """
    def __init__(self, game, T, learning_rule):
        self.T = T          # number of learning iterations

        self.learning_rule = learning_rule

        if learning_rule.norm_rewards:
            self.game = self._normalize_rewards(game, learning_rule.reward_prec)
        else:
            self.game = game


        # Definition of variables

        # Q[player][stage h][state_index][action_pl1][action_pl2]
        self.Q = np.zeros((self.game.N, self.game.H + 1, len(self.game.s_map[2]), len(self.game.actions), len(self.game.actions)))

        # V[player][stage h][state_index]
        self.V = np.zeros((self.game.N, self.game.H + 2, len(self.game.s_map[2])))
        
        # a[stage h][state_index] -> (a1, a2)
        self.a = {}

        # xi[stage h][state_index] -> (xi_1, xi_2)
        self.hidden = {}

        # Save cronology of the state s1 to check convergence
        self.V_history = []             # V-value just of player 0 (we have symmetric games)
        self.s1_action_history = []     # pair of actions taken by both players

    def _normalize_rewards(self,game: Game, prec: int) -> Game:
        g = copy.deepcopy(game)
        
        max_val = 0.0
        for stage_data in g.rewards.values():
            for reward_matrix in stage_data.values():
                max_val = max(max_val, np.max(np.abs(reward_matrix)))

        if max_val == 0:
            return g  # nothing to normalize

        normalized = {}
        for stage, stage_data in g.rewards.items():
            normalized[stage] = {}
            for state, reward_matrix in stage_data.items():
                normalized[stage][state] = np.round(reward_matrix / max_val,prec)
        g.rewards = normalized

        return g

    def _initialize(self):
        """ Initialisation of Q-values, actions and hidden variables. """
        
        actions = self.game.actions

        # Q values are initialised to rewards
        for h in range(1, self.game.H + 1):
            for s_str, reward_matrix in self.game.rewards[h].items():
                s_idx = self.game.s_map[h][s_str]
                for a1 in actions:
                    for a2 in actions:
                        reward = reward_matrix[a1, a2]

                        for i in range(self.game.N):
                            self.Q[i, h, s_idx, a1, a2] = reward[i]

        # Actions and hidden variables are initialised randomly
        for h in range(1, self.game.H + 1):
            self.a[h] = {}
            self.hidden[h] = {}
            for _, s_idx in self.game.s_map[h].items():

                a1_rand = np.random.choice(actions)
                a2_rand = np.random.choice(actions)
                self.a[h][s_idx] = [a1_rand, a2_rand]
                
                if isinstance(self.learning_rule, MardenMoodRule):
                    hidd1_rand = np.random.choice(['C','D'])
                    hidd2_rand = np.random.choice(['C','D'])
                    self.hidden[h][s_idx] = [hidd1_rand, hidd2_rand]
                else:
                    self.hidden[h][s_idx] = [0.0, 0.0]


    def run(self):
        """ Run of the main learning cycle. """
        self._initialize()

        for t in range(self.T):

            V_t = np.copy(self.V)
            self.V_history.append(V_t[0, 1, 0])

            for h in range(self.game.H, 0, -1):
                
                # Actor: computes new actions and new auxiliary variables for all the states in stage h, using Q^(t)
                new_action_h = {}   #a_h_t_plus
                new_hidden_h = {}

                for s_str, s_idx in self.game.s_map[h].items():
                    current_a = self.a[h][s_idx]
                    current_hid = self.hidden[h][s_idx]

                    q_vals = self.Q[:, h, s_idx, :, :]
                             
                    new_action_h[s_idx], new_hidden_h[s_idx] = self.learning_rule.update_vars(current_a, current_hid, self.game.N, self.game.actions, q_vals)


                # Critic: updates V_{i,h} and Q_{i,h} for all the states in stage h, based on the new actions
                for s_str, s_idx in self.game.s_map[h].items():
                                        
                    # V-values update
                    t_joint_action = self.a[h][s_idx] 
                    for i in range(self.game.N):
                        q_val_t = self.Q[i, h, s_idx, t_joint_action[0], t_joint_action[1]]
                        
                        # Calcola la media mobile
                        if t == 0:
                           self.V[i, h, s_idx] = q_val_t
                        else:
                           old_v = V_t[i, h, s_idx]
                           self.V[i, h, s_idx] = (t / (t + 1)) * old_v + (1 / (t + 1)) * q_val_t

                    # Q-values update
                    for i in range(self.game.N):
                        for a1 in self.game.actions:
                            for a2 in self.game.actions:
                                expected_V = 0
                                if h < self.game.H: # Per h=1, calcola il valore atteso da h=2
                                    next_s_str = self.game.transition(a1, a2)
                                    next_s_idx = self.game.s_map[h + 1][next_s_str]
                                    expected_V = self.V[i, h + 1, next_s_idx]
                                
                                reward = self.game.rewards[h][s_str][a1,a2]
                                self.Q[i, h, s_idx, a1, a2] = reward[i] + expected_V

                # Save variables new values
                self.a[h] = new_action_h  
                self.hidden[h] = new_hidden_h              
            
            # Save history of the initial state
            action_in_s1 = self.a[1][0]         # action taken in h=1, s_idx=0

            self.s1_action_history.append(action_in_s1)


    def print_results(self):
        """ For each stage and state, prints the final V-values and Q-values learnt by player 0, and the pair of actions learnt. """
        print("\n--- Learnt Values  (Player 0)---")
        for h in range(1, self.game.H + 1):
            print(f"\n--- Stage h={h} ---")
            for s_str, s_idx in self.game.s_map[h].items():
                print(f"  State '{s_str}':")
                print(f"    V-value: {self.V[0, h, s_idx]:.4f}")
                print("    Q-values:")
                q_matrix = self.Q[0, h, s_idx, :, :]
                print("         a2=0    a2=1")
                print(f"    a1=0 [{q_matrix[0,0]:.2f}]  [{q_matrix[0,1]:.2f}]")
                print(f"    a1=1 [{q_matrix[1,0]:.2f}]  [{q_matrix[1,1]:.2f}]")
                print(f"    Learnt joint action (Policy): {[int(x) for x in self.a[h][s_idx]]}")

    
    def plot_convergence(self):
        """ Plots the convergence trajectory of the V-value of the initial state. """
        plt.figure(figsize=(10, 6))
        plt.plot(self.V_history)
        plt.xlabel("Iteration (t)")
        plt.ylabel("V(s1)")
        plt.title("Convergence of the V-value in the initial state 's1'")
        plt.grid(True)
        plt.show()


    def plot_policy_evolution(self,history, params):
        """
        Plots the policy evolution (just of the pair of actions (0,0) and (1,1)) according to the runs in hystory_of_all_runs,
        computed with empirical mean and a confidence interval at 60%.

        Args:
            history (list): List of lists. Each sublist contains the hystory of tuples of actions taken on each run.
            params (list): contains the parameters of the learning rule used to compute the hystory,
                                    - log learning: params = [epsilon]
                                    - marden mood:  params = [epsilon, c]
        """
        runs = np.array(self._normalize_runs(history))
        num_runs = len(runs)
        T = self.T

        freq_00 = np.zeros((num_runs, T))
        freq_11 = np.zeros((num_runs, T))
           
        for i in range(num_runs):
            count_00 = 0
            count_11 = 0
            
            for t in range(T):
                action = tuple(runs[i, t])
                if action == (0, 0):
                    count_00 += 1
                elif action == (1, 1):
                    count_11 += 1
                
                freq_00[i, t] = count_00 / (t + 1)
                freq_11[i, t] = count_11 / (t + 1)
                

        # Mean of frequences
        mean_freq_00 = np.mean(freq_00, axis=0)
        mean_freq_11 = np.mean(freq_11, axis=0)

        # CI of frequences from 20° to 80° percentile
        lower_bound_00 = np.percentile(freq_00, 20, axis=0)
        upper_bound_00 = np.percentile(freq_00, 80, axis=0)
        
        lower_bound_11 = np.percentile(freq_11, 20, axis=0)
        upper_bound_11 = np.percentile(freq_11, 80, axis=0)


        plt.figure(figsize=(12, 7))
        
        plt.plot(mean_freq_00, color='blue', label='Prob(a=(0,0) | s1)')
        plt.plot(mean_freq_11, color='orange', label='Prob(a=(1,1) | s1)')
        if num_runs>1:
            plt.fill_between(range(T), lower_bound_00, upper_bound_00, color='blue', alpha=0.2, 
                        label='CI 60% for (0,0)')
            plt.fill_between(range(T), lower_bound_11, upper_bound_11, color='orange', alpha=0.2,
                        label='CI 60% for (1,1)')
        
        if isinstance(params, float):
            eps = params
            plt.title(f"Evolution of policy in state s1, eps = {eps}")
        elif isinstance(params, tuple):
            eps, c = params
            plt.title(f"Evolution of policy in state s1, eps = {eps}, c = {c}")

        plt.xlabel("Iteration (t)")
        plt.ylabel("Empirical probability of actions")
        plt.ylim(0, 1)
        plt.xlim(0, T)
        plt.legend()
        plt.grid(True)
        plt.show()

    def _normalize_runs(self, history):
        """
        Accepts:
            - single trajectory: List[List]
            - list of trajectories: List[List[List]]
        Returns:
            List[List[List]]
        """
        if len(history) == 0:
            raise ValueError("Empty history.")

        if isinstance(history[0][0], (numbers.Number,str)):
            return [history]
        return history
    