# Equilibrium Selection for Multi-agent Reinforcement Learning: A Unified Framework

<sub>_Final project for the Reinforcement Learning course, developed together with Ludovica Bianchi._</sub>

This work implements what proposed in [Equilibrium Selection for Multi-agent Reinforcement Learning: A Unified Framework](https://arxiv.org/abs/2406.08844), written by Runyu Zhang, Jeff Shamma, Na Li. The paper addresses the problem of equilibrium selection in multi-agent reinforcement learning (MARL). 

While many MARL algorithms ensure convergence to some Nash equilibrium (NE), these equilibria can differ significantly in terms of rewards and social welfare. Building on insights from classical game theory, where decentralized learning can select potential-maximizing or Pareto-optimal NEs in normal-form games, we implement a unified framework for equilibrium selection in stochastic games. The framework is modular and adaptable, allowing established learning rules and selection results from normal-form games to be extended to the MARL setting.

<img src="img/algorithm1.png" alt="Learning algorithm" width="50%">

We apply this framework to two normal-form games with two players and two stages $h = 1,2$, showing the convergence with two different learning rules.

## Learning Rules
For equilibrium selection, we assume that the group of agents follows certain iterative learning rules, which describe how they respond to the reward outcome from the previous action. In particular this is represented through a transition kernel $K^\epsilon$:

$$
(a^{(t+1)}, \xi^{(t+1)}) \sim K^\epsilon(\cdot,\cdot \mid a^{(t)},\xi^{(t)}),
$$
      
where $\xi \in \mathcal{E}$ are auxiliary variables, that depend on the rule, and $\epsilon \in (0,1)$ is a parameter that represents the ‘rate of mistakes’, assuming that agents does not respond in a fully rational and accurate way towards their observation of the reward outcome (rationality comes when $\epsilon \rightarrow 0$). 
     

### Log-linear Learning Rule
For this learning rule, there is no need for the auxiliary variables, $\mathcal{E} = \emptyset$. The kernel determines the new action as:

$$
K^\epsilon \left(a^{(t+1)} = (a_{-i}^{(t)},a_{i}^{(t+1)})\mid a^{(t)} ; \{r_i\}_{i=1}^n \right) = \frac{1}{n} \frac{\epsilon^{-r_i(a^{(t+1)})}}{\sum_{a'_i}{\epsilon^{-r_i(a'_i,a_{-i}^{(t)})}}}
$$

### Marden Mood Learning Rule
For this learning rule, the hidden variable $\xi$ takes the form $\xi = (\xi_1, \dots, \xi_n)$, where $\xi_i \in \{ C,D \}$ represents the mood of agent *i* (C stands for Content, D for Discontent). The updating rule is the following.

*Action dynamics:* 

$$
\begin{cases}
    \text{if } \xi_i^{(t)} = D \rightarrow a_i^{(t+1)} \sim \text{Unif}(\mathcal{A}_i)\\ 
    \\
    \text{if } \xi_i^{(t)} = C \rightarrow a_i^{(t+1)} \begin{cases}
        = a_i^{(t)} &\text{with prob } 1 - \epsilon^c\\
        \sim \text{Unif}(\mathcal{A}_i\backslash \{a_i^{(t)}\}) &\text{with prob } \epsilon^c
    \end{cases} 
\end{cases}
$$

*Mood dynamics:*

$$
\begin{cases} 
    \text{if } \xi_i^{(t)} = C \text{ and } a_i^{(t+1)} = a_i^{(t)} \rightarrow \xi_i^{(t+1)} = C \\ 
    \\ 
    \text{else } \rightarrow \xi_i^{(t+1)} = \begin{cases} 
        C &\text{with prob } \epsilon^{1-r_i(a^{(t+1)})} \\ 
        D &\text{otherwise} 
    \end{cases} 
\end{cases}
$$


**Notice:** In the case of the Marden Mood learning rule, the term $\epsilon^{1-r_i(a^{(t+1)})}$ must be a valid probability. For this reason, the implementation takes care of the rewards normalisation. On the contrary, the Log-linear learning rule employs a softmax update, which inherently normalizes the rewards; applying normalization beforehand would result in an additional compression of the value range.

## Games

### Treasure Game

This identical interest game captures the process of two players collaboratively digging treasure at two different locations 0 and 1. Each location has a shallow level and a deep level, and it requires the players to collaboratively dig at the same location to make progress. 

- Location 0 has reward 1 at the shallow level and 0.5 at the deep level. 
- Location 1 has reward 0 at the shallow level and 2 at the deep level. 

<img src="img/treasure_game.png" alt="Description of the Treasure Game" width="35%">

The process can be summarized as a stochastic game. For h = 1, there’s only one state and thus the stage reward is given by a 2 by 2 reward matrix with nonzero reward of value 1 only when a = (0,0). The transition to the second stage follows the following transition rule: 
- if both players 1 and 2 choose to dig at location 0 ($a_1 = a_2 = 0$), then the game will transit to state $s = A$, where they can arrive at the deep level for location 0 and shallow level at location 1;
- if $a_1 = a_2 = 1$ then the game will transit to $s = B$, where they can arrive at the deep level for location 1 and shallow level at location 0;
- if the players fail to agree on a location, ($a = (0,1) or (1,0)$), they make no progress ($s = O$) and the reward will remain the same as stage $h = 1$.

<img src="img/treasure_rewards.png" alt="Description of the Treasure Game rewards" width="50%">

In this game, there are two strict NEs:
1. agreeing twice on location 0, i.e. $a_{i,h} = 0$ for $h = 1,2$ and $i = 1,2$; 
2. agreeing twice on location 1, i.e. $a_{i,h} = 1$ for $h = 1,2$ and $i = 1,2$. 

Notice that the second NE is also the global optimal policy that gives a total reward 2, while the first only gives reward 1.5. 

We are going to show that for log-linear learning, the stochastically stable policy is the global optimal policy, thus it should converge to the second instead of the first NE.

### Stag Hunt game

This coordination game models the process of two players choosing whether to hunt a stag or a hare over two stages. Hunting the stag is harder, and thus requires both players to fully cooperate and to keep hunting persistently for two stages to success, but it has a high reward (7.5 allocated evenly to both players). Hunting the hare instead is easier and can be done individually with one stage, but also with lower payoff: the value is 2, eventually to be split if both players decide to catch the hare.

The process can be summarized as a stochastic game:

<img src="img/staghunt_rewards1.png" alt="Description of the Stag Hunt Game rewards" width="50%">

In this game, there are two strict Nash equilibria:

1. Stag NE: both players choose Stag at both stages ($a_{i,h} = 0$ for $h = 1,2$ and $i = 1,2$), yielding the Pareto-dominant outcome with total reward 7.5.
2. Hare NE: both players choose Hare at both stages ($a_{i,h} = 1$ for $h = 1,2$ and $i = 1,2$), yielding the risk-dominant outcome with total reward 2.

While the Stag equilibrium gives the higher social welfare, the Hare equilibrium is risk-dominant. We are going to show that running log-linear learning in this setting leads to convergence to the Hare equilibrium, while the Marden Mood learning rule favors the Pareto-optimal Stag equilibrium.

---

## Results
We reproduced the results presented in the paper, as shown in the following plots.

### Treasure Game

**Log-linear learning rule**  
$\epsilon = 0.01$, $50.000$ iterations, 100 runs.

<img src="img/Treasure_loglinear_EPS=0.01_T50_000_R100.png" alt="Log-linear learning results on the Treasure Game" width="75%">

### Stag Hunt Game

**Log-linear learning rule**  
$\epsilon = 0.01$, $100.000$ iterations, 100 runs.

<img src="img/StagHunt_loglinear_EPS=0.01_T100_000_R100.png" alt="Log-linear learning results on the Stag Hunt Game" width="75%">

**Marden Mood learning rule**  
$\epsilon = 0.01, c = 2.0$, 1M iterations, 100 runs.

<img src="img/StagHunt_marden_EPS=0.01_C=2_T=1M_R=100.png" alt="Marden Mood learning results on the Stag Hunt Game" width="75%">

---


## Project Structure

```
.
├── img/                                    # Images for documentation
├── notebooks/                              # Jupyter notebooks for experimentation
│   ├── Equilibrium_selection_MARL.ipynb    # Reproduction of paper's results
├── src/                                    # Source code for the project
│   ├── game.py                             # Games implementation
│   ├── learning_rule.py                    # Learning rules implementation
│   ├── main.py                             # Entry point for running the experiments
│   └── unified_learning.py                 # MARL framework implementation
├── slides.pdf                              # Project presentation slides
├── requirements.txt                        # List of required Python dependencies
└── README.md                               # You're here!
```

---

## How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/silviacalabretta/marl-equilibrium-selection
cd marl-equilibrium-selection
```

### 2. Set up Virtual Environment (Recommended)

```bash
python -m venv .env
source .env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Default Example:

You can run experiments by executing

```bash
python -m src.main
```

and providing the following optional arguments:

* **`--iterations`** (int): total number of learning steps the simulation will run [default: 1000];
* **`--game`** (str): `"treasure"` or `"staghunt"` [default: `"treasure"`];
* **`--learning-rule`** (str):`"loglinear"` or `"mardenmood"` [default: `"loglinear"`];
* **`--rule-coeffs`** (float): parameters for the chosen learning rule [default: 0.01];
* **`--num-runs`** (int): the number of independent learning simulations to execute (each run executes the learning process for `--iterations` steps) [default: 1];

Example:

```bash
python -m src.main --iterations 2000 --game staghunt --learning-rule mardenmood --rule-coeffs 0.01 2
```

---

## References

* **Zhang, R., Shamma, J., & Li, N. (2024).**
  *Equilibrium Selection for Multi-agent Reinforcement Learning: A Unified Framework.*
  arXiv preprint, arXiv:[2406.08844](https://arxiv.org/abs/2406.08844).

* **Blume, L. E. (1993).**
  *The Statistical Mechanics of Strategic Interaction.*
  Games and Economic Behavior, 5(3), 387–424.

* **Marden, J. R., Young, H. P., & Pao, L. Y. (2012).**
  *Achieving Pareto Optimality Through Distributed Learning.*
  In Proceedings of the 51st IEEE Conference on Decision and Control (CDC), 7419–7424.
  [https://doi.org/10.1109/CDC.2012.6426834](https://doi.org/10.1109/CDC.2012.6426834)

* **Young, H. P. (1993).**
  *The Evolution of Conventions.*
  Econometrica, 61(1), 57–84.

* **Marden, J. R., & Shamma, J. S. (2012).**
  *Revisiting Log-linear Learning: Asynchrony, Completeness and Payoff-based Implementation.*
  Games and Economic Behavior, 75(2), 788–808.
