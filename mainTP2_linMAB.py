import numpy as np
from linearmab_models import ToyLinearModel, ColdStartMovieLensModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import importlib

import functionsTP2_linMAB as funcs_linmab
importlib.reload(funcs_linmab)

random_state = np.random.randint(0, 375551485)
# model = ToyLinearModel(
#     n_features=8,
#     n_actions=5,
#     random_state=random_state,
#     noise=0.1)

model = ColdStartMovieLensModel(
    random_state=random_state,
    noise=0.1
)

# Number of movies
T = 207

# Regularization
lamb = 0.01

# Alpha
alphas = np.ones((T, ))

# Number of trajectories for MC simulations
ntrajs = 500


# MC estimation of regret and distance to theta* for UCB
regret_ucb, norms_ucb = funcs_linmab.mc_regret_norm_UCB1(ntrajs, model, T, lamb, alphas, eps=0)

# MC estimation of regret and distance to theta* for greedy policy
eps = 0.2
regret_greedy, norms_greedy = funcs_linmab.mc_regret_norm_UCB1(ntrajs, model, T, lamb, alphas, eps=eps)

# MC estimation of regret for random
regret_random = funcs_linmab.mc_regret_random(ntrajs, model, T)

# Plot the results
fig, axes = plt.subplots(ncols = 2)
axes[0].plot(regret_ucb, label="UCB")
axes[0].plot(regret_greedy, label="Greedy")
axes[0].plot(regret_random, label="Random")
axes[0].set_xlabel("t")
axes[0].set_ylabel("Empirical regret")
axes[0].legend()
axes[1].plot(norms_ucb, label="UCB")
axes[1].plot(norms_greedy, label="Greedy")
axes[1].set_xlabel("t")
axes[1].set_ylabel("Distance l2 to true preference vector")
axes[1].legend()