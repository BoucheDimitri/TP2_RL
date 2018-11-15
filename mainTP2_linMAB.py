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

T = 200

lamb = 0.01
alphas = np.ones((T, ))

actions, rewards, thetas = funcs_linmab.lin_UCB(model, T, lamb, alphas, eps=0)

ntrajs = 50

regret_ucb, norms_ucb = funcs_linmab.mc_regret_norm_UCB1(ntrajs, model, T, lamb, alphas, eps=0)

eps = 0.1
regret_greedy, norms_greedy = funcs_linmab.mc_regret_norm_UCB1(ntrajs, model, T, lamb, alphas, eps=eps)

regret_random = funcs_linmab.mc_regret_random(ntrajs, model, T)
