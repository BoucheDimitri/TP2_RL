import numpy as np
import arms
import functionsTP2_SMAB as funcs_smab
import matplotlib.pyplot as plt
import importlib

importlib.reload(funcs_smab)

# Plot parameters
plt.rcParams.update({"font.size": 30})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})


# ####################### Question 1: Bernoulli Bandits ##############################################################

# Means of the MAB's Bernoullis
# pks = np.array([0.1, 0.2, 0.5, 0.3, 0.45])
pks = np.array([0.55, 0.5, 0.45, 0.4, 0.6])

# Define MAB model
MAB = funcs_smab.create_bernouilli_MAB(pks)

# Time horison
T = 5000

# Learning parameters
rhoseq = np.ones((T, ))

# Number of trajectories for MC regret computation
ntrajs = 1000

# Compute regrets by MC
regret_ucb = funcs_smab.mc_regret_UCB1(T, MAB, rhoseq, ntrajs)
regret_ts = funcs_smab.mc_regret_TS(T, MAB, ntrajs)

# Oracle bound
oracle = funcs_smab.oracle_bound(T, MAB)

# Plot the results
plt.figure()
plt.plot(regret_ucb, label="UCB1")
plt.plot(regret_ts, label="TS")
plt.plot(oracle, label="Oracle")
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.title("Estimated regret comparison - Bernouilli case")


# ####################### Question 2: Non parametric Bandits #########################################################

# Define non parametric MAB
arm1 = arms.ArmBernoulli(0.2, random_state=np.random.randint(1, 312414))
arm2 = arms.ArmBeta(2, 10, random_state=np.random.randint(1, 312414))
arm3 = arms.ArmBeta(5, 15, random_state=np.random.randint(1, 312414))
arm4 = arms.ArmBernoulli(0.35, random_state=np.random.randint(1, 312414))
MAB2 = [arm1, arm2, arm3, arm4]

# Time horizon
T = 5000

# Number of trajectories for MC estimation
ntrajs = 1000

# Learning rates
rhoseq = np.ones((T, ))

# Compute regrets
regret_ucb = funcs_smab.mc_regret_UCB1(T, MAB2, rhoseq, ntrajs)
regret_ts_nonbin = funcs_smab.mc_regret_TS_nonbinary(T, MAB2, ntrajs)

# Plot the results
plt.figure()
plt.plot(regret_ucb, label="UCB1")
plt.plot(regret_ts_nonbin, label="TS non binary")
plt.legend()
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.title("Estimated regret comparison - Non parametric case")