import numpy as np
import arms
import functionsTP2_SMAB as funcs_smab
import matplotlib.pyplot as plt
import importlib

importlib.reload(funcs_smab)


# ####################### Question 1: Bernoulli Bandits ##############################################################
pks = np.array([0.1, 0.2, 0.5, 0.3, 0.45])
MAB = funcs_smab.create_bernouilli_MAB(pks)
T = 5000
rhoseq = np.ones((T, ))
actions_ucb, rewards_ucb = funcs_smab.UCB1(T, MAB, rhoseq)
actions_ts, rewards_ts = funcs_smab.TS(T, MAB)

ntrajs = 100
regret_ucb = funcs_smab.mc_regret_UCB1(T, MAB, rhoseq, ntrajs)
regret_ts = funcs_smab.mc_regret_TS(T, MAB, ntrajs)

oracle = funcs_smab.oracle_bound(T, MAB)

plt.figure()
plt.plot(regret_ucb, label="UCB1")
plt.plot(regret_ts, label="TS")
plt.plot(oracle, label="Oracle")
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.legend()


# ####################### Question 2: Non parametric Bandits #########################################################
arm1 = arms.ArmBernoulli(0.2, random_state=np.random.randint(1, 312414))
arm2 = arms.ArmBeta(2, 10, random_state=np.random.randint(1, 312414))
arm3 = arms.ArmBeta(5, 15, random_state=np.random.randint(1, 312414))
arm4 = arms.ArmBernoulli(0.35, random_state=np.random.randint(1, 312414))

MAB2 = [arm1, arm2, arm3, arm4]

T = 5000
ntrajs = 100

rhoseq = np.ones((T, ))

regret_ucb = funcs_smab.mc_regret_UCB1(T, MAB2, rhoseq, ntrajs)
regret_ts_nonbin = funcs_smab.mc_regret_TS_nonbinary(T, MAB2, ntrajs)


plt.figure()
plt.plot(regret_ucb, label="UCB1")
plt.plot(regret_ts_nonbin, label="TS non binary")
plt.legend()
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')

