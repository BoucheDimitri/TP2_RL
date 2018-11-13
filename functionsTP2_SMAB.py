import numpy as np
import arms


def create_bernouilli_MAB(pks):
    MAB = []
    for k in range(0, pks.shape[0]):
        MAB.append(arms.ArmBernoulli(pks[k], random_state=np.random.randint(1, 312414)))
    return MAB


def get_N_S(rewards, actions, t):
    N = np.sum(actions[:t, :], axis=0)
    S = np.sum(rewards[:t, :], axis=0)
    return N, S


def compute_mus(N, S):
    mus = S / N
    mus[mus == np.nan] = np.inf
    return mus


def confidence_bound(mu, t, N, rho):
    return mu + rho * np.sqrt(np.log(t) / (2 * N))


def UCB1(T, MAB, rhoseq):
    k = len(MAB)
    rewards = np.zeros((T, k))
    actions = np.zeros((T, k))
    for t in range(0, k):
        actions[t, t] = 1
        rewards[t, t] = MAB[t].sample()
    for t in range(k, T):
        N, S = get_N_S(rewards, actions, t)
        mus = compute_mus(N, S)
        a = np.argmax(confidence_bound(mus, t, N, rhoseq[t]))
        actions[t, a] = 1
        rewards[t, a] = MAB[a].sample()
    return actions, rewards


def draw_betas(N, S):
    k = N.shape[0]
    pi = np.zeros((k, ))
    for a in range(0, k):
        pi[a] = np.random.beta(S[a] + 1, N[a] - S[a] + 1)
    return pi


def TS(T, MAB):
    k = len(MAB)
    rewards = np.zeros((T, k))
    actions = np.zeros((T, k))
    actions[0, 0] = 1
    rewards[0, 0] = MAB[0].sample()[0]
    for t in range(1, T):
        N, S = get_N_S(rewards, actions, t)
        pi = draw_betas(N, S)
        a = np.argmax(pi)
        actions[t, a] = 1
        rewards[t, a] = MAB[a].sample()
    return actions, rewards


def collect_trajs_UCB1(T, MAB, rhoseq, ntrajs):
    k = len(MAB)
    rewards_tensor = np.zeros((T, k, ntrajs))
    for traj in range(0, ntrajs):
        actions, rewards = UCB1(T, MAB, rhoseq)
        rewards_tensor[:, :, traj] = rewards
        print(traj)
    return rewards_tensor


def collect_trajs_TS(T, MAB, ntrajs):
    k = len(MAB)
    rewards_tensor = np.zeros((T, k, ntrajs))
    for traj in range(0, ntrajs):
        actions, rewards = TS(T, MAB)
        rewards_tensor[:, :, traj] = rewards
        print(traj)
    return rewards_tensor


def mc_rewards_expectation(rewards_tensor):
    return np.mean(np.cumsum(np.sum(rewards_tensor, axis=1), axis=0), axis=1)


def mc_regret_UCB1(T, MAB, rhoseq, ntrajs):
    k = len(MAB)
    rewards_tensor = collect_trajs_UCB1(T, MAB, rhoseq, ntrajs)
    means = np.array([MAB[a].mean for a in range(0, k)])
    best = np.max(means) * np.arange(1, T + 1)
    mc_expec = mc_rewards_expectation(rewards_tensor)
    return best - mc_expec


def mc_regret_TS(T, MAB, ntrajs):
    k = len(MAB)
    rewards_tensor = collect_trajs_TS(T, MAB, ntrajs)
    means = np.array([MAB[a].mean for a in range(0, k)])
    best = np.max(means) * np.arange(1, T + 1)
    mc_expec = mc_rewards_expectation(rewards_tensor)
    return best - mc_expec


def kulback_bernouilli(p1, p2):
    return p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))


def complexity_bernouilli(MAB):
    k = len(MAB)
    means = np.array([MAB[a].mean for a in range(0, k)])
    best = np.max(means)
    inds = np.argwhere(best - means > 0)
    kls = np.apply_along_axis(func1d=lambda p: kulback_bernouilli(p, best), arr=means[inds], axis=0)
    print(kls)
    best_minus_means = best - means
    return np.sum(best_minus_means[inds] / kls)


def oracle_bound(T, MAB):
    comp = complexity_bernouilli(MAB)
    return np.log(np.arange(1, T+1)) * comp


def TS_nonbinary(T, MAB):
    k = len(MAB)
    rewards = np.zeros((T, k))
    actions = np.zeros((T, k))
    bernoulli_trials = np.zeros((T, k))
    actions[0, 0] = 1
    rewards[0, 0] = MAB[0].sample()[0]
    if int(rewards[0, 0]) == 1 or int(rewards[0, 0]) == 0:
        bernoulli_trials[0, 0] = rewards[0, 0]
    else:
        bernoulli_trials[0, 0] = np.random.binomial(2, rewards[0, 0], 1)
    print(bernoulli_trials[0, 0])
    for t in range(1, T):
        N, S = get_N_S(bernoulli_trials, actions, t)
        pi = draw_betas(N, S)
        a = np.argmax(pi)
        actions[t, a] = 1
        rewards[t, a] = MAB[a].sample()[0]
        if int(rewards[t, a]) == 1 or int(rewards[t, a]) == 0:
            bernoulli_trials[t, a] = rewards[t, a]
        else:
            bernoulli_trials[t, a] = np.random.binomial(2, rewards[t, a], 1)
    return actions, rewards, bernoulli_trials


def collect_trajs_TS_nonbinary(T, MAB, ntrajs):
    k = len(MAB)
    rewards_tensor = np.zeros((T, k, ntrajs))
    for traj in range(0, ntrajs):
        actions, rewards, bernouilli_trials = TS_nonbinary(T, MAB)
        rewards_tensor[:, :, traj] = rewards
        print(traj)
    return rewards_tensor


def mc_regret_TS_nonbinary(T, MAB, ntrajs):
    k = len(MAB)
    rewards_tensor = collect_trajs_TS_nonbinary(T, MAB, ntrajs)
    means = np.array([MAB[a].mean for a in range(0, k)])
    best = np.max(means) * np.arange(1, T + 1)
    mc_expec = mc_rewards_expectation(rewards_tensor)
    return best - mc_expec


