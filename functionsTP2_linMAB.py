import numpy as np


def update_A_and_b(A, b, r, phis, a):
    phia = phis[a, ]
    A += np.tensordot(phia, phia, 0)
    b += r * phis[a, ]
    return A, b


def compute_beta(Ainv, alpha, phis):
    n_a = phis.shape[0]
    betas = np.zeros((n_a, ))
    for a in range(0, n_a):
        betas[a] = alpha * np.sqrt(np.dot(np.dot(phis[a, :], Ainv), phis[a, :].T))
    return betas


def lin_UCB(model, T, lamb, alphas, eps=0):
    d = model.n_features
    n_a = model.n_actions
    A = lamb * np.eye(d)
    b = np.zeros((d, ))
    rewards = np.zeros((T, n_a))
    actions = np.zeros((T, n_a))
    thetas = np.zeros((d, T - 1))
    for t in range(0, T):
        if t < n_a:
            a = t
        else:
            greedy_ind = np.random.binomial(2, eps)
            if greedy_ind == 1:
                a = np.random.randint(0, n_a)
            else:
                a = np.argmax(np.dot(model.features, thetas[:, t - 1]) + betas)
        actions[t, a] = 1
        rewards[t, a] = model.reward(a)
        A, b = update_A_and_b(A, b, rewards[t, a], model.features, a)
        Ainv = np.linalg.inv(A)
        betas = compute_beta(Ainv, alphas[t], model.features)
        if t < T - 1:
            thetas[:, t] = np.linalg.solve(A, b)
    return actions, rewards, thetas


def random_strat(model, T, lamb):
    d = model.n_features
    n_a = model.n_actions
    A = lamb * np.eye(d)
    b = np.zeros((d, ))
    rewards = np.zeros((T, n_a))
    actions = np.zeros((T, n_a))
    thetas = np.zeros((d, T - 1))
    for t in range(0, T):
        a = np.random.randint(0, n_a)
        actions[t, a] = 1
        rewards[t, a] = model.reward(a)
        A, b = update_A_and_b(A, b, rewards[t, a], model.features, a)
        if t < T - 1:
            thetas[:, t] = np.linalg.solve(A, b)
    return actions, rewards, thetas


def collect_UCB_trajs(ntrajs, model, T, lamb, alphas, eps=0):
    n_a = model.n_actions
    d = model.n_features
    rewards_tensor = np.zeros((T, n_a, ntrajs))
    norms_matrix = np.zeros((T - 1, ntrajs))
    for traj in range(0, ntrajs):
        actions, rewards, thetas = lin_UCB(model, T, lamb, alphas, eps)
        norms_matrix[:, traj] = np.linalg.norm(model.real_theta.reshape((d, 1)) - thetas, 2, axis=0)
        rewards_tensor[:, :, traj] = rewards
        print(traj)
    return rewards_tensor, norms_matrix


def collect_random_trajs(ntrajs, model, T, lamb):
    n_a = model.n_actions
    d = model.n_features
    rewards_tensor = np.zeros((T, n_a, ntrajs))
    norms_matrix = np.zeros((T - 1, ntrajs))
    for traj in range(0, ntrajs):
        actions, rewards, thetas = random_strat(model, T, lamb)
        norms_matrix[:, traj] = np.linalg.norm(model.real_theta.reshape((d, 1)) - thetas, 2, axis=0)
        rewards_tensor[:, :, traj] = rewards
        print(traj)
    return rewards_tensor, norms_matrix


def mc_rewards_expectation(rewards_tensor):
    return np.mean(np.sum(np.cumsum(rewards_tensor, axis=0), axis=1), axis=1)


def mc_regret_norm_UCB1(ntrajs, model, T, lamb, alphas, eps=0):
    rewards_tensor, norms_matrix = collect_UCB_trajs(ntrajs, model, T, lamb, alphas, eps)
    mc_expec = mc_rewards_expectation(rewards_tensor)
    best = model.best_arm_reward() * np.arange(1, T + 1)
    return best - mc_expec, np.mean(norms_matrix, axis=1)


def mc_regret_random(ntrajs, model, T, lamb):
    rewards_tensor, norms_matrix = collect_random_trajs(ntrajs, model, T, lamb)
    mc_expec = mc_rewards_expectation(rewards_tensor)
    best = model.best_arm_reward() * np.arange(1, T + 1)
    return best - mc_expec, np.mean(norms_matrix, axis=1)
