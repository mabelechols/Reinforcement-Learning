import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from env import ApartmentEnv
from policies import RandomPolicy, ThresholdPolicy, OptimalPolicy


def RunPolicyEpisode(env, policy, n_max, *args, **kwargs):
    obs, _ = env.reset()
    action = policy(obs, *args, **kwargs)

    for _ in range(n_max):
        obs, rew, term, _, _ = env.step(action)

        if term:
            return (rew, rew == 0)

        action = policy(obs, *args, **kwargs)

    return (0, True)


def TestPolicy(env, policy, n_max, N, *args, **kwargs):
    rewards = np.zeros((N,), dtype=np.int32)
    reject_all = np.zeros((N,), dtype=bool)

    for i in range(N):
        reward, rejected = RunPolicyEpisode(env, policy, n_max, *args, **kwargs)

        rewards[i] = reward
        reject_all[i] = rejected

    return (
        np.mean(rewards),
        np.std(rewards) / np.sqrt(rewards.size),
        np.mean(reject_all),
        rewards,
    )


if __name__ == "__main__":
    N = 10000
    T = 4
    K = 4

    SIGMA = 3

    gymnasium.register(
        id="CS5180/PS0_ApartmentEnv",
        entry_point=ApartmentEnv,
    )
    env = gymnasium.make("CS5180/PS0_ApartmentEnv", T=T, K=K, sigma=SIGMA)

    # Random Policy
    mean_random, se_random, reject_random, reward_random = TestPolicy(
        env, RandomPolicy, T + 1, N
    )

    # Threshold Policy
    THRESHOLDS = [1, 2, 3, 4]
    means_threshold, ses_threshold, rejects_threshold, rewards_threshold = (
        np.zeros((len(THRESHOLDS),)),
        np.zeros((len(THRESHOLDS),)),
        np.zeros((len(THRESHOLDS),)),
        np.zeros((len(THRESHOLDS), N)),
    )

    for i, u_min in enumerate(THRESHOLDS):
        mu, se, r, rew = TestPolicy(env, ThresholdPolicy, T + 1, N, u_min=u_min)

        means_threshold[i] = mu
        ses_threshold[i] = se
        rejects_threshold[i] = r
        rewards_threshold[i, :] = rew

    # Optimal Policy
    mean_optimal, se_optimal, reject_optimal, reward_optimal = TestPolicy(
        env, OptimalPolicy, T + 1, N
    )

    # Report Values

    # Mean utility
    print("Mean Utility\n------------")
    print(f"\tRandomPolicy: {mean_random:.2f} | SE: {se_random:.2f}")
    print(f"\tThresholdPolicy")
    for i, u_min in enumerate(THRESHOLDS):
        print(
            f"\t\tu_min={u_min}: {means_threshold[i]:.2f} | SE: {ses_threshold[i]:.2f}"
        )
    print(f"\tOptimalPolicy: {mean_optimal:.2f} | SE: {se_optimal:.2f}")

    # Fraction of reject all
    print("\nReject All Fraction\n-------------------")
    print(f"\tRandomPolicy: {reject_random:.3f}")
    print(f"\tThresholdPolicy")
    for i, u_min in enumerate(THRESHOLDS):
        print(f"\t\tu_min={u_min}: {rejects_threshold[i]:.3f}")
    print(f"\tOptimalPolicy: {reject_optimal:.3f}")

    # Histograms
    mpl.use("tkagg")

    BIN_EDGES = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]

    plt.hist(reward_random, BIN_EDGES)
    plt.xticks([0, 1, 2, 3, 4])
    plt.xlabel("Utility")
    plt.ylabel("Count")
    plt.title("Utility Histogram (RandomPolicy)")

    plt.figure()

    for i in range(len(THRESHOLDS)):
        plt.hist(rewards_threshold[i, :], BIN_EDGES)
        plt.xticks([0, 1, 2, 3, 4])
        plt.xlabel("Utility")
        plt.ylabel("Count")
        plt.title(f"Utility Histogram (ThresholdPolicy [u_min={THRESHOLDS[i]}])")

        plt.figure()

    plt.hist(reward_optimal, BIN_EDGES)
    plt.xticks([0, 1, 2, 3, 4])
    plt.xlabel("Utility")
    plt.ylabel("Count")
    plt.title(f"Utility Histogram (OptimalPolicy)")

    plt.show()
