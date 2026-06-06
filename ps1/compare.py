import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

from vi import value_iteration
from pi import policy_iteration

if __name__ == "__main__":
    GAMMA = [0.5, 0.9, 0.99, 0.999]
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P

    n_value = []
    n_policy = []

    for gamma in GAMMA:
        start = time.time()
        val, pol, n = value_iteration(P, gamma, 1e-4)
        delta = time.time() - start

        n_value.append(n)

        print(f"Value Iteration: {n} Iters, {delta * 1000:.2f} ms")

        start = time.time()
        val, pol, n = policy_iteration(P, gamma)
        delta = time.time() - start

        n_policy.append(n)

        print(f"Value Iteration: {n} Iters, {delta * 1000:.2f} ms")

    mpl.use("tkagg")

    plt.scatter(GAMMA, n_value, label="Value Iteration")
    plt.scatter(GAMMA, n_policy, label="Policy Iteration")

    plt.legend()
    plt.xlabel("$\\gamma$")
    plt.ylabel("Iterations")
    plt.show()
