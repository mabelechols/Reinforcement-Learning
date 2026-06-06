import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from vi import value_iteration
from pi import policy_iteration


def state_to_coords(state):
    col = state % 4
    row = state // 4

    return (row, col)


def print_value(value):
    print("State: (X,Y) | Value\n-------------|------")
    for i in range(value.size):
        coords = state_to_coords(i)
        print(f"    ({coords[1]},{coords[0]})    |  {value[i]:.2f}")


def render_to_grid(policy, show=True, title=None):
    DIRS = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    L = 0.5

    fig, ax = plt.subplots()

    ax.grid(True)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)

    for i in range(policy.size):
        coords = state_to_coords(i)

        dir = DIRS[policy[i]]
        cent = (0.5 + coords[1], 0.5 + coords[0])

        ax.annotate(
            "",
            xytext=(cent[0] - L / 2 * dir[0], cent[1] - L / 2 * dir[1]),
            xy=(cent[0] + L / 2 * dir[0], cent[1] + L / 2 * dir[1]),
            arrowprops=dict(arrowstyle="->"),
        )

    if show:
        plt.show()


if __name__ == "__main__":
    mpl.use("tkagg")

    env = gym.make("FrozenLake-v1", is_slippery=True)

    val, pol, n = value_iteration(env.unwrapped.P, 0.99, 1e-4)

    print(f"Value Iteration\n---------------\n\t{n} Iterations\n")

    print_value(val)
    render_to_grid(pol, show=False, title="Optimal Policy (Value Iteration)")

    val, pol, n = policy_iteration(env.unwrapped.P, 0.99)

    print(f"\nPolicy Iteration\n---------------\n\t{n} Iterations\n")

    print_value(val)
    render_to_grid(pol, title="Optimal Policy (Policy Iteration)")
