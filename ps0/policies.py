import numpy as np


def RandomPolicy(observation, T=4):
    r = np.random.random()

    if r <= 1 / T:
        return 0
    else:
        return 1


def ThresholdPolicy(observation, u_min):
    U_t = observation["U_t"]

    if U_t >= u_min:
        return 0
    else:
        return 1


def OptimalPolicy(observation):
    THRESH = [3.25, 3, 3, 0]

    t, U_t = observation["t"], observation["U_t"]

    if (
        U_t >= THRESH[t - 1]
    ):  # This abuses t=0 being the terminal state in that THRESH[-1] is defined and accepting/rejecting are the same
        return 0
    return 1
