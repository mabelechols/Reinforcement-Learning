import numpy as np


def value_iteration(P, gamma, theta):
    # Implementing this naively for readability, this is also able to be sped up trivially with numpy operations
    def max_Q(state, value):
        max_Q = None
        max_act = None

        for action in range(n_actions):
            transition = P[state][action]
            q = 0

            for i in range(len(transition)):
                prob, new_state, reward, _ = transition[i]

                q += prob * (reward + gamma * value[new_state])

            if max_Q is None or q > max_Q:
                max_Q = q
                max_act = action

        return (max_Q, max_act)

    n_states = len(P)
    n_actions = len(P[0])

    value = np.zeros((n_states,))
    new_value = np.zeros((n_states,))

    eps = theta * (1 - gamma) / gamma

    n_iters = 0

    while True:
        # Update Step
        n_iters += 1
        for state in range(n_states):
            new_value[state] = max_Q(state, value)[0]

        # Check for termination
        delta = np.abs(new_value - value)

        policy = np.zeros((n_states,), dtype=int)
        for state in range(n_states):
            policy[state] = max_Q(state, new_value)[1]

        if np.max(delta) < eps:
            policy = np.zeros((n_states,), dtype=int)
            for state in range(n_states):
                policy[state] = max_Q(state, new_value)[1]

            return (new_value, policy, n_iters)

        value = new_value.copy()
