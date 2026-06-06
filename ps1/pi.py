import numpy as np


def policy_iteration(P, gamma):
    n_states = len(P)
    n_actions = len(P[0])

    policy = np.zeros((n_states,), dtype=int)
    new_policy = np.zeros((n_states,), dtype=int)

    n_iters = 0

    while True:
        n_iters += 1

        kern = np.zeros((n_states, n_states))
        rew = np.zeros((n_states,))

        for old_state in range(n_states):
            # construct P_pi, R_pi
            action = policy[old_state]

            transitions = P[old_state][action]
            R = 0

            for i in range(len(transitions)):
                prob, new_state, reward, _ = transitions[i]

                kern[old_state][new_state] += prob
                R += prob * reward

            rew[old_state] = R

        # Solve linear system
        value = np.linalg.solve(np.eye(n_states) - gamma * kern, rew)

        # Policy improvement
        for state in range(n_states):
            max_Q = None
            max_act = None

            for action in range(n_actions):
                transition = P[state][action]
                q = 0

                for i in range(len(transition)):
                    prob, new_state, reward, _ = transition[i]

                    q += prob * (reward + gamma * value[new_state])

                if max_Q is None or q - max_Q > 1e-8:
                    max_Q = q
                    max_act = action

            new_policy[state] = max_act

        if np.all(new_policy == policy):
            return value, new_policy, n_iters

        policy = new_policy.copy()
